"""
Train ELECTRA with Entity-Aware Contrastive Learning

This script implements Step 3 of Entity-Aware Contrastive Training strategy:
- Loads entity-augmented training data with hard negatives
- Uses custom EntityAwareQATrainer with contrastive ranking loss
- Fine-tunes ELECTRA-base with entity discrimination objective
- Evaluates on both clean (SQuAD) and adversarial (AddSent) data
"""

import sys
import os

# Add parent directory to path to import helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    default_data_collator,
)
import evaluate

from helpers import prepare_train_dataset_qa, prepare_validation_dataset_qa

# Add scripts directory to path for importing entity_aware_trainer
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from entity_aware_trainer import EntityAwareQATrainer


def load_entity_aware_dataset(path: str):
    """
    Load dataset with entity metadata (hard negatives, loss weights).

    Args:
        path: Path to JSONL file with entity metadata

    Returns:
        HuggingFace Dataset object
    """
    print(f"Loading dataset from {path}...")
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)

    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(examples)

    # Print statistics
    has_hard_negs = [ex for ex in examples if ex.get("hard_negatives")]
    entity_examples = [ex for ex in examples if ex.get("is_entity_example", False)]
    augmented = [ex for ex in examples if "entity_sub" in ex.get("id", "")]

    total_hard_negs = sum(len(ex.get("hard_negatives", [])) for ex in has_hard_negs)    

    if has_hard_negs:
        avg_hard_negs = total_hard_negs / len(has_hard_negs)
        print(f"  Average hard negatives per example: {avg_hard_negs:.2f}")

    # Entity type distribution
    entity_types = {}
    for ex in examples:
        ent_type = ex.get("answer_entity_type", "UNKNOWN")
        entity_types[ent_type] = entity_types.get(ent_type, 0) + 1

    if entity_types:
        print(f"\nEntity Type Distribution:")
        for ent_type, count in sorted(entity_types.items(), key=lambda x: -x[1])[:5]:
            print(f"  {ent_type}: {count} ({count/len(examples)*100:.1f}%)")

    return dataset


def prepare_train_dataset_with_entities(examples, tokenizer):
    """
    Prepare training dataset with entity metadata.

    Extends standard prepare_train_dataset_qa to include:
    - loss_weights
    - hard_negatives with token positions
    Handles the case where one example can produce multiple features.
    """
    # First, do the standard tokenization which handles overflow
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Get sample mapping before it's used
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    
    # Now call the standard function which will pop sample_mapping and process it
    tokenized = prepare_train_dataset_qa(examples, tokenizer)
    
    # Add entity metadata using the sample mapping we saved
    if "loss_weight" in examples:
        # Map each feature to its original example's loss weight
        loss_weights = [examples["loss_weight"][sample_idx] for sample_idx in sample_mapping]
        tokenized["loss_weights"] = loss_weights
    else:
        # Default weight of 1.0 for all examples
        tokenized["loss_weights"] = [1.0] * len(tokenized["input_ids"])

    # Add hard negatives (replicate for overflow tokens)
    if "hard_negatives" in examples:
        hard_negatives = [examples["hard_negatives"][sample_idx] for sample_idx in sample_mapping]
        tokenized["hard_negatives"] = hard_negatives

    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description="Train with Entity-Aware Contrastive Learning"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to entity-augmented training data (JSONL)",
    )
    parser.add_argument(
        "--eval-squad",
        type=str,
        default="../data/squad.jsonl",
        help="Path to clean SQuAD evaluation data",
    )
    parser.add_argument(
        "--eval-addsent",
        type=str,
        default="../data/addsent_eval.jsonl",
        help="Path to adversarial AddSent evaluation data",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="google/electra-base-discriminator",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained model",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.5,
        help="Weight for contrastive loss (0-1, default: 0.5)",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    train_dataset = load_entity_aware_dataset(args.train_data)
    eval_squad = datasets.load_dataset(
        "json", data_files=args.eval_squad, split="train"
    )
    eval_addsent = datasets.load_dataset(
        "json", data_files=args.eval_addsent, split="train"
    )

    train_dataset_tokenized = train_dataset.map(
        lambda ex: prepare_train_dataset_with_entities(ex, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )

    eval_squad_tokenized = eval_squad.map(
        lambda ex: prepare_validation_dataset_qa(ex, tokenizer),
        batched=True,
        remove_columns=eval_squad.column_names,
        desc="Tokenizing SQuAD eval",
    )

    eval_addsent_tokenized = eval_addsent.map(
        lambda ex: prepare_validation_dataset_qa(ex, tokenizer),
        batched=True,
        remove_columns=eval_addsent.column_names,
        desc="Tokenizing AddSent eval",
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,  # Disable for now due to metric issues
        seed=args.seed,
        fp16=True,  # Use mixed precision
        report_to="none",
    )

    # Setup evaluation metric
    metric = evaluate.load("squad")

    def compute_metrics(eval_preds):
        return metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids
        )

    trainer = EntityAwareQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_squad_tokenized,
        eval_examples=eval_squad,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        contrastive_weight=args.contrastive_weight,
        log_stats=True,
    )

    train_result = trainer.train()

    # Log summary
    trainer.log_final_summary()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    squad_metrics = trainer.evaluate(
        eval_dataset=eval_squad_tokenized,
        eval_examples=eval_squad,
        metric_key_prefix="eval_squad",
    )

    # Create temporary trainer for AddSent evaluation
    addsent_trainer = EntityAwareQATrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_addsent_tokenized,
        eval_examples=eval_addsent,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        contrastive_weight=args.contrastive_weight,
    )

    addsent_metrics = addsent_trainer.evaluate(metric_key_prefix="eval_addsent")

    all_metrics = {
        "train": metrics,
        "squad": squad_metrics,
        "addsent": addsent_metrics,
        "config": {
            "model": args.model,
            "train_data": args.train_data,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "contrastive_weight": args.contrastive_weight,
        },
    }

    results_path = Path(args.output_dir) / "entity_aware_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
