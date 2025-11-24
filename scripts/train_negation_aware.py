#!/usr/bin/env python3
"""
Train ELECTRA with Negation-Aware Contrastive Learning

This script implements Step 3 of the Negation-Aware Contrastive Training strategy:
- Loads negation-augmented training data
- Uses custom NegationAwareQATrainer with weighted loss
- Fine-tunes ELECTRA-base model with 3x weight on negation examples
- Evaluates on both clean (SQuAD) and adversarial (AddSent) data

Goal: Reduce "Negation Confusion" errors from 40.4% baseline
Expected improvement: +10-15% on negation-specific examples
"""

import sys
sys.path.append('..')

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
from negation_aware_trainer import NegationAwareQATrainer


def load_negation_aware_dataset(path: str):
    """
    Load dataset with loss_weight metadata.
    
    Args:
        path: Path to JSONL file with loss_weight field
        
    Returns:
        HuggingFace Dataset object
    """
    print(f"Loading dataset from {path}...")
    examples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(examples)
    
    # Print statistics
    has_weights = [ex for ex in examples if 'loss_weight' in ex]
    weighted_examples = [ex for ex in examples if ex.get('loss_weight', 1.0) > 1.0]
    negation_examples = [ex for ex in examples if ex.get('is_negation_example', False)]
    
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Examples with loss_weight: {len(has_weights)}")
    print(f"  Weighted examples (>1.0): {len(weighted_examples)} ({len(weighted_examples)/len(examples)*100:.1f}%)")
    print(f"  Negation examples: {len(negation_examples)} ({len(negation_examples)/len(examples)*100:.1f}%)")
    
    if weighted_examples:
        avg_weight = sum(ex.get('loss_weight', 1.0) for ex in weighted_examples) / len(weighted_examples)
        print(f"  Average weight for weighted examples: {avg_weight:.2f}x")
    
    return dataset


def prepare_train_dataset_with_weights(examples, tokenizer):
    """
    Prepare training dataset with loss weights.
    
    Extends standard prepare_train_dataset_qa to include loss_weights.
    """
    # Use standard tokenization
    tokenized = prepare_train_dataset_qa(examples, tokenizer)
    
    # Add loss weights
    if 'loss_weight' in examples:
        tokenized['loss_weights'] = examples['loss_weight']
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description='Train with Negation-Aware Contrastive Learning'
    )
    
    # Data arguments
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to negation-augmented training data (JSONL)'
    )
    parser.add_argument(
        '--eval-squad',
        type=str,
        default='../data/squad.jsonl',
        help='Path to clean SQuAD evaluation data'
    )
    parser.add_argument(
        '--eval-addsent',
        type=str,
        default='../data/addsent_eval.jsonl',
        help='Path to adversarial AddSent evaluation data'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='google/electra-base-discriminator',
        help='Base model to fine-tune'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for trained model'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Warmup ratio'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=384,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Negation-Aware Contrastive Training")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print()
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    
    # Make tensors contiguous (ELECTRA-specific fix)
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print()
    
    # Load datasets
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    
    train_dataset = load_negation_aware_dataset(args.train_data)
    eval_squad = datasets.load_dataset('json', data_files=args.eval_squad, split='train')
    eval_addsent = datasets.load_dataset('json', data_files=args.eval_addsent, split='train')
    
    print(f"\nEvaluation datasets:")
    print(f"  SQuAD: {len(eval_squad)} examples")
    print(f"  AddSent: {len(eval_addsent)} examples")
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_dataset_tokenized = train_dataset.map(
        lambda ex: prepare_train_dataset_with_weights(ex, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    
    eval_squad_tokenized = eval_squad.map(
        lambda ex: prepare_validation_dataset_qa(ex, tokenizer),
        batched=True,
        remove_columns=eval_squad.column_names,
        desc="Tokenizing SQuAD eval"
    )
    
    eval_addsent_tokenized = eval_addsent.map(
        lambda ex: prepare_validation_dataset_qa(ex, tokenizer),
        batched=True,
        remove_columns=eval_addsent.column_names,
        desc="Tokenizing AddSent eval"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_exact_match",
        seed=args.seed,
        fp16=True,  # Use mixed precision for faster training
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Setup evaluation metric
    metric = evaluate.load('squad')
    
    def compute_metrics(eval_preds):
        return metric.compute(
            predictions=eval_preds.predictions,
            references=eval_preds.label_ids
        )
    
    # Initialize custom trainer
    print("\n" + "=" * 80)
    print("Initializing Negation-Aware Trainer")
    print("=" * 80)
    
    trainer = NegationAwareQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_squad_tokenized,  # Primary eval on SQuAD
        eval_examples=eval_squad,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        log_weight_stats=True,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    train_result = trainer.train()
    
    # Log weight statistics
    trainer.log_weight_summary()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on SQuAD
    print("\n" + "=" * 80)
    print("Evaluating on SQuAD (Clean)")
    print("=" * 80)
    
    squad_metrics = trainer.evaluate(
        eval_dataset=eval_squad_tokenized,
        eval_examples=eval_squad,
        metric_key_prefix="eval_squad"
    )
    
    print(f"\nSQuAD Results:")
    print(f"  Exact Match: {squad_metrics['eval_squad_exact_match']:.2f}")
    print(f"  F1: {squad_metrics['eval_squad_f1']:.2f}")
    
    # Evaluate on AddSent
    print("\n" + "=" * 80)
    print("Evaluating on AddSent (Adversarial)")
    print("=" * 80)
    
    # Create temporary trainer for AddSent evaluation
    addsent_trainer = NegationAwareQATrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_addsent_tokenized,
        eval_examples=eval_addsent,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    addsent_metrics = addsent_trainer.evaluate(
        metric_key_prefix="eval_addsent"
    )
    
    print(f"\nAddSent Results:")
    print(f"  Exact Match: {addsent_metrics['eval_addsent_exact_match']:.2f}")
    print(f"  F1: {addsent_metrics['eval_addsent_f1']:.2f}")
    
    # Save all metrics
    print("\nSaving evaluation results...")
    all_metrics = {
        'train': metrics,
        'squad': squad_metrics,
        'addsent': addsent_metrics,
        'config': {
            'model': args.model,
            'train_data': args.train_data,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
        }
    }
    
    results_path = Path(args.output_dir) / 'negation_aware_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}")
    print(f"\nFinal Performance:")
    print(f"  SQuAD EM:   {squad_metrics['eval_squad_exact_match']:.2f}%")
    print(f"  AddSent EM: {addsent_metrics['eval_addsent_exact_match']:.2f}%")
    print(f"  Drop: {squad_metrics['eval_squad_exact_match'] - addsent_metrics['eval_addsent_exact_match']:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
