#!/usr/bin/env python3
"""
Entity-Aware Contrastive Trainer

This script implements Step 3 of Entity-Aware Contrastive Training strategy:
- Custom Trainer with contrastive ranking loss
- Maximizes score difference between correct entity and hard negatives
- Uses weighted loss for entity-rich examples
- Compatible with HuggingFace Trainer API

Contrastive Loss:
    L_contrastive = -log(exp(S_correct) / Σ exp(S_distractor))

where S_correct is the score for the ground truth answer span,
and S_distractor are scores for hard negative entity spans.

Usage:
    from entity_aware_trainer import EntityAwareQATrainer

    trainer = EntityAwareQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        contrastive_weight=0.5,  # Balance between QA loss and contrastive loss
        ...
    )
    trainer.train()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class EntityAwareQATrainer(Trainer):
    """
    Custom Trainer that implements entity-aware contrastive learning for QA.

    Key features:
    - Computes standard QA loss for start/end positions
    - Adds contrastive ranking loss for hard negative entities
    - Applies weighted loss for entity-rich examples
    - Balances QA and contrastive objectives
    """

    def __init__(
        self,
        *args,
        contrastive_weight: float = 0.5,
        margin: float = 1.0,
        log_stats: bool = True,
        eval_examples=None,
        **kwargs,
    ):
        """
        Initialize EntityAwareQATrainer.

        Args:
            contrastive_weight: Weight for contrastive loss (0-1)
                0 = pure QA loss, 1 = pure contrastive loss
            margin: Margin for ranking loss
            log_stats: Whether to log statistics during training
            eval_examples: Evaluation examples (for QA evaluation)
            *args, **kwargs: Standard Trainer arguments
        """
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.log_stats = log_stats
        self.eval_examples = eval_examples

        self.stats = {
            "total_steps": 0,
            "contrastive_steps": 0,
            "avg_qa_loss": 0.0,
            "avg_contrastive_loss": 0.0,
            "avg_hard_negatives": 0.0,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined QA loss and contrastive loss.

        Args:
            model: The QA model
            inputs: Dict containing input tensors
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)

        Returns:
            Combined loss (and optionally model outputs)
        """
        # Extract entity metadata
        loss_weights = inputs.pop("loss_weights", None)
        hard_negatives = inputs.pop("hard_negatives", None)

        # Forward pass
        outputs = model(**inputs)

        # Standard QA loss
        qa_loss = outputs.loss

        # Apply weight to QA loss if provided
        if loss_weights is not None:
            if isinstance(loss_weights, torch.Tensor):
                loss_weights = loss_weights.to(qa_loss.device)
                avg_weight = loss_weights.mean()
                qa_loss = qa_loss * avg_weight

        # Compute contrastive loss if hard negatives provided
        contrastive_loss = None
        if hard_negatives is not None and len(hard_negatives) > 0:
            contrastive_loss = self._compute_contrastive_loss(
                outputs.start_logits,
                outputs.end_logits,
                inputs.get("start_positions"),
                inputs.get("end_positions"),
                hard_negatives,
                inputs.get("attention_mask"),
            )

        # Combine losses
        if contrastive_loss is not None:
            total_loss = (
                1 - self.contrastive_weight
            ) * qa_loss + self.contrastive_weight * contrastive_loss

            # Update statistics
            self.stats["contrastive_steps"] += 1
            self.stats["avg_contrastive_loss"] += contrastive_loss.item()
        else:
            total_loss = qa_loss

        # Update statistics
        self.stats["total_steps"] += 1
        self.stats["avg_qa_loss"] += qa_loss.item()

        # Log periodically
        if self.log_stats and self.stats["total_steps"] % 100 == 0:
            self._log_training_stats()

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_contrastive_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        hard_negatives: List[List[Dict]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive ranking loss for entity discrimination.

        The loss encourages the model to:
        1. Assign high scores to the correct answer span
        2. Assign low scores to hard negative entity spans

        Formula: L = -log(exp(S_correct) / (exp(S_correct) + Σ exp(S_neg)))

        Args:
            start_logits: Start position logits [batch_size, seq_len]
            end_logits: End position logits [batch_size, seq_len]
            start_positions: Ground truth start positions [batch_size]
            end_positions: Ground truth end positions [batch_size]
            hard_negatives: List of hard negative entities per example
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Contrastive loss scalar
        """
        batch_size = start_logits.size(0)
        device = start_logits.device

        losses = []
        num_hard_negatives = 0

        for b in range(batch_size):
            # Skip if no hard negatives for this example
            if not hard_negatives[b]:
                continue

            # Get correct answer score
            correct_start = start_positions[b].item()
            correct_end = end_positions[b].item()

            # Handle impossible/unanswerable cases
            if correct_start >= start_logits.size(1) or correct_end >= end_logits.size(
                1
            ):
                continue

            # Score for correct span: sum of start and end logits
            correct_score = start_logits[b, correct_start] + end_logits[b, correct_end]

            # Collect scores for hard negative spans
            negative_scores = []

            for hn in hard_negatives[b]:
                # Hard negative span positions (from token offsets)
                # Note: This is simplified - in practice you'd need proper token alignment
                hn_start = hn.get("start_token", hn.get("start", 0))
                hn_end = hn.get("end_token", hn.get("end", 0))

                # Ensure positions are within bounds
                if hn_start >= start_logits.size(1) or hn_end >= end_logits.size(1):
                    continue
                if hn_start < 0 or hn_end < 0:
                    continue

                # Score for negative span
                neg_score = start_logits[b, hn_start] + end_logits[b, hn_end]
                negative_scores.append(neg_score)

            if not negative_scores:
                continue

            # Compute contrastive loss using log-sum-exp for numerical stability
            # L = -log(exp(S_pos) / (exp(S_pos) + Σ exp(S_neg)))
            # = -S_pos + log(exp(S_pos) + Σ exp(S_neg))

            all_scores = torch.stack([correct_score] + negative_scores)

            # Log-sum-exp trick for numerical stability
            max_score = all_scores.max()
            log_sum_exp = max_score + torch.log(
                torch.sum(torch.exp(all_scores - max_score))
            )

            loss = -correct_score + log_sum_exp
            losses.append(loss)
            num_hard_negatives += len(negative_scores)

        if not losses:
            # No valid contrastive examples in batch
            return torch.tensor(0.0, device=device)

        # Average loss
        contrastive_loss = torch.stack(losses).mean()

        # Update statistics
        if num_hard_negatives > 0:
            self.stats["avg_hard_negatives"] += num_hard_negatives / len(losses)

        return contrastive_loss

    def _log_training_stats(self):
        """Log training statistics."""
        if self.stats["total_steps"] == 0:
            return

        avg_qa = self.stats["avg_qa_loss"] / self.stats["total_steps"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Training Statistics - Step {self.stats['total_steps']}")
        logger.info(f"{'='*70}")
        logger.info(f"Average QA Loss: {avg_qa:.4f}")

        if self.stats["contrastive_steps"] > 0:
            avg_contr = (
                self.stats["avg_contrastive_loss"] / self.stats["contrastive_steps"]
            )
            avg_hn = self.stats["avg_hard_negatives"] / self.stats["contrastive_steps"]

            logger.info(f"Average Contrastive Loss: {avg_contr:.4f}")
            logger.info(
                f"Contrastive steps: {self.stats['contrastive_steps']} / {self.stats['total_steps']}"
            )
            logger.info(f"Average hard negatives per example: {avg_hn:.2f}")
        else:
            logger.info("No contrastive loss computed yet")

        logger.info(f"{'='*70}\n")
    
    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Evaluate with QA-specific metrics.
        
        This method handles eval_examples for QA evaluation.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        
        try:
            # Compute predictions
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # Compute metrics if available
        if self.compute_metrics is not None and eval_examples is not None:
            from helpers import postprocess_qa_predictions
            
            # Post-process predictions
            eval_preds = postprocess_qa_predictions(
                eval_examples,
                eval_dataset,
                output.predictions
            )
            
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
            
            # Compute metrics
            metrics = self.compute_metrics(
                type('EvalPrediction', (), {'predictions': formatted_predictions, 'label_ids': references})()
            )
            
            # Prefix all keys
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            
            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def log_final_summary(self):
        """Log final training summary."""
        if self.stats["total_steps"] == 0:
            return

        logger.info("\n" + "=" * 70)
        logger.info("Entity-Aware Training Summary")
        logger.info("=" * 70)
        logger.info(f"Total training steps: {self.stats['total_steps']}")
        logger.info(
            f"Steps with contrastive loss: {self.stats['contrastive_steps']} "
            f"({self.stats['contrastive_steps']/self.stats['total_steps']*100:.1f}%)"
        )

        avg_qa = self.stats["avg_qa_loss"] / self.stats["total_steps"]
        logger.info(f"Average QA loss: {avg_qa:.4f}")

        if self.stats["contrastive_steps"] > 0:
            avg_contr = (
                self.stats["avg_contrastive_loss"] / self.stats["contrastive_steps"]
            )
            avg_hn = self.stats["avg_hard_negatives"] / self.stats["contrastive_steps"]

            logger.info(f"Average contrastive loss: {avg_contr:.4f}")
            logger.info(f"Average hard negatives: {avg_hn:.2f}")

        logger.info(f"Contrastive weight: {self.contrastive_weight}")
        logger.info("=" * 70)


def prepare_inputs_with_hard_negatives(examples, tokenizer, max_length=384):
    """
    Prepare inputs with hard negative metadata for entity-aware training.

    This function extends standard tokenization to include:
    - loss_weights for entity-rich examples
    - hard_negatives with token positions

    Args:
        examples: Batch of examples with 'hard_negatives' field
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dict with tokenized inputs including entity metadata
    """
    # Standard tokenization
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # Extract offsets for mapping character positions to tokens
    offset_mapping = tokenized.pop("offset_mapping")

    # Process hard negatives to get token positions
    batch_hard_negatives = []
    loss_weights = []

    for idx, ex in enumerate(examples):
        ex_hard_negatives = []

        if "hard_negatives" in ex and ex["hard_negatives"]:
            # Map character positions to token positions
            for hn in ex["hard_negatives"]:
                char_start = hn.get("start", 0)
                char_end = hn.get("end", 0)

                # Find token positions
                token_start = None
                token_end = None

                for token_idx, (offset_start, offset_end) in enumerate(
                    offset_mapping[idx]
                ):
                    if offset_start <= char_start < offset_end:
                        token_start = token_idx
                    if offset_start < char_end <= offset_end:
                        token_end = token_idx
                        break

                if token_start is not None and token_end is not None:
                    ex_hard_negatives.append(
                        {
                            "text": hn["text"],
                            "start_token": token_start,
                            "end_token": token_end,
                            "type": hn.get("type", "UNKNOWN"),
                        }
                    )

        batch_hard_negatives.append(ex_hard_negatives)

        # Set loss weight
        loss_weight = ex.get("loss_weight", 1.0)
        loss_weights.append(loss_weight)

    # Add to tokenized inputs
    tokenized["hard_negatives"] = batch_hard_negatives
    tokenized["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)

    return tokenized


if __name__ == "__main__":
    print("Entity-Aware Contrastive Trainer")
    print("=" * 70)
    print("\nThis module provides custom trainers for entity-aware training.")
    print("\nKey Features:")
    print("  - Contrastive ranking loss for entity discrimination")
    print("  - Weighted loss for entity-rich examples")
    print("  - Compatible with HuggingFace Trainer API")
    print("\nUsage:")
    print("  from entity_aware_trainer import EntityAwareQATrainer")
    print("  ")
    print("  trainer = EntityAwareQATrainer(")
    print("      model=model,")
    print("      args=training_args,")
    print("      train_dataset=train_dataset,")
    print("      contrastive_weight=0.5,  # Balance QA and contrastive loss")
    print("      margin=1.0,")
    print("  )")
    print("  trainer.train()")
    print("  trainer.log_final_summary()")
