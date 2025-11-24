#!/usr/bin/env python3
"""
Negation-Aware Contrastive Trainer

This script implements Step 2 of the Negation-Aware Contrastive Training strategy:
- Custom Trainer with weighted loss function
- Multiplies loss by 3x for examples marked with negation
- Forces model to pay 3x more attention to negation cues
- Compatible with HuggingFace Trainer API

Usage:
    from negation_aware_trainer import NegationAwareTrainer

    trainer = NegationAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        ...
    )
    trainer.train()
"""

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class NegationAwareTrainer(Trainer):
    """
    Custom Trainer that implements weighted loss for negation-aware contrastive training.

    Key features:
    - Automatically detects 'loss_weight' field in training examples
    - Applies weight multiplier to loss for negation examples
    - Maintains standard Trainer functionality for all other operations
    """

    def __init__(self, *args, log_weight_stats: bool = True, **kwargs):
        """
        Initialize NegationAwareTrainer.

        Args:
            log_weight_stats: Whether to log weight statistics during training
            *args, **kwargs: Standard Trainer arguments
        """
        super().__init__(*args, **kwargs)
        self.log_weight_stats = log_weight_stats
        self.weight_stats = {
            "total_steps": 0,
            "weighted_steps": 0,
            "total_loss": 0.0,
            "weighted_loss": 0.0,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss for the model.

        This method overrides the default Trainer.compute_loss to:
        1. Extract loss weights from inputs (if present)
        2. Compute standard loss
        3. Apply per-example weights
        4. Return weighted average loss

        Args:
            model: The model being trained
            inputs: Dict containing input tensors
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)

        Returns:
            Loss tensor (and optionally model outputs)
        """
        # Extract loss weights if present (default to 1.0)
        loss_weights = inputs.pop("loss_weights", None)

        # Forward pass
        outputs = model(**inputs)

        # Extract loss from outputs
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            start_logits = outputs.get("start_logits")
            end_logits = outputs.get("end_logits")
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            start_logits = (
                outputs.start_logits if hasattr(outputs, "start_logits") else None
            )
            end_logits = outputs.end_logits if hasattr(outputs, "end_logits") else None

        # Apply loss weights if provided
        if loss_weights is not None:
            # Ensure weights are on the same device as loss
            if isinstance(loss_weights, torch.Tensor):
                loss_weights = loss_weights.to(loss.device)

                # Handle per-example loss for QA models
                if start_logits is not None and end_logits is not None:
                    # QA model: recompute per-example loss and apply weights
                    loss = self._compute_weighted_qa_loss(
                        start_logits, end_logits, inputs, loss_weights
                    )
                else:
                    # Classification model: apply weights directly
                    # Expand loss to per-example if needed
                    if loss.dim() == 0:
                        # Loss is already averaged, we need per-example losses
                        # This happens when loss is computed inside the model
                        # In this case, we approximate by scaling the batch loss
                        batch_size = loss_weights.size(0)
                        avg_weight = loss_weights.mean()
                        loss = loss * avg_weight
                    else:
                        # Per-example loss available
                        loss = loss * loss_weights
                        loss = loss.mean()

                # Update statistics
                self.weight_stats["total_steps"] += 1
                self.weight_stats["total_loss"] += loss.item()

                if (loss_weights > 1.0).any():
                    self.weight_stats["weighted_steps"] += 1
                    self.weight_stats["weighted_loss"] += loss.item()

                # Log statistics periodically
                if (
                    self.log_weight_stats
                    and self.weight_stats["total_steps"] % 100 == 0
                ):
                    logger.info(
                        f"Weight Stats - Step {self.weight_stats['total_steps']}: "
                        f"Weighted steps: {self.weight_stats['weighted_steps']}, "
                        f"Avg loss: {self.weight_stats['total_loss'] / self.weight_stats['total_steps']:.4f}, "
                        f"Avg weighted: {loss_weights.mean().item():.2f}x"
                    )

        return (loss, outputs) if return_outputs else loss

    def _compute_weighted_qa_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss for QA models.

        Args:
            start_logits: Start position logits [batch_size, seq_len]
            end_logits: End position logits [batch_size, seq_len]
            inputs: Input dict containing start_positions and end_positions
            loss_weights: Per-example loss weights [batch_size]

        Returns:
            Weighted average loss
        """
        start_positions = inputs.get("start_positions")
        end_positions = inputs.get("end_positions")

        if start_positions is None or end_positions is None:
            raise ValueError(
                "start_positions and end_positions must be in inputs for QA loss"
            )

        # Clamp positions to valid range
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute cross-entropy loss (without reduction)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        # Combine start and end losses
        total_loss = (start_loss + end_loss) / 2

        # Apply per-example weights
        weighted_loss = total_loss * loss_weights

        # Return mean
        return weighted_loss.mean()

    def log_weight_summary(self):
        """Log summary statistics of weight usage."""
        if self.weight_stats["total_steps"] > 0:
            logger.info("\n" + "=" * 70)
            logger.info("Negation-Aware Training Summary")
            logger.info("=" * 70)
            logger.info(f"Total training steps: {self.weight_stats['total_steps']}")
            logger.info(
                f"Steps with weighted examples: {self.weight_stats['weighted_steps']} "
                f"({self.weight_stats['weighted_steps']/self.weight_stats['total_steps']*100:.1f}%)"
            )
            logger.info(
                f"Average loss: {self.weight_stats['total_loss']/self.weight_stats['total_steps']:.4f}"
            )
            if self.weight_stats["weighted_steps"] > 0:
                logger.info(
                    f"Average weighted loss: "
                    f"{self.weight_stats['weighted_loss']/self.weight_stats['weighted_steps']:.4f}"
                )
            logger.info("=" * 70)


class NegationAwareQATrainer(NegationAwareTrainer):
    """
    Negation-Aware Trainer specifically for Question Answering tasks.

    Extends NegationAwareTrainer with QA-specific evaluation metrics.
    Compatible with the QuestionAnsweringTrainer from helpers.py.
    """

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        """
        Initialize QA-specific trainer.

        Args:
            eval_examples: Evaluation examples (unprocessed)
            post_process_function: Function to post-process predictions
            *args, **kwargs: Standard Trainer arguments
        """
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Evaluate with QA-specific metrics.

        This method is compatible with the SQuAD evaluation pipeline.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Compute predictions
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Post-process predictions if function provided
        if self.post_process_function is not None and eval_examples is not None:
            processed = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            # Update metrics with processed results
            if hasattr(processed, 'metrics'):
                output.metrics.update(processed.metrics)

        self.log(output.metrics)

        return output.metrics


def prepare_inputs_with_weights(examples, tokenizer, max_length=384):
    """
    Prepare inputs with loss weights for negation-aware training.

    This function extends the standard tokenization to include loss_weights.
    Compatible with the prepare_train_dataset_qa function from helpers.py.

    Args:
        examples: Batch of examples with 'loss_weight' field
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dict with tokenized inputs including loss_weights
    """
    # Extract loss weights (default to 1.0 if not present)
    loss_weights = [ex.get("loss_weight", 1.0) for ex in examples]

    # Standard tokenization (simplified - adapt to your tokenization pipeline)
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Add loss weights
    tokenized["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)

    return tokenized


if __name__ == "__main__":
    print("Negation-Aware Contrastive Trainer")
    print("=" * 70)
    print("\nThis module provides custom trainers for negation-aware training.")
    print("\nUsage:")
    print("  from negation_aware_trainer import NegationAwareQATrainer")
    print("  ")
    print("  trainer = NegationAwareQATrainer(")
    print("      model=model,")
    print("      args=training_args,")
    print("      train_dataset=train_dataset,")
    print("      eval_dataset=eval_dataset,")
    print("      eval_examples=eval_examples,")
    print("      tokenizer=tokenizer,")
    print("  )")
    print("  trainer.train()")
    print("  trainer.log_weight_summary()")
