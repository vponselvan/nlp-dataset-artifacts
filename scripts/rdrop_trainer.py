"""
R-Drop: Regularized Dropout for Neural Networks
Paper: https://arxiv.org/abs/2106.14448

This implementation adds consistency regularization to prevent overfitting
to adversarial training data. It's particularly effective for the 70-30+ ratios
that showed catastrophic performance collapse.

Expected improvement: +3-5% EM on adversarial data
"""

import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss


def compute_kl_loss(p_logits, q_logits, pad_mask=None):
    """
    Compute KL divergence loss between two distributions.

    Args:
        p_logits: First set of logits (batch_size, seq_len)
        q_logits: Second set of logits (batch_size, seq_len)
        pad_mask: Optional mask for padded positions (batch_size, seq_len)

    Returns:
        KL divergence loss (scalar)
    """
    p_loss = F.kl_div(
        F.log_softmax(p_logits, dim=-1), F.softmax(q_logits, dim=-1), reduction="none"
    )

    q_loss = F.kl_div(
        F.log_softmax(q_logits, dim=-1), F.softmax(p_logits, dim=-1), reduction="none"
    )

    # Symmetric KL divergence
    loss = (p_loss + q_loss) / 2

    # Apply mask if provided
    if pad_mask is not None:
        loss = loss * pad_mask.unsqueeze(-1)
        loss = loss.sum() / pad_mask.sum()
    else:
        loss = loss.mean()

    return loss


def rdrop_training_step(model, batch, alpha=5.0):
    """
    Single training step with R-Drop regularization.

    Args:
        model: Question answering model
        batch: Input batch with keys: input_ids, attention_mask, start_positions, end_positions
        alpha: Weight for consistency loss (default: 5.0 as in paper)

    Returns:
        Dictionary with:
            - total_loss: Combined loss (task + consistency)
            - task_loss: Original QA loss
            - consistency_loss: R-Drop consistency loss
    """
    # Ensure model is in training mode (dropout enabled)
    model.train()

    # Forward pass 1: First dropout sample
    outputs1 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        start_positions=batch["start_positions"],
        end_positions=batch["end_positions"],
    )

    # Forward pass 2: Second dropout sample (different dropout mask)
    outputs2 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        start_positions=batch["start_positions"],
        end_positions=batch["end_positions"],
    )

    # Task losses (standard QA loss)
    task_loss = (outputs1.loss + outputs2.loss) / 2

    # Consistency losses (KL divergence between two predictions)
    # For both start and end positions
    start_kl_loss = compute_kl_loss(
        outputs1.start_logits, outputs2.start_logits, pad_mask=batch["attention_mask"]
    )

    end_kl_loss = compute_kl_loss(
        outputs1.end_logits, outputs2.end_logits, pad_mask=batch["attention_mask"]
    )

    consistency_loss = (start_kl_loss + end_kl_loss) / 2

    # Combined loss
    total_loss = task_loss + alpha * consistency_loss

    return {
        "total_loss": total_loss,
        "task_loss": task_loss.item(),
        "consistency_loss": consistency_loss.item(),
    }


class RDropTrainer:
    """
    Wrapper trainer that implements R-Drop regularization.

    Usage:
        trainer = RDropTrainer(model, optimizer, alpha=5.0)

        for batch in train_dataloader:
            loss_dict = trainer.training_step(batch)
            print(f"Loss: {loss_dict['total_loss']:.4f}")
    """

    def __init__(self, model, optimizer, alpha=5.0, max_grad_norm=1.0):
        """
        Args:
            model: Question answering model
            optimizer: PyTorch optimizer
            alpha: Weight for consistency loss (default: 5.0)
            max_grad_norm: Gradient clipping threshold (default: 1.0)
        """
        self.model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def training_step(self, batch):
        """
        Execute one training step with R-Drop.

        Args:
            batch: Input batch

        Returns:
            Dictionary with losses
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute R-Drop loss
        loss_dict = rdrop_training_step(self.model, batch, self.alpha)

        # Backward pass
        loss_dict["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Optimizer step
        self.optimizer.step()

        self.global_step += 1

        return loss_dict

    def eval_step(self, batch):
        """
        Evaluation step (no R-Drop, just standard forward pass).

        Args:
            batch: Input batch

        Returns:
            Model outputs
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch.get("start_positions"),
                end_positions=batch.get("end_positions"),
            )

        return outputs


# Example usage in training script
def example_training_loop():
    """
    Example showing how to integrate R-Drop into your training loop.
    """
    from transformers import AutoModelForQuestionAnswering, AdamW
    from torch.utils.data import DataLoader

    # Initialize model and optimizer
    model = AutoModelForQuestionAnswering.from_pretrained(
        "google/electra-base-discriminator"
    )
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Create R-Drop trainer
    trainer = RDropTrainer(
        model=model, optimizer=optimizer, alpha=5.0  # Recommended: 5.0 for QA tasks
    )

    # Training loop
    # train_dataloader = ...  # Your data loader

    # for epoch in range(num_epochs):
    #     model.train()
    #
    #     for batch in train_dataloader:
    #         # Move batch to device
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #
    #         # Training step with R-Drop
    #         loss_dict = trainer.training_step(batch)
    #
    #         # Logging
    #         if trainer.global_step % 100 == 0:
    #             print(f"Step {trainer.global_step}: "
    #                   f"Total Loss: {loss_dict['total_loss']:.4f}, "
    #                   f"Task Loss: {loss_dict['task_loss']:.4f}, "
    #                   f"Consistency Loss: {loss_dict['consistency_loss']:.4f}")
    #
    #     # Validation
    #     model.eval()
    #     for batch in val_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = trainer.eval_step(batch)
    #         # Compute metrics...


if __name__ == "__main__":
    print("R-Drop Implementation for Adversarial QA Training")
    print("=" * 60)
    print()
    print("This module provides R-Drop regularization to prevent")
    print("overfitting to adversarial training data.")
    print()
    print("Key Benefits:")
    print("  - Reduces overfitting at higher adversarial ratios (70-30+)")
    print("  - Improves generalization to unseen adversarial examples")
    print("  - Expected improvement: +3-5% EM on adversarial data")
    print()
    print("Usage:")
    print("  from rdrop_trainer import RDropTrainer")
    print("  trainer = RDropTrainer(model, optimizer, alpha=5.0)")
    print("  loss_dict = trainer.training_step(batch)")
    print()
    print("Recommended hyperparameters:")
    print("  - alpha=5.0 for QA tasks")
    print("  - alpha=1.0 for classification tasks")
    print("  - max_grad_norm=1.0 (gradient clipping)")
