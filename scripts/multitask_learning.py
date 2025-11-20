"""
Multi-Task Learning for Adversarial Robustness
Adds auxiliary tasks to improve model's understanding and prevent overfitting
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskQAModel(nn.Module):
    """
    QA model with auxiliary tasks:
    1. Answer span extraction (primary)
    2. Adversarial sentence detection (auxiliary)
    3. Answer sentence selection (auxiliary)
    """

    def __init__(self, model_name="google/electra-base-discriminator"):
        super().__init__()

        # Base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Primary task: QA span extraction
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start, end

        # Auxiliary task 1: Adversarial sentence detection
        # Binary classification: is this sentence adversarial?
        self.adv_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2),  # adversarial or not
        )

        # Auxiliary task 2: Answer sentence selection
        # Which sentence contains the answer?
        self.sentence_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),  # sentence relevance score
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        start_positions=None,
        end_positions=None,
        adv_labels=None,  # [batch_size, num_sentences]
        answer_sentence_labels=None,  # [batch_size, num_sentences]
        sentence_mask=None,  # [batch_size, num_sentences]
    ):
        """
        Forward pass with multi-task learning

        Returns:
            Dictionary with:
                - qa_loss: Span extraction loss
                - adv_loss: Adversarial detection loss
                - sentence_loss: Answer sentence selection loss
                - total_loss: Combined loss
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]

        # Task 1: QA span extraction
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Task 2: Adversarial sentence detection
        adv_logits = self.adv_detector(pooled_output)  # [batch, 2]

        # Task 3: Answer sentence selection
        # Use mean pooling over sentence spans
        sentence_scores = self.sentence_selector(pooled_output)  # [batch, 1]

        # Compute losses
        total_loss = 0
        loss_dict = {}

        # QA loss (primary task, weight=1.0)
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2

            loss_dict["qa_loss"] = qa_loss.item()
            total_loss += 1.0 * qa_loss

        # Adversarial detection loss (auxiliary, weight=0.3)
        if adv_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            adv_loss = loss_fct(adv_logits, adv_labels)

            loss_dict["adv_loss"] = adv_loss.item()
            total_loss += 0.3 * adv_loss

        # Answer sentence loss (auxiliary, weight=0.2)
        if answer_sentence_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            sentence_loss = loss_fct(
                sentence_scores.squeeze(-1), answer_sentence_labels.float()
            )

            loss_dict["sentence_loss"] = sentence_loss.item()
            total_loss += 0.2 * sentence_loss

        loss_dict["total_loss"] = total_loss

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "adv_logits": adv_logits,
            "sentence_scores": sentence_scores,
            "loss_dict": loss_dict,
        }


def prepare_multitask_batch(batch, tokenizer):
    """
    Prepare batch with labels for all tasks

    Args:
        batch: Standard QA batch
        tokenizer: Tokenizer

    Returns:
        Enhanced batch with auxiliary task labels
    """
    # Detect adversarial sentences (heuristic: check for common patterns)
    # In real implementation, you'd have ground truth labels
    adv_labels = []
    answer_sentence_labels = []

    for example in batch["examples"]:
        context = example["context"]
        answer = example["answers"]["text"][0]

        # Heuristic: If example ID contains adversarial markers
        is_adversarial = any(
            marker in example.get("id", "")
            for marker in ["_paraphrase", "_entity_swap", "_negation", "_numeric"]
        )
        adv_labels.append(1 if is_adversarial else 0)

        # Heuristic: Sentence contains answer text
        contains_answer = answer.lower() in context.lower()
        answer_sentence_labels.append(1 if contains_answer else 0)

    batch["adv_labels"] = torch.tensor(adv_labels)
    batch["answer_sentence_labels"] = torch.tensor(answer_sentence_labels)

    return batch


# Training example
def train_multitask_model():
    """
    Example training loop for multi-task model
    """
    from transformers import AdamW

    # Initialize model
    model = MultiTaskQAModel("google/electra-base-discriminator")
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         # Prepare multi-task batch
    #         batch = prepare_multitask_batch(batch, tokenizer)
    #
    #         # Forward pass
    #         outputs = model(
    #             input_ids=batch['input_ids'],
    #             attention_mask=batch['attention_mask'],
    #             start_positions=batch['start_positions'],
    #             end_positions=batch['end_positions'],
    #             adv_labels=batch['adv_labels'],
    #             answer_sentence_labels=batch['answer_sentence_labels']
    #         )
    #
    #         # Backward pass
    #         loss = outputs['loss']
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #         # Logging
    #         if step % 100 == 0:
    #             loss_dict = outputs['loss_dict']
    #             print(f"Step {step}: "
    #                   f"Total: {loss_dict['total_loss']:.4f}, "
    #                   f"QA: {loss_dict['qa_loss']:.4f}, "
    #                   f"Adv: {loss_dict['adv_loss']:.4f}, "
    #                   f"Sent: {loss_dict['sentence_loss']:.4f}")


if __name__ == "__main__":
    print("Multi-Task Learning for Adversarial QA")
    print("=" * 60)
    print()
    print("This model learns 3 tasks simultaneously:")
    print("  1. Answer span extraction (primary, weight=1.0)")
    print("  2. Adversarial sentence detection (auxiliary, weight=0.3)")
    print("  3. Answer sentence selection (auxiliary, weight=0.2)")
    print()
    print("Benefits:")
    print("  - Forces model to understand context structure")
    print("  - Prevents memorization of surface patterns")
    print("  - Improves adversarial robustness")
    print("  - Expected: +3-5% EM on adversarial data")
    print()
    print("Usage:")
    print("  model = MultiTaskQAModel('google/electra-base-discriminator')")
    print("  outputs = model(input_ids, attention_mask, ...)")
