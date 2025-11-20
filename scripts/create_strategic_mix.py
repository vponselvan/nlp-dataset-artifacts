#!/usr/bin/env python3
"""
Strategic Re-mixing: Combine best of both worlds
- Use original AddSent for adversarial robustness (80-20 ratio)
- Add augmented examples for generalization (small percentage)
- This balances adversarial performance + clean performance
"""

import json
import random
from pathlib import Path


def load_jsonl(path):
    """Load JSONL file"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_jsonl(examples, path):
    """Save to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def strategic_remix(
    squad_path, addsent_original_path, addsent_augmented_path, output_path, seed=42
):
    """
    Create strategically mixed dataset:
    - 80% SQuAD (clean data foundation)
    - 15% Original AddSent (adversarial robustness)
    - 5% Augmented AddSent (generalization)

    Total adversarial: 20% (matching best ratio)
    But split to get benefits of both datasets
    """
    random.seed(seed)

    print("=" * 60)
    print("Strategic Dataset Re-mixing")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    squad = load_jsonl(squad_path)
    addsent_orig = load_jsonl(addsent_original_path)
    addsent_aug = load_jsonl(addsent_augmented_path)

    # Remove original examples from augmented to avoid duplicates
    addsent_aug_only = [
        ex
        for ex in addsent_aug
        if ex["id"].endswith(("_paraphrase", "_entity_swap", "_negation", "_numeric"))
    ]

    print(f"  SQuAD: {len(squad)} examples")
    print(f"  AddSent (original): {len(addsent_orig)} examples")
    print(f"  AddSent (augmented only): {len(addsent_aug_only)} examples")

    # Calculate target sizes
    total_target = 10570  # Same as original 80-20
    squad_target = int(total_target * 0.80)
    addsent_orig_target = int(total_target * 0.15)
    addsent_aug_target = int(total_target * 0.05)

    print(f"\nTarget distribution:")
    print(f"  SQuAD: {squad_target} (80%)")
    print(f"  AddSent original: {addsent_orig_target} (15%)")
    print(f"  AddSent augmented: {addsent_aug_target} (5%)")
    print(f"  Total: {total_target}")

    # Sample from each
    squad_sampled = random.sample(squad, min(squad_target, len(squad)))
    addsent_orig_sampled = random.sample(
        addsent_orig, min(addsent_orig_target, len(addsent_orig))
    )
    addsent_aug_sampled = random.sample(
        addsent_aug_only, min(addsent_aug_target, len(addsent_aug_only))
    )

    # Combine and shuffle
    combined = squad_sampled + addsent_orig_sampled + addsent_aug_sampled
    random.shuffle(combined)

    print(f"\nActual distribution:")
    print(
        f"  SQuAD: {len(squad_sampled)} ({len(squad_sampled)/len(combined)*100:.1f}%)"
    )
    print(
        f"  AddSent original: {len(addsent_orig_sampled)} "
        f"({len(addsent_orig_sampled)/len(combined)*100:.1f}%)"
    )
    print(
        f"  AddSent augmented: {len(addsent_aug_sampled)} "
        f"({len(addsent_aug_sampled)/len(combined)*100:.1f}%)"
    )
    print(
        f"  Total adversarial: {len(addsent_orig_sampled) + len(addsent_aug_sampled)} "
        f"({(len(addsent_orig_sampled) + len(addsent_aug_sampled))/len(combined)*100:.1f}%)"
    )
    print(f"  Final size: {len(combined)}")

    # Save
    print(f"\nSaving to: {output_path}")
    save_jsonl(combined, output_path)

    print("\n✅ Strategic remix completed!")
    print("\nRationale:")
    print("  - Original AddSent (15%) maintains adversarial robustness")
    print("  - Augmented AddSent (5%) adds generalization")
    print("  - Total 20% adversarial matches optimal ratio")
    print("  - Should achieve: high adversarial EM + good clean performance")


if __name__ == "__main__":
    strategic_remix(
        squad_path="./data/squad_train.jsonl",
        addsent_original_path="./data/addsent_train.jsonl",
        addsent_augmented_path="./data/addsent_train_augmented.jsonl",
        output_path="./data/mixed_training_strategic.jsonl",
        seed=42,
    )
