#!/usr/bin/env python3
"""
Augment adversarial training data with diverse perturbation types.
This helps prevent overfitting to AddSent-specific patterns.
"""

import json
import random
from pathlib import Path
import re


def load_jsonl(path):
    """Load JSONL file"""
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_jsonl(examples, path):
    """Save to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def create_paraphrase_attack(example):
    """
    Create paraphrased adversarial examples.
    Paraphrase the answer to create near-miss distractors.
    """
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]

    # Simple paraphrase templates
    paraphrases = [
        f"Some might argue it was {answer}, though this is debated.",
        f"According to certain sources, the answer could be {answer}.",
        f"While {answer} is mentioned, other interpretations exist.",
    ]

    adversarial_sentence = random.choice(paraphrases)

    return {
        **example,
        "context": context + " " + adversarial_sentence,
        "id": example["id"] + "_paraphrase",
    }


def create_entity_swap_attack(example):
    """
    Swap entities of the same type to create distractors.
    """
    context = example["context"]
    answer = example["answers"]["text"][0]

    # Extract entities (simple heuristic: capitalized words)
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", context)
    entities = [e for e in entities if e != answer and len(e) > 2]

    if entities:
        distractor = random.choice(entities)
        adversarial_sentence = f"However, some records indicate {distractor} instead."

        return {
            **example,
            "context": context + " " + adversarial_sentence,
            "id": example["id"] + "_entity_swap",
        }

    return None


def create_negation_attack(example):
    """
    Add negation patterns to create adversarial examples.
    """
    context = example["context"]
    answer = example["answers"]["text"][0]

    negation_templates = [
        f"It should be noted that {answer} did not participate in this event.",
        f"Despite speculation, {answer} was not involved.",
        f"Contrary to popular belief, {answer} is not the correct answer.",
    ]

    adversarial_sentence = random.choice(negation_templates)

    return {
        **example,
        "context": context + " " + adversarial_sentence,
        "id": example["id"] + "_negation",
    }


def create_numeric_attack(example):
    """
    Add misleading numbers to create adversarial examples.
    """
    context = example["context"]
    answer = example["answers"]["text"][0]

    # Check if answer contains numbers
    if re.search(r"\d", answer):
        # Generate fake numbers
        fake_numbers = [
            str(random.randint(1000, 9999)),
            str(random.randint(10, 99)),
            f"{random.randint(1, 50)},{random.randint(100, 999)}",
        ]

        fake_num = random.choice(fake_numbers)
        adversarial_sentence = f"Some sources cite {fake_num} as an alternative figure."

        return {
            **example,
            "context": context + " " + adversarial_sentence,
            "id": example["id"] + "_numeric",
        }

    return None


def augment_dataset(input_path, output_path, augmentation_ratio=0.5):
    """
    Augment dataset with diverse adversarial patterns.

    Args:
        input_path: Original AddSent training data
        output_path: Augmented output path
        augmentation_ratio: Ratio of examples to augment (0.5 = 50%)
    """
    print("=" * 60)
    print("Augmenting Adversarial Training Data")
    print("=" * 60)

    # Load original data
    print(f"\nLoading data from: {input_path}")
    examples = load_jsonl(input_path)
    print(f"  Loaded {len(examples)} examples")

    # Calculate how many to augment
    num_to_augment = int(len(examples) * augmentation_ratio)
    examples_to_augment = random.sample(examples, num_to_augment)

    print(f"\nAugmenting {num_to_augment} examples ({augmentation_ratio*100:.0f}%)")

    augmented = []
    attack_types = {"paraphrase": 0, "entity_swap": 0, "negation": 0, "numeric": 0}

    for ex in examples_to_augment:
        # Randomly choose augmentation type
        attack_type = random.choice(
            ["paraphrase", "entity_swap", "negation", "numeric"]
        )

        if attack_type == "paraphrase":
            aug_ex = create_paraphrase_attack(ex)
            if aug_ex:
                augmented.append(aug_ex)
                attack_types["paraphrase"] += 1

        elif attack_type == "entity_swap":
            aug_ex = create_entity_swap_attack(ex)
            if aug_ex:
                augmented.append(aug_ex)
                attack_types["entity_swap"] += 1

        elif attack_type == "negation":
            aug_ex = create_negation_attack(ex)
            if aug_ex:
                augmented.append(aug_ex)
                attack_types["negation"] += 1

        elif attack_type == "numeric":
            aug_ex = create_numeric_attack(ex)
            if aug_ex:
                augmented.append(aug_ex)
                attack_types["numeric"] += 1

    # Combine original + augmented
    final_data = examples + augmented
    random.shuffle(final_data)

    print(f"\nAugmentation breakdown:")
    print(f"  Paraphrase attacks: {attack_types['paraphrase']}")
    print(f"  Entity swap attacks: {attack_types['entity_swap']}")
    print(f"  Negation attacks: {attack_types['negation']}")
    print(f"  Numeric attacks: {attack_types['numeric']}")
    print(f"  Total augmented: {len(augmented)}")

    print(f"\nFinal dataset size: {len(final_data)} examples")
    print(f"  Original: {len(examples)} ({len(examples)/len(final_data)*100:.1f}%)")
    print(f"  Augmented: {len(augmented)} ({len(augmented)/len(final_data)*100:.1f}%)")

    # Save
    print(f"\nSaving to: {output_path}")
    save_jsonl(final_data, output_path)

    print("\n✅ Augmented dataset created successfully!")

    # Show sample
    print("\n" + "=" * 60)
    print("Sample Augmented Example")
    print("=" * 60)

    sample = random.choice(augmented)
    print(f"\nID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answers']['text'][0]}")
    print(f"Context (last 200 chars): ...{sample['context'][-200:]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment adversarial data with diverse attack types"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/addsent_train.jsonl",
        help="Path to AddSent training data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/addsent_train_augmented.jsonl",
        help="Output path for augmented dataset",
    )
    parser.add_argument(
        "--augmentation_ratio",
        type=float,
        default=0.5,
        help="Ratio of examples to augment (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    augment_dataset(args.input_path, args.output_path, args.augmentation_ratio)
