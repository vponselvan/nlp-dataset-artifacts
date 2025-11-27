"""
Negation-Aware Contrastive Pair Generation

This script implements Step 1 of the Negation-Aware Contrastive Training strategy:
- Identifies examples containing negation words in the training set
- Generates contrastive pairs (positive/affirmative vs negative/negated versions)
- Marks examples for weighted loss during training (3x weight for negation)
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


# Comprehensive list of negation indicators
NEGATION_WORDS = {
    # Basic negation
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    # Contractions
    "n't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "cannot",
    "couldn't",
    "shouldn't",
    "mustn't",
    "haven't",
    "hasn't",
    "hadn't",
    "aren't",
    "isn't",
    "wasn't",
    "weren't",
    # Prefixes (as separate words)
    "without",
    "unless",
    "except",
    # Negative adverbs
    "hardly",
    "scarcely",
    "barely",
    "rarely",
    "seldom",
}

# Template-based negation transformations
NEGATION_TEMPLATES = [
    # Simple insertion patterns
    {
        "pattern": r"\b(is|was|are|were|will be|has been|have been)\b",
        "replacement": r"\1 not",
        "description": 'Insert "not" after be-verbs',
    },
    {
        "pattern": r"\b(did|does|do|can|could|should|would|may|might|must)\b",
        "replacement": r"\1 not",
        "description": 'Insert "not" after auxiliary verbs',
    },
    # Contraction patterns
    {
        "pattern": r"\bis\b",
        "replacement": "isn't",
        "description": 'Convert "is" to "isn\'t"',
    },
    {
        "pattern": r"\bwas\b",
        "replacement": "wasn't",
        "description": 'Convert "was" to "wasn\'t"',
    },
    {
        "pattern": r"\bare\b",
        "replacement": "aren't",
        "description": 'Convert "are" to "aren\'t"',
    },
    {
        "pattern": r"\bwere\b",
        "replacement": "weren't",
        "description": 'Convert "were" to "weren\'t"',
    },
    {
        "pattern": r"\bdid\b",
        "replacement": "didn't",
        "description": 'Convert "did" to "didn\'t"',
    },
]


def contains_negation(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains negation words.

    Returns:
        Tuple of (has_negation, list of negation words found)
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b|n\'t", text_lower)
    found_negations = [w for w in words if w in NEGATION_WORDS]
    return len(found_negations) > 0, found_negations


def create_negated_context(context: str, max_attempts: int = 3) -> str:
    """
    Create a negated version of the context using rule-based templates.

    Args:
        context: Original context text
        max_attempts: Maximum number of template attempts

    Returns:
        Negated context string
    """
    # Try different templates in random order
    templates = random.sample(
        NEGATION_TEMPLATES, min(max_attempts, len(NEGATION_TEMPLATES))
    )

    for template in templates:
        sentences = re.split(r"[.!?]", context)
        if not sentences:
            continue

        first_sentence = sentences[0]

        # Apply the negation template
        negated_sentence = re.sub(
            template["pattern"],
            template["replacement"],
            first_sentence,
            count=1,
            flags=re.IGNORECASE,
        )

        if negated_sentence != first_sentence:
            negated_context = negated_sentence + "." + ".".join(sentences[1:])
            return negated_context

    return f"It is not the case that {context[0].lower()}{context[1:]}"


def create_additive_negation(context: str, answer: str) -> str:
    """
    Create an additive negation sentence similar to AddSent attacks.

    Args:
        context: Original context
        answer: The answer entity/phrase

    Returns:
        Context with added negation sentence
    """
    negation_templates = [
        f"However, some sources claim it was not {answer}.",
        f"Some might argue that {answer} is incorrect.",
        f"It should be noted that {answer} is disputed.",
        f"Other records indicate it wasn't {answer}.",
        f"Despite popular belief, {answer} may not be accurate.",
        f"Some historians dispute that {answer} is correct.",
    ]

    # Add negation sentence at a natural break point
    sentences = re.split(r"([.!?])", context)
    if len(sentences) > 2:
        # Insert after first or second sentence
        insert_pos = random.choice([2, 4]) if len(sentences) > 4 else 2
        sentences.insert(insert_pos, " " + random.choice(negation_templates))

    return "".join(sentences)


def generate_contrastive_pairs(
    dataset_path: str,
    output_path: str,
    negation_weight: float = 3.0,
    augmentation_ratio: float = 0.3,
    seed: int = 42,
):
    """
    Generate negation-aware contrastive training pairs.

    Args:
        dataset_path: Path to original training dataset (JSONL)
        output_path: Path to save augmented dataset
        negation_weight: Weight multiplier for negation examples (default: 3x)
        augmentation_ratio: Ratio of examples to augment with negation (0-1)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load original dataset
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Statistics
    stats = {
        "total": len(examples),
        "already_negated": 0,
        "positive_examples": 0,
        "augmented_with_negation": 0,
        "additive_negation": 0,
        "transformative_negation": 0,
    }

    augmented_examples = []

    # Process each example
    for i, ex in enumerate(examples):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(examples)} examples...")

        # Add original example
        original_ex = dict(ex)

        # Check if already contains negation
        has_neg, neg_words = contains_negation(ex["context"])

        if has_neg:
            # Mark as high-weight example if already contains negation
            original_ex["loss_weight"] = negation_weight
            original_ex["is_negation_example"] = True
            original_ex["negation_type"] = "original"
            stats["already_negated"] += 1
        else:
            # Mark as regular weight
            original_ex["loss_weight"] = 1.0
            original_ex["is_negation_example"] = False
            stats["positive_examples"] += 1

        augmented_examples.append(original_ex)

        # Decide whether to augment this example with negation
        if not has_neg and random.random() < augmentation_ratio:
            # Get the first answer text
            answer_text = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

            if not answer_text:
                continue

            # Create TWO types of negation augmentations

            # Type 1: Additive Negation (like AddSent attack)
            additive_ex = dict(ex)
            additive_ex["id"] = f"{ex['id']}_neg_additive"
            additive_ex["context"] = create_additive_negation(
                ex["context"], answer_text
            )
            additive_ex["loss_weight"] = negation_weight
            additive_ex["is_negation_example"] = True
            additive_ex["negation_type"] = "additive"
            # Keep same answer to teach model to be robust to distractors
            augmented_examples.append(additive_ex)
            stats["additive_negation"] += 1

            # Type 2: Transformative Negation (modify original sentence)
            transformative_ex = dict(ex)
            transformative_ex["id"] = f"{ex['id']}_neg_transform"
            transformative_ex["context"] = create_negated_context(ex["context"])
            # Answer becomes unanswerable or needs to be negated
            transformative_ex["answers"] = {"text": [], "answer_start": []}
            transformative_ex["is_impossible"] = True
            transformative_ex["loss_weight"] = negation_weight
            transformative_ex["is_negation_example"] = True
            transformative_ex["negation_type"] = "transformative"
            augmented_examples.append(transformative_ex)
            stats["transformative_negation"] += 1

            stats["augmented_with_negation"] += 1

    # Save augmented dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in augmented_examples:
            f.write(json.dumps(ex) + "\n")

    # Save statistics
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate negation-aware contrastive training pairs"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input training dataset (JSONL format)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save augmented dataset"
    )
    parser.add_argument(
        "--negation-weight",
        type=float,
        default=3.0,
        help="Loss weight multiplier for negation examples (default: 3.0)",
    )
    parser.add_argument(
        "--augmentation-ratio",
        type=float,
        default=0.3,
        help="Ratio of positive examples to augment with negation (default: 0.3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_contrastive_pairs(
        dataset_path=args.input,
        output_path=args.output,
        negation_weight=args.negation_weight,
        augmentation_ratio=args.augmentation_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
