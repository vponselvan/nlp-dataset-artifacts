#!/usr/bin/env python3
"""
Entity-Aware Contrastive Pair Generation

This script implements Step 1-2 of Entity-Aware Contrastive Training strategy:
- Extracts entities from contexts using NER (spaCy)
- Identifies hard negative entities of the same type as ground truth
- Marks entity spans for contrastive loss during training
- Creates augmented examples with entity substitutions

Goal: Address 29.9% of errors caused by "Entity Substitution"
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    print("WARNING: spaCy not available. Install with: pip install spacy")
    print("         and download model: python -m spacy download en_core_web_sm")
    SPACY_AVAILABLE = False


# Entity type mapping for NER
ENTITY_TYPE_MAPPING = {
    # spaCy types -> Simplified types
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",  # Geopolitical entity
    "LOC": "LOCATION",
    "FAC": "LOCATION",  # Facility
    "DATE": "DATE",
    "TIME": "TIME",
    "CARDINAL": "NUMBER",
    "ORDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    "MONEY": "NUMBER",
    "PERCENT": "NUMBER",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "NORP": "GROUP",  # Nationalities or religious/political groups
}


class EntityExtractor:
    """Extract entities using spaCy NER."""

    def __init__(self, model_name="en_core_web_sm"):
        """Initialize spaCy NER model."""
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy not available. Please install it first.")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract all entities from text.

        Returns:
            List of dicts with keys: text, start, end, type
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity_type = ENTITY_TYPE_MAPPING.get(ent.label_, ent.label_)
            entities.append(
                {
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": entity_type,
                    "label": ent.label_,  # Original spaCy label
                }
            )

        return entities

    def get_entity_type(self, text: str, answer_text: str) -> str:
        """
        Determine the entity type of the answer.

        Args:
            text: Full context text
            answer_text: The answer string

        Returns:
            Entity type (PERSON, LOCATION, etc.) or 'UNKNOWN'
        """
        # Try to find answer in extracted entities
        entities = self.extract_entities(text)

        for ent in entities:
            if ent["text"].lower() == answer_text.lower():
                return ent["type"]
            # Check if answer is substring of entity
            if answer_text.lower() in ent["text"].lower():
                return ent["type"]

        # Fallback: run NER on answer alone
        doc = self.nlp(answer_text)
        if doc.ents:
            return ENTITY_TYPE_MAPPING.get(doc.ents[0].label_, doc.ents[0].label_)

        return "UNKNOWN"


def find_hard_negatives(
    context: str,
    answer_text: str,
    answer_start: int,
    entities: List[Dict],
    answer_type: str,
) -> List[Dict]:
    """
    Find hard negative entities of the same type as the answer.

    Args:
        context: Context text
        answer_text: Ground truth answer
        answer_start: Start position of answer
        entities: All entities in context
        answer_type: Type of the answer entity

    Returns:
        List of hard negative entities (same type, different text)
    """
    hard_negatives = []
    answer_end = answer_start + len(answer_text)

    for ent in entities:
        # Skip if not same type
        if ent["type"] != answer_type:
            continue

        # Skip if it's the answer itself (check overlap)
        if (ent["start"] >= answer_start and ent["start"] < answer_end) or (
            ent["end"] > answer_start and ent["end"] <= answer_end
        ):
            continue

        # Skip if text matches answer (case-insensitive)
        if ent["text"].lower() == answer_text.lower():
            continue

        hard_negatives.append(ent)

    return hard_negatives


def create_entity_substitution(
    context: str, answer_text: str, answer_start: int, hard_negative: Dict
) -> Tuple[str, int]:
    """
    Create an augmented example by substituting answer with hard negative.

    Args:
        context: Original context
        answer_text: Original answer
        answer_start: Start position of answer
        hard_negative: Hard negative entity dict

    Returns:
        Tuple of (modified_context, new_answer_start)
    """
    # Replace answer with hard negative entity in context
    answer_end = answer_start + len(answer_text)

    # Create modified context
    modified_context = (
        context[:answer_start] + hard_negative["text"] + context[answer_end:]
    )

    # New answer is the hard negative
    new_answer_start = answer_start

    return modified_context, new_answer_start


def generate_entity_contrastive_pairs(
    dataset_path: str,
    output_path: str,
    entity_weight: float = 2.5,
    augmentation_ratio: float = 0.2,
    max_hard_negatives: int = 5,
    seed: int = 42,
):
    """
    Generate entity-aware contrastive training pairs.

    Args:
        dataset_path: Path to original training dataset (JSONL)
        output_path: Path to save augmented dataset
        entity_weight: Weight multiplier for entity examples (default: 2.5x)
        augmentation_ratio: Ratio of examples to augment with substitution
        max_hard_negatives: Maximum hard negatives to store per example
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("=" * 70)
    print("Entity-Aware Contrastive Pair Generation")
    print("=" * 70)
    print(f"Input dataset: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Entity weight: {entity_weight}x")
    print(f"Augmentation ratio: {augmentation_ratio * 100:.1f}%")
    print()

    # Initialize entity extractor
    print("Initializing spaCy NER model...")
    extractor = EntityExtractor()
    print("✓ NER model loaded\n")

    # Load dataset
    print("Loading dataset...")
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples\n")

    # Statistics
    stats = {
        "total": len(examples),
        "with_entities": 0,
        "with_hard_negatives": 0,
        "entity_type_distribution": defaultdict(int),
        "hard_negatives_found": 0,
        "augmented_substitutions": 0,
        "avg_hard_negatives": 0,
    }

    augmented_examples = []

    # Process each example
    print("Extracting entities and finding hard negatives...")
    for i, ex in enumerate(examples):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(examples)} examples...")

        context = ex["context"]

        # Handle multiple answers
        if not ex["answers"]["text"]:
            # No answer - keep as is
            augmented_examples.append(ex)
            continue

        answer_text = ex["answers"]["text"][0]
        answer_start = ex["answers"]["answer_start"][0]

        # Extract all entities from context
        entities = extractor.extract_entities(context)

        if entities:
            stats["with_entities"] += 1

        # Get answer entity type
        answer_type = extractor.get_entity_type(context, answer_text)

        if answer_type != "UNKNOWN":
            stats["entity_type_distribution"][answer_type] += 1

        # Find hard negatives
        hard_negatives = find_hard_negatives(
            context, answer_text, answer_start, entities, answer_type
        )

        # Limit number of hard negatives
        if len(hard_negatives) > max_hard_negatives:
            hard_negatives = random.sample(hard_negatives, max_hard_negatives)

        if hard_negatives:
            stats["with_hard_negatives"] += 1
            stats["hard_negatives_found"] += len(hard_negatives)

        # Add original example with entity metadata
        original_ex = dict(ex)
        original_ex["loss_weight"] = entity_weight if hard_negatives else 1.0
        original_ex["is_entity_example"] = bool(hard_negatives)
        original_ex["answer_entity_type"] = answer_type
        original_ex["hard_negatives"] = [
            {
                "text": hn["text"],
                "start": hn["start"],
                "end": hn["end"],
                "type": hn["type"],
            }
            for hn in hard_negatives
        ]

        augmented_examples.append(original_ex)

        # Decide whether to create entity substitution augmentation
        if hard_negatives and random.random() < augmentation_ratio:
            # Choose a random hard negative for substitution
            selected_hn = random.choice(hard_negatives)

            # Create augmented example with entity substitution
            modified_context, new_answer_start = create_entity_substitution(
                context, answer_text, answer_start, selected_hn
            )

            augmented_ex = dict(ex)
            augmented_ex["id"] = f"{ex['id']}_entity_sub"
            augmented_ex["context"] = modified_context
            augmented_ex["answers"] = {
                "text": [selected_hn["text"]],
                "answer_start": [new_answer_start],
            }
            augmented_ex["loss_weight"] = entity_weight
            augmented_ex["is_entity_example"] = True
            augmented_ex["entity_augmentation_type"] = "substitution"
            augmented_ex["original_answer"] = answer_text
            augmented_ex["substituted_entity"] = selected_hn["text"]
            augmented_ex["answer_entity_type"] = answer_type

            augmented_examples.append(augmented_ex)
            stats["augmented_substitutions"] += 1

    # Calculate average hard negatives
    if stats["with_hard_negatives"] > 0:
        stats["avg_hard_negatives"] = (
            stats["hard_negatives_found"] / stats["with_hard_negatives"]
        )

    # Save augmented dataset
    print(f"\nSaving augmented dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in augmented_examples:
            f.write(json.dumps(ex) + "\n")

    # Print statistics
    print("\n" + "=" * 70)
    print("Generation Statistics")
    print("=" * 70)
    print(f"Original examples: {stats['total']}")
    print(
        f"  - Examples with entities: {stats['with_entities']} ({stats['with_entities']/stats['total']*100:.1f}%)"
    )
    print(
        f"  - Examples with hard negatives: {stats['with_hard_negatives']} ({stats['with_hard_negatives']/stats['total']*100:.1f}%)"
    )
    print(f"\nHard negatives found: {stats['hard_negatives_found']}")
    print(f"  - Average per example: {stats['avg_hard_negatives']:.2f}")
    print(f"\nEntity substitution augmentations: {stats['augmented_substitutions']}")
    print(
        f"\nTotal output examples: {len(augmented_examples)} ({len(augmented_examples)/stats['total']*100:.1f}% of original)"
    )
    print(
        f"Entity examples (weighted {entity_weight}x): {stats['with_hard_negatives'] + stats['augmented_substitutions']}"
    )

    print(f"\nEntity Type Distribution:")
    for ent_type, count in sorted(
        stats["entity_type_distribution"].items(), key=lambda x: -x[1]
    ):
        print(f"  {ent_type}: {count} ({count/stats['total']*100:.1f}%)")

    print("=" * 70)

    # Save statistics
    stats_dict = {
        "total": stats["total"],
        "with_entities": stats["with_entities"],
        "with_hard_negatives": stats["with_hard_negatives"],
        "hard_negatives_found": stats["hard_negatives_found"],
        "avg_hard_negatives": stats["avg_hard_negatives"],
        "augmented_substitutions": stats["augmented_substitutions"],
        "total_output": len(augmented_examples),
        "entity_type_distribution": dict(stats["entity_type_distribution"]),
    }

    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate entity-aware contrastive training pairs"
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
        "--entity-weight",
        type=float,
        default=2.5,
        help="Loss weight multiplier for entity examples (default: 2.5)",
    )
    parser.add_argument(
        "--augmentation-ratio",
        type=float,
        default=0.2,
        help="Ratio of examples to augment with entity substitution (default: 0.2)",
    )
    parser.add_argument(
        "--max-hard-negatives",
        type=int,
        default=5,
        help="Maximum hard negatives to store per example (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if not SPACY_AVAILABLE:
        print("ERROR: spaCy is required for entity extraction.")
        print("Install with: pip install spacy")
        print("Download model: python -m spacy download en_core_web_sm")
        return 1

    generate_entity_contrastive_pairs(
        dataset_path=args.input,
        output_path=args.output,
        entity_weight=args.entity_weight,
        augmentation_ratio=args.augmentation_ratio,
        max_hard_negatives=args.max_hard_negatives,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    exit(main())
