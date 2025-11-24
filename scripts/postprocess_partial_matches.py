#!/usr/bin/env python3
"""
Post-Processing for Partial Match Errors

This script fixes partial match errors (30.6% of errors) where the model predicts
a substring of the correct answer (e.g., "Broncos" instead of "Denver Broncos").

Approach:
1. Run NER on the context to identify full entity boundaries
2. Check if prediction is a substring of any entity
3. Expand prediction to full entity if conditions are met

Key insight: No training required - pure inference-time fix!
"""

import argparse
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    print("ERROR: spaCy not installed")
    print("Install with: pip install spacy")
    print("Download model: python -m spacy download en_core_web_sm")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PartialMatchPostprocessor:
    """Post-processor to fix partial match errors using NER expansion."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the post-processor.

        Args:
            spacy_model: Name of spaCy model to use for NER
        """
        logger.info(f"Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f"Model {spacy_model} not found. Downloading...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        # Statistics
        self.stats = {
            "total_predictions": 0,
            "expanded_predictions": 0,
            "exact_matches": 0,
            "no_entity_found": 0,
            "multiple_entities_found": 0,
            "expansion_by_type": {},
        }

    def find_containing_entity(
        self, context: str, prediction: str, prediction_start: int
    ) -> Optional[Dict[str, any]]:
        """
        Find the entity that contains the prediction, using context expansion.

        Args:
            context: Full context paragraph
            prediction: Model's predicted answer
            prediction_start: Character start position of prediction

        Returns:
            Dictionary with entity info, or None if no containing entity found
        """
        # Run NER on context
        doc = self.nlp(context)

        prediction_end = prediction_start + len(prediction)
        prediction_lower = prediction.strip().lower()

        # Strategy 1: Look for entities that contain the prediction as substring
        containing_entities = []

        for ent in doc.ents:
            ent_text = ent.text.strip()
            ent_lower = ent_text.lower()

            # Check if prediction is substring of entity text
            if prediction_lower not in ent_lower:
                continue

            # Check positional overlap
            ent_start = ent.start_char
            ent_end = ent.end_char

            if ent_start <= prediction_start and prediction_end <= ent_end:
                if ent_text != prediction.strip():
                    containing_entities.append(
                        {
                            "text": ent_text,
                            "start": ent_start,
                            "end": ent_end,
                            "type": ent.label_,
                        }
                    )
            elif (
                ent_start <= prediction_start < ent_end
                or ent_start < prediction_end <= ent_end
            ):
                if ent_text != prediction.strip():
                    containing_entities.append(
                        {
                            "text": ent_text,
                            "start": ent_start,
                            "end": ent_end,
                            "type": ent.label_,
                        }
                    )

        # Strategy 2: Look for multi-word entities near prediction
        # Example: "Denver" + "Broncos" or "Dr." + "Jane Smith"
        if len(containing_entities) == 0:
            # Extract context window around prediction (±30 chars)
            window_start = max(0, prediction_start - 30)
            window_end = min(len(context), prediction_end + 30)

            # Find entities in this window
            nearby_entities = []
            for ent in doc.ents:
                if window_start <= ent.start_char <= window_end:
                    nearby_entities.append(ent)

            # Check if we can merge adjacent entities with prediction
            for i in range(len(nearby_entities)):
                ent = nearby_entities[i]

                # Check if entity is adjacent to prediction
                gap_before = prediction_start - ent.end_char
                gap_after = ent.start_char - prediction_end

                # If entity is within 1-2 chars (whitespace), merge
                if 0 <= gap_before <= 2:
                    # Entity is before prediction: "Denver" + "Broncos"
                    merged_text = context[ent.start_char : prediction_end].strip()
                    if len(merged_text) > len(prediction) * 1.2:
                        containing_entities.append(
                            {
                                "text": merged_text,
                                "start": ent.start_char,
                                "end": prediction_end,
                                "type": ent.label_,
                            }
                        )
                elif 0 <= gap_after <= 2:
                    # Entity is after prediction: "New York" + "City"
                    merged_text = context[prediction_start : ent.end_char].strip()
                    if len(merged_text) > len(prediction) * 1.2:
                        containing_entities.append(
                            {
                                "text": merged_text,
                                "start": prediction_start,
                                "end": ent.end_char,
                                "type": ent.label_,
                            }
                        )

        if len(containing_entities) == 0:
            return None

        # Choose the best entity (largest that contains prediction)
        return max(containing_entities, key=lambda e: len(e["text"]))

    def should_expand(
        self, prediction: str, entity: Dict[str, any], min_expansion_ratio: float = 1.5
    ) -> bool:
        """
        Determine if prediction should be expanded to full entity.

        Args:
            prediction: Model's predicted answer
            entity: Containing entity information
            min_expansion_ratio: Minimum ratio of entity/prediction length

        Returns:
            True if should expand, False otherwise
        """
        # Don't expand if prediction is already very close to entity
        entity_text = entity["text"].strip()
        prediction_text = prediction.strip()

        # Check length ratio
        if len(entity_text) < len(prediction_text) * min_expansion_ratio:
            return False

        # Check if prediction is a meaningful substring
        # (not just punctuation or single character)
        if len(prediction_text) < 2:
            return False

        # Check if entity seems like a proper expansion
        # (should contain the prediction as a word/token boundary)
        pred_lower = prediction_text.lower()
        entity_lower = entity_text.lower()

        # Simple substring check
        if pred_lower not in entity_lower:
            return False

        return True

    def postprocess_prediction(
        self,
        context: str,
        question: str,
        prediction: str,
        prediction_start: int,
        min_expansion_ratio: float = 1.3,
    ) -> Tuple[str, int, Dict[str, any]]:
        """
        Post-process a single prediction.

        Args:
            context: Context paragraph
            question: Question text
            prediction: Model's predicted answer
            prediction_start: Character start position
            min_expansion_ratio: Minimum expansion ratio

        Returns:
            Tuple of (expanded_text, new_start, metadata)
        """
        self.stats["total_predictions"] += 1

        # Find containing entity
        entity = self.find_containing_entity(context, prediction, prediction_start)

        if entity is None:
            self.stats["no_entity_found"] += 1
            return (
                prediction,
                prediction_start,
                {"expanded": False, "reason": "no_containing_entity"},
            )

        # Check if should expand
        if not self.should_expand(prediction, entity, min_expansion_ratio):
            self.stats["exact_matches"] += 1
            return (
                prediction,
                prediction_start,
                {
                    "expanded": False,
                    "reason": "expansion_not_beneficial",
                    "entity_found": entity["text"],
                },
            )

        # Expand to full entity
        self.stats["expanded_predictions"] += 1
        entity_type = entity["type"]
        self.stats["expansion_by_type"][entity_type] = (
            self.stats["expansion_by_type"].get(entity_type, 0) + 1
        )

        logger.debug(
            f"Expanding '{prediction}' -> '{entity['text']}' " f"(type: {entity_type})"
        )

        return (
            entity["text"],
            entity["start"],
            {
                "expanded": True,
                "original_prediction": prediction,
                "original_start": prediction_start,
                "expanded_prediction": entity["text"],
                "expanded_start": entity["start"],
                "entity_type": entity_type,
            },
        )

    def postprocess_file(
        self,
        input_file: str,
        output_file: str,
        min_expansion_ratio: float = 1.3,
        save_metadata: bool = True,
    ):
        """
        Post-process all predictions in a file.

        Args:
            input_file: Path to input JSONL with predictions
            output_file: Path to output JSONL with expanded predictions
            min_expansion_ratio: Minimum expansion ratio
            save_metadata: Whether to save expansion metadata
        """
        logger.info(f"Processing: {input_file}")
        logger.info(f"Output: {output_file}")

        # Reset statistics
        self.stats = {
            "total_predictions": 0,
            "expanded_predictions": 0,
            "exact_matches": 0,
            "no_entity_found": 0,
            "multiple_entities_found": 0,
            "expansion_by_type": {},
        }

        processed_examples = []

        with open(input_file, "r") as f_in:
            for line_num, line in enumerate(f_in, 1):
                try:
                    example = json.loads(line.strip())

                    # Extract required fields
                    context = example.get("context", "")
                    question = example.get("question", "")
                    prediction = example.get(
                        "predicted_answer", example.get("answer", "")
                    )
                    prediction_start = example.get(
                        "predicted_start", example.get("answer_start", 0)
                    )

                    # Post-process
                    expanded_text, expanded_start, metadata = (
                        self.postprocess_prediction(
                            context=context,
                            question=question,
                            prediction=prediction,
                            prediction_start=prediction_start,
                            min_expansion_ratio=min_expansion_ratio,
                        )
                    )

                    # Update example
                    example["predicted_answer"] = expanded_text
                    example["predicted_start"] = expanded_start

                    if save_metadata:
                        example["postprocessing_metadata"] = metadata

                    processed_examples.append(example)

                    if line_num % 100 == 0:
                        logger.info(f"Processed {line_num} examples...")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

        # Write output
        with open(output_file, "w") as f_out:
            for example in processed_examples:
                f_out.write(json.dumps(example) + "\n")

        # Print statistics
        self._print_statistics()

        logger.info(f"✓ Processed {len(processed_examples)} examples")
        logger.info(f"✓ Output saved to: {output_file}")

    def _print_statistics(self):
        """Print post-processing statistics."""
        total = self.stats["total_predictions"]
        expanded = self.stats["expanded_predictions"]

        print("\n" + "=" * 70)
        print("Post-Processing Statistics")
        print("=" * 70)
        print(f"Total predictions: {total}")
        print(f"Expanded predictions: {expanded} ({100*expanded/total:.2f}%)")
        print(
            f"Exact matches (no expansion): {self.stats['exact_matches']} "
            f"({100*self.stats['exact_matches']/total:.2f}%)"
        )
        print(
            f"No entity found: {self.stats['no_entity_found']} "
            f"({100*self.stats['no_entity_found']/total:.2f}%)"
        )

        if self.stats["expansion_by_type"]:
            print("\nExpansions by entity type:")
            for entity_type, count in sorted(
                self.stats["expansion_by_type"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(
                    f"  {entity_type:15s}: {count:4d} "
                    f"({100*count/expanded:.1f}% of expansions)"
                )

        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process predictions to fix partial match errors"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL file with predictions"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with expanded predictions",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use for NER (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--min-expansion-ratio",
        type=float,
        default=1.3,
        help="Minimum ratio of entity/prediction length for expansion (default: 1.3)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not save expansion metadata in output",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create post-processor
    postprocessor = PartialMatchPostprocessor(spacy_model=args.spacy_model)

    # Process file
    postprocessor.postprocess_file(
        input_file=args.input,
        output_file=args.output,
        min_expansion_ratio=args.min_expansion_ratio,
        save_metadata=not args.no_metadata,
    )


if __name__ == "__main__":
    main()
