#!/usr/bin/env python3
"""
Evaluate model with post-processing for partial matches.

This script:
1. Runs inference on test set
2. Applies partial match post-processing
3. Evaluates the expanded predictions
4. Compares before/after performance
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_predictions(predictions_file: str, gold_file: str) -> Dict[str, float]:
    """
    Evaluate predictions against gold answers.

    Args:
        predictions_file: File with predictions
        gold_file: File with gold answers

    Returns:
        Dictionary with metrics
    """
    # Load predictions
    predictions = {}
    with open(predictions_file, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            qid = example.get("id", example.get("question_id"))
            pred = example.get("predicted_answer", example.get("answer", ""))
            predictions[qid] = pred

    # Load gold answers
    gold_answers = {}
    with open(gold_file, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            qid = example.get("id", example.get("question_id"))
            
            # Handle different answer formats
            if "answers" in example and isinstance(example["answers"], dict):
                # SQuAD format: {"answers": {"text": ["answer1", "answer2"]}}
                answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            else:
                # Simple format: {"answer": "answer"}
                answer = example.get("answer", example.get("gold_answer", ""))
            
            gold_answers[qid] = answer

    # Compute metrics
    exact_matches = []
    f1_scores = []

    for qid in predictions:
        if qid not in gold_answers:
            logger.warning(f"Question ID {qid} not found in gold file")
            continue

        pred = predictions[qid]
        gold = gold_answers[qid]

        em = compute_exact_match(pred, gold)
        f1 = compute_f1(pred, gold)

        exact_matches.append(em)
        f1_scores.append(f1)
    
    logger.info(f"Evaluated {len(exact_matches)} examples")
    logger.info(f"Sample - Pred: '{list(predictions.values())[0]}', Gold: '{list(gold_answers.values())[0]}'")

    return {
        "exact_match": (
            100.0 * sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        ),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "num_examples": len(exact_matches),
    }


def run_inference_with_postprocessing(
    model_dir: str, test_file: str, output_dir: str, min_expansion_ratio: float = 1.3
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run inference with and without post-processing.

    Args:
        model_dir: Path to trained model
        test_file: Path to test data
        output_dir: Directory to save outputs
        min_expansion_ratio: Minimum expansion ratio

    Returns:
        Tuple of (metrics_before, metrics_after)
    """
    import subprocess
    import sys

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Step 1: Run inference
    logger.info("Step 1: Running inference...")
    predictions_file = output_dir / "predictions_raw.jsonl"

    cmd = [
        sys.executable,
        "../run.py",
        "--model",
        model_dir,
        "--task",
        "qa",
        "--dataset",
        test_file,
        "--do_eval",
        "--output_dir",
        str(output_dir),
        "--per_device_eval_batch_size",
        "32",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Inference failed: {result.stderr}")
        raise RuntimeError("Inference failed")

    logger.info("✓ Inference complete")

    # Step 2: Apply post-processing
    logger.info("Step 2: Applying post-processing...")
    postprocessed_file = output_dir / "predictions_postprocessed.jsonl"

    postprocessor = PartialMatchPostprocessor()
    postprocessor.postprocess_file(
        input_file=str(predictions_file),
        output_file=str(postprocessed_file),
        min_expansion_ratio=min_expansion_ratio,
    )

    logger.info("✓ Post-processing complete")

    # Step 3: Evaluate both
    logger.info("Step 3: Evaluating...")

    metrics_before = evaluate_predictions(str(predictions_file), test_file)
    metrics_after = evaluate_predictions(str(postprocessed_file), test_file)

    return metrics_before, metrics_after


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with post-processing")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../postprocessing_results",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--min-expansion-ratio", type=float, default=1.3, help="Minimum expansion ratio"
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        help="Pre-computed predictions file (skip inference)",
    )
    parser.add_argument(
        "--gold-file", type=str, help="Gold answers file for evaluation"
    )

    args = parser.parse_args()

    if args.predictions_file:
        # Use pre-computed predictions
        logger.info("Using pre-computed predictions")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        postprocessed_file = output_dir / "predictions_postprocessed.jsonl"

        # Check if postprocessed file already exists
        if postprocessed_file.exists():
            logger.info(f"Using existing postprocessed file: {postprocessed_file}")
        else:
            # Apply post-processing
            from postprocess_partial_matches import PartialMatchPostprocessor
            postprocessor = PartialMatchPostprocessor()
            postprocessor.postprocess_file(
                input_file=args.predictions_file,
                output_file=str(postprocessed_file),
                min_expansion_ratio=args.min_expansion_ratio,
            )

        # Evaluate
        gold_file = args.gold_file or args.test_data
        metrics_before = evaluate_predictions(args.predictions_file, gold_file)
        metrics_after = evaluate_predictions(str(postprocessed_file), gold_file)
    else:
        # Run full pipeline
        metrics_before, metrics_after = run_inference_with_postprocessing(
            model_dir=args.model,
            test_file=args.test_data,
            output_dir=args.output_dir,
            min_expansion_ratio=args.min_expansion_ratio,
        )

    # Print comparison
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"\nBefore post-processing:")
    print(f"  Exact Match: {metrics_before['exact_match']:.2f}%")
    print(f"  F1 Score:    {metrics_before['f1']:.2f}%")
    print(f"\nAfter post-processing:")
    print(f"  Exact Match: {metrics_after['exact_match']:.2f}%")
    print(f"  F1 Score:    {metrics_after['f1']:.2f}%")
    print(f"\nImprovement:")
    print(
        f"  ΔEM: +{metrics_after['exact_match'] - metrics_before['exact_match']:.2f} points"
    )
    print(f"  ΔF1: +{metrics_after['f1'] - metrics_before['f1']:.2f} points")
    print("=" * 70 + "\n")

    # Save results
    results = {
        "before_postprocessing": metrics_before,
        "after_postprocessing": metrics_after,
        "improvement": {
            "exact_match": metrics_after["exact_match"] - metrics_before["exact_match"],
            "f1": metrics_after["f1"] - metrics_before["f1"],
        },
    }

    results_file = Path(args.output_dir) / "postprocessing_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
