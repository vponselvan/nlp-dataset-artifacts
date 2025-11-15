#!/usr/bin/env python3
"""
Advanced Error Analysis for AddSent Adversarial QA
Provides multiple categorization schemes beyond question type
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Any

def load_data(predictions_path: str, dataset_path: str) -> tuple:
    """Load predictions and dataset"""
    with open(predictions_path, 'r') as f:
        predictions = [json.loads(line) for line in f]
    
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    return predictions, dataset

def get_answer_type(answer: str) -> str:
    """Categorize answer by its semantic type"""
    answer = answer.strip().lower()
    
    # Date patterns
    if re.search(r'\d{4}', answer) and len(answer) < 20:
        return "YEAR"
    if re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', answer, re.I):
        return "DATE"
    
    # Number patterns
    if re.match(r'^[\d,]+$', answer.replace(',', '')):
        return "PURE_NUMBER"
    if re.search(r'\d+\s*(million|billion|thousand|hundred)', answer, re.I):
        return "QUANTITY"
    if re.search(r'\d+[-–]\d+', answer):  # Score like "24-10"
        return "SCORE"
    if re.search(r'\d+', answer) and len(answer.split()) <= 3:
        return "NUMERIC_PHRASE"
    
    # Location patterns
    if re.search(r'\b(stadium|arena|field|dome|center|park)\b', answer, re.I):
        return "VENUE"
    if re.search(r'\b(city|state|country|county|province)\b', answer, re.I):
        return "LOCATION"
    if re.search(r',\s*[A-Z]{2}\b', answer):  # "Santa Clara, CA"
        return "LOCATION"
    
    # Organization/Team patterns
    if re.search(r'\b(team|club|organization|company|corporation|inc|ltd)\b', answer, re.I):
        return "ORGANIZATION"
    if len(answer.split()) <= 3 and answer[0].isupper():
        if re.search(r'\b(broncos|panthers|49ers|raiders|patriots|seahawks)\b', answer, re.I):
            return "TEAM_NAME"
    
    # Person patterns
    if re.search(r'\b(president|ceo|coach|player|director|manager|doctor|professor)\b', answer, re.I):
        return "PERSON_TITLE"
    if len(answer.split()) >= 2 and all(word[0].isupper() for word in answer.split()[:2] if word):
        return "PERSON_NAME"
    
    # Abstract concepts
    if len(answer.split()) > 5:
        return "LONG_PHRASE"
    
    return "SHORT_PHRASE"

def get_question_complexity(question: str) -> str:
    """Categorize question by complexity"""
    question = question.lower()
    words = question.split()
    
    # Multi-part questions
    if ' and ' in question or ' or ' in question:
        return "MULTI_PART"
    
    # Counting/quantification
    if any(word in question for word in ['how many', 'how much', 'number of']):
        return "COUNTING"
    
    # Comparison
    if any(word in question for word in ['more', 'less', 'better', 'worse', 'compared', 'versus', 'difference']):
        return "COMPARISON"
    
    # Superlatives
    if any(word in question for word in ['most', 'least', 'best', 'worst', 'first', 'last', 'highest', 'lowest']):
        return "SUPERLATIVE"
    
    # Causation/reasoning
    if question.startswith('why') or question.startswith('how'):
        if 'how many' not in question and 'how much' not in question:
            return "CAUSAL_REASONING"
    
    # Simple factual
    if len(words) <= 7:
        return "SIMPLE_FACTUAL"
    
    return "COMPLEX_FACTUAL"

def get_error_type(ground_truth: str, prediction: str, context: str) -> str:
    """Categorize the type of error made"""
    gt_lower = ground_truth.lower().strip()
    pred_lower = prediction.lower().strip()
    
    # Exact match (shouldn't be error, but check)
    if gt_lower == pred_lower:
        return "FALSE_NEGATIVE"
    
    # Partial match
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return "PARTIAL_MATCH"
    
    # Similar entity type (same answer type)
    gt_type = get_answer_type(ground_truth)
    pred_type = get_answer_type(prediction)
    if gt_type == pred_type:
        return f"WRONG_{gt_type}"
    
    # Distractor from context (check if prediction appears in context)
    if prediction.lower() in context.lower():
        # Check if it's near ground truth
        gt_pos = context.lower().find(gt_lower)
        pred_pos = context.lower().find(pred_lower)
        if gt_pos != -1 and pred_pos != -1:
            distance = abs(gt_pos - pred_pos)
            if distance < 100:
                return "NEARBY_DISTRACTOR"
            else:
                return "DISTANT_DISTRACTOR"
        return "CONTEXT_DISTRACTOR"
    
    # Hallucination (not in context at all)
    return "HALLUCINATION"

def get_answer_length_category(answer: str) -> str:
    """Categorize by answer length"""
    words = len(answer.split())
    if words == 1:
        return "SINGLE_WORD"
    elif words == 2:
        return "TWO_WORDS"
    elif words <= 4:
        return "SHORT_PHRASE"
    else:
        return "LONG_PHRASE"

def get_context_length_category(context_length: int) -> str:
    """Categorize by context length"""
    if context_length < 400:
        return "SHORT_CONTEXT"
    elif context_length < 800:
        return "MEDIUM_CONTEXT"
    else:
        return "LONG_CONTEXT"

def analyze_errors_advanced(predictions: List[Dict], dataset: List[Dict]) -> Dict:
    """Perform advanced error analysis with multiple categorization schemes"""
    
    # Create lookup dictionary
    pred_dict = {p['id']: p for p in predictions}
    
    results = {
        "summary": {"total": 0, "correct": 0, "incorrect": 0},
        "by_answer_type": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "by_question_complexity": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "by_error_type": defaultdict(int),
        "by_answer_length": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "by_context_length": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "combined_patterns": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "sample_errors_by_type": defaultdict(list)
    }
    
    for example in dataset:
        example_id = example['id']
        if example_id not in pred_dict:
            continue
        
        pred = pred_dict[example_id]
        ground_truth = example['answers']['text'][0]
        prediction = pred.get('predicted_answer', pred.get('prediction_text', ''))
        context = example['context']
        question = example['question']
        
        # Check correctness
        is_correct = ground_truth.lower().strip() == prediction.lower().strip()
        
        # Update summary
        results["summary"]["total"] += 1
        if is_correct:
            results["summary"]["correct"] += 1
        else:
            results["summary"]["incorrect"] += 1
        
        # Categorize by answer type
        answer_type = get_answer_type(ground_truth)
        results["by_answer_type"][answer_type]["total"] += 1
        if is_correct:
            results["by_answer_type"][answer_type]["correct"] += 1
        else:
            results["by_answer_type"][answer_type]["incorrect"] += 1
        
        # Categorize by question complexity
        complexity = get_question_complexity(question)
        results["by_question_complexity"][complexity]["total"] += 1
        if is_correct:
            results["by_question_complexity"][complexity]["correct"] += 1
        else:
            results["by_question_complexity"][complexity]["incorrect"] += 1
        
        # Categorize by answer length
        answer_length = get_answer_length_category(ground_truth)
        results["by_answer_length"][answer_length]["total"] += 1
        if is_correct:
            results["by_answer_length"][answer_length]["correct"] += 1
        else:
            results["by_answer_length"][answer_length]["incorrect"] += 1
        
        # Categorize by context length
        context_length_cat = get_context_length_category(len(context))
        results["by_context_length"][context_length_cat]["total"] += 1
        if is_correct:
            results["by_context_length"][context_length_cat]["correct"] += 1
        else:
            results["by_context_length"][context_length_cat]["incorrect"] += 1
        
        # Combined pattern: complexity + answer_type
        combined_key = f"{complexity}_{answer_type}"
        results["combined_patterns"][combined_key]["total"] += 1
        if is_correct:
            results["combined_patterns"][combined_key]["correct"] += 1
        else:
            results["combined_patterns"][combined_key]["incorrect"] += 1
        
        # Error type analysis (only for incorrect)
        if not is_correct:
            error_type = get_error_type(ground_truth, prediction, context)
            results["by_error_type"][error_type] += 1
            
            # Sample errors for each type (max 3 per type)
            if len(results["sample_errors_by_type"][error_type]) < 3:
                results["sample_errors_by_type"][error_type].append({
                    "id": example_id,
                    "question": question,
                    "complexity": complexity,
                    "answer_type": answer_type,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "error_type": error_type
                })
    
    # Calculate accuracies
    for category in ["by_answer_type", "by_question_complexity", "by_answer_length", "by_context_length", "combined_patterns"]:
        for key in results[category]:
            total = results[category][key]["total"]
            correct = results[category][key]["correct"]
            results[category][key]["accuracy"] = (correct / total * 100) if total > 0 else 0
    
    # Convert defaultdicts to regular dicts for JSON serialization
    results["by_answer_type"] = dict(results["by_answer_type"])
    results["by_question_complexity"] = dict(results["by_question_complexity"])
    results["by_answer_length"] = dict(results["by_answer_length"])
    results["by_context_length"] = dict(results["by_context_length"])
    results["combined_patterns"] = dict(results["combined_patterns"])
    results["by_error_type"] = dict(results["by_error_type"])
    results["sample_errors_by_type"] = dict(results["sample_errors_by_type"])
    
    return results

def main():
    predictions_path = "./eval_results_adversarial/eval_predictions.jsonl"
    dataset_path = "./data/addsent.jsonl"
    output_path = "./analysis/advanced_error_analysis.json"
    
    print("Loading data...")
    predictions, dataset = load_data(predictions_path, dataset_path)
    print(f"Loaded {len(predictions)} predictions and {len(dataset)} examples")
    
    print("\nPerforming advanced error analysis...")
    results = analyze_errors_advanced(predictions, dataset)
    
    print(f"\nTotal: {results['summary']['total']}")
    print(f"Correct: {results['summary']['correct']}")
    print(f"Incorrect: {results['summary']['incorrect']}")
    print(f"Accuracy: {results['summary']['correct'] / results['summary']['total'] * 100:.2f}%")
    
    print("\n=== Performance by Answer Type ===")
    sorted_answer_types = sorted(results['by_answer_type'].items(), 
                                 key=lambda x: x[1]['accuracy'])
    for answer_type, stats in sorted_answer_types:
        print(f"{answer_type:20s}: {stats['correct']:4d}/{stats['total']:4d} = {stats['accuracy']:5.1f}%")
    
    print("\n=== Performance by Question Complexity ===")
    sorted_complexity = sorted(results['by_question_complexity'].items(), 
                               key=lambda x: x[1]['accuracy'])
    for complexity, stats in sorted_complexity:
        print(f"{complexity:20s}: {stats['correct']:4d}/{stats['total']:4d} = {stats['accuracy']:5.1f}%")
    
    print("\n=== Error Type Distribution ===")
    sorted_errors = sorted(results['by_error_type'].items(), 
                          key=lambda x: x[1], reverse=True)
    for error_type, count in sorted_errors:
        print(f"{error_type:25s}: {count:4d} ({count/results['summary']['incorrect']*100:5.1f}%)")
    
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
