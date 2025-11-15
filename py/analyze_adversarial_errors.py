#!/usr/bin/env python3
"""
Analyze errors on AddSent adversarial dataset
Categorize by error type and question type
"""

import json
from pathlib import Path
from collections import defaultdict
import re

def load_predictions(pred_file):
    """Load predictions from file (supports both JSON and JSONL)"""
    predictions = {}
    with open(pred_file, 'r') as f:
        # Try JSONL first (one JSON object per line)
        if pred_file.endswith('.jsonl'):
            for line in f:
                item = json.loads(line)
                # Handle both 'prediction_text' and 'predicted_answer' field names
                pred = item.get('prediction_text') or item.get('predicted_answer')
                predictions[item['id']] = pred
        else:
            # Try regular JSON array
            data = json.load(f)
            for item in data:
                # Handle both 'prediction_text' and 'predicted_answer' field names
                pred = item.get('prediction_text') or item.get('predicted_answer')
                predictions[item['id']] = pred
    return predictions

def load_dataset(dataset_file):
    """Load dataset"""
    examples = []
    with open(dataset_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def normalize_answer(s):
    """Normalize answer for comparison"""
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, ground_truths):
    """Check if prediction matches any ground truth"""
    prediction = normalize_answer(prediction)
    return any(normalize_answer(gt) == prediction for gt in ground_truths)

def get_question_type(question):
    """Categorize question by type"""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ['who', 'whom']):
        return 'WHO'
    elif any(word in q_lower for word in ['where', 'which city', 'which country', 'which place']):
        return 'WHERE'
    elif any(word in q_lower for word in ['when', 'what year', 'what date', 'which year']):
        return 'WHEN'
    elif any(word in q_lower for word in ['how many', 'how much', 'what number']):
        return 'NUMBER'
    elif any(word in q_lower for word in ['what', 'which']):
        return 'WHAT'
    elif any(word in q_lower for word in ['why', 'how']):
        return 'WHY/HOW'
    else:
        return 'OTHER'

def find_distractor_in_context(context):
    """Try to identify if context contains adversarial distractor"""
    # Look for common distractor patterns
    distractor_patterns = [
        r'however[,\s]',
        r'according to[,\s]',
        r'some sources',
        r'initial reports',
        r'recent reports',
        r'historians',
        r'documents show',
        r'research reveals',
        r'evidence points',
    ]
    
    for pattern in distractor_patterns:
        if re.search(pattern, context.lower()):
            return True
    return False

def get_answer_position(context, answer_start):
    """Determine if answer is in first half or second half of context"""
    if answer_start < len(context) / 2:
        return 'FIRST_HALF'
    else:
        return 'SECOND_HALF'

def analyze_errors(dataset_file, predictions_file, output_file='error_analysis.json'):
    """
    Analyze errors and categorize them
    """
    print("Loading data...")
    examples = load_dataset(dataset_file)
    predictions = load_predictions(predictions_file)
    
    print(f"Loaded {len(examples)} examples and {len(predictions)} predictions")
    
    # Statistics
    stats = {
        'total': len(examples),
        'correct': 0,
        'incorrect': 0,
        'by_question_type': defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0}),
        'by_answer_position': defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0}),
        'errors': []
    }
    
    print("\nAnalyzing predictions...")
    for example in examples:
        ex_id = example['id']
        question = example['question']
        context = example['context']
        ground_truths = example['answers']['text']
        answer_starts = example['answers']['answer_start']
        
        if ex_id not in predictions:
            print(f"Warning: No prediction for {ex_id}")
            continue
        
        prediction = predictions[ex_id]
        
        # Check if correct
        is_correct = exact_match(prediction, ground_truths)
        
        # Question type
        q_type = get_question_type(question)
        stats['by_question_type'][q_type]['total'] += 1
        
        # Answer position
        answer_pos = get_answer_position(context, answer_starts[0])
        stats['by_answer_position'][answer_pos]['total'] += 1
        
        if is_correct:
            stats['correct'] += 1
            stats['by_question_type'][q_type]['correct'] += 1
            stats['by_answer_position'][answer_pos]['correct'] += 1
        else:
            stats['incorrect'] += 1
            stats['by_question_type'][q_type]['incorrect'] += 1
            stats['by_answer_position'][answer_pos]['incorrect'] += 1
            
            # Store error for analysis
            has_distractor = find_distractor_in_context(context)
            
            stats['errors'].append({
                'id': ex_id,
                'question': question,
                'question_type': q_type,
                'answer_position': answer_pos,
                'has_distractor': has_distractor,
                'context': context[:200] + '...' if len(context) > 200 else context,
                'ground_truth': ground_truths[0],
                'prediction': prediction,
                'context_length': len(context)
            })
    
    # Calculate accuracy by category
    print("\n" + "="*60)
    print("ERROR ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total: {stats['total']}")
    print(f"  Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)")
    print(f"  Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.2f}%)")
    
    print(f"\n📋 Performance by Question Type:")
    for q_type, counts in sorted(stats['by_question_type'].items()):
        total = counts['total']
        correct = counts['correct']
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {q_type:12s}: {correct:4d}/{total:4d} = {acc:5.2f}%")
    
    print(f"\n📍 Performance by Answer Position:")
    for pos, counts in sorted(stats['by_answer_position'].items()):
        total = counts['total']
        correct = counts['correct']
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {pos:12s}: {correct:4d}/{total:4d} = {acc:5.2f}%")
    
    # Save detailed results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'accuracy': stats['correct'] / stats['total'] * 100
            },
            'by_question_type': dict(stats['by_question_type']),
            'by_answer_position': dict(stats['by_answer_position']),
            'sample_errors': stats['errors'][:50]  # Save first 50 errors
        }, f, indent=2)
    
    print(f"\n✅ Detailed analysis saved to: {output_path}")
    print(f"   Total errors stored: {len(stats['errors'])}")
    
    # Show some example errors
    print(f"\n🔍 Sample Errors (first 5):")
    for i, error in enumerate(stats['errors'][:5], 1):
        print(f"\n--- Error {i} ---")
        print(f"Question Type: {error['question_type']}")
        print(f"Question: {error['question']}")
        print(f"Ground Truth: {error['ground_truth']}")
        print(f"Prediction: {error['prediction']}")
        print(f"Has Distractor: {error['has_distractor']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_adversarial_errors.py <dataset.jsonl> <predictions.json>")
        print("\nExample:")
        print("  python3 analyze_adversarial_errors.py ./data/addsent.jsonl ./eval_addsent/predictions.json")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    predictions_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'error_analysis.json'
    
    analyze_errors(dataset_file, predictions_file, output_file)
