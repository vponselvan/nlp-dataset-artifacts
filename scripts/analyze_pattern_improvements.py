#!/usr/bin/env python3
"""
Analyze which adversarial patterns were corrected by the best model (70-30).
Compares error patterns between baseline and adversarially trained model.
"""

import json
from collections import defaultdict
from pathlib import Path

def load_predictions(pred_file):
    """Load predictions from JSONL file"""
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            pred = item.get('prediction_text') or item.get('predicted_answer')
            predictions[item['id']] = {
                'prediction': pred,
                'context': item.get('context', ''),
                'question': item.get('question', ''),
                'ground_truth': item['answers']['text'][0] if 'answers' in item else None
            }
    return predictions

def is_correct(prediction, ground_truth):
    """Check if prediction matches ground truth"""
    if not prediction or not ground_truth:
        return False
    return prediction.lower().strip() == ground_truth.lower().strip()

def load_baseline_patterns():
    """Load baseline error patterns from linguistic analysis"""
    pattern_file = './analysis/linguistic_pattern_analysis.json'
    if not Path(pattern_file).exists():
        return None
    
    with open(pattern_file, 'r') as f:
        return json.load(f)

def analyze_improvements(baseline_preds_file, model_preds_file, output_file='./evaluation/pattern_improvements.json'):
    """Analyze which patterns were corrected"""
    
    print("=" * 70)
    print("Pattern Improvement Analysis")
    print("=" * 70)
    print()
    
    # Load predictions
    print("Loading predictions...")
    baseline_preds = load_predictions(baseline_preds_file)
    model_preds = load_predictions(model_preds_file)
    print(f"  Baseline: {len(baseline_preds)} predictions")
    print(f"  Model: {len(model_preds)} predictions")
    print()
    
    # Find common examples
    common_ids = set(baseline_preds.keys()) & set(model_preds.keys())
    print(f"Common examples: {len(common_ids)}")
    print()
    
    # Analyze corrections
    stats = {
        'total': len(common_ids),
        'baseline_correct': 0,
        'baseline_incorrect': 0,
        'model_correct': 0,
        'model_incorrect': 0,
        'corrections': 0,  # baseline wrong → model correct
        'regressions': 0,  # baseline correct → model wrong
        'still_wrong': 0,  # both wrong
        'still_correct': 0,  # both correct
        'correction_examples': [],
        'regression_examples': [],
        'persistent_error_examples': []
    }
    
    for ex_id in common_ids:
        baseline = baseline_preds[ex_id]
        model = model_preds[ex_id]
        
        baseline_correct = is_correct(baseline['prediction'], baseline['ground_truth'])
        model_correct = is_correct(model['prediction'], model['ground_truth'])
        
        if baseline_correct:
            stats['baseline_correct'] += 1
        else:
            stats['baseline_incorrect'] += 1
        
        if model_correct:
            stats['model_correct'] += 1
        else:
            stats['model_incorrect'] += 1
        
        # Categorize change
        if not baseline_correct and model_correct:
            stats['corrections'] += 1
            if len(stats['correction_examples']) < 10:
                stats['correction_examples'].append({
                    'id': ex_id,
                    'question': baseline['question'],
                    'ground_truth': baseline['ground_truth'],
                    'baseline_prediction': baseline['prediction'],
                    'model_prediction': model['prediction'],
                    'context_snippet': baseline['context'][:200] + '...'
                })
        elif baseline_correct and not model_correct:
            stats['regressions'] += 1
            if len(stats['regression_examples']) < 10:
                stats['regression_examples'].append({
                    'id': ex_id,
                    'question': baseline['question'],
                    'ground_truth': baseline['ground_truth'],
                    'baseline_prediction': baseline['prediction'],
                    'model_prediction': model['prediction']
                })
        elif not baseline_correct and not model_correct:
            stats['still_wrong'] += 1
            if len(stats['persistent_error_examples']) < 10:
                stats['persistent_error_examples'].append({
                    'id': ex_id,
                    'question': baseline['question'],
                    'ground_truth': baseline['ground_truth'],
                    'baseline_prediction': baseline['prediction'],
                    'model_prediction': model['prediction']
                })
        else:
            stats['still_correct'] += 1
    
    # Calculate metrics
    baseline_accuracy = stats['baseline_correct'] / stats['total'] * 100
    model_accuracy = stats['model_correct'] / stats['total'] * 100
    improvement = model_accuracy - baseline_accuracy
    
    correction_rate = stats['corrections'] / stats['baseline_incorrect'] * 100 if stats['baseline_incorrect'] > 0 else 0
    regression_rate = stats['regressions'] / stats['baseline_correct'] * 100 if stats['baseline_correct'] > 0 else 0
    
    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    print(f"Overall Performance:")
    print(f"  Baseline:  {stats['baseline_correct']:4d}/{stats['total']:4d} = {baseline_accuracy:5.2f}%")
    print(f"  Model:     {stats['model_correct']:4d}/{stats['total']:4d} = {model_accuracy:5.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    print()
    
    print(f"Error Corrections:")
    print(f"  Baseline errors: {stats['baseline_incorrect']}")
    print(f"  Corrected:       {stats['corrections']} ({correction_rate:.1f}%)")
    print(f"  Still wrong:     {stats['still_wrong']} ({stats['still_wrong']/stats['baseline_incorrect']*100:.1f}%)")
    print()
    
    print(f"Regressions:")
    print(f"  Baseline correct: {stats['baseline_correct']}")
    print(f"  Regressed:        {stats['regressions']} ({regression_rate:.1f}%)")
    print(f"  Still correct:    {stats['still_correct']} ({stats['still_correct']/stats['baseline_correct']*100:.1f}%)")
    print()
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Show examples
    print("=" * 70)
    print("SAMPLE CORRECTIONS (Baseline Wrong → Model Correct)")
    print("=" * 70)
    
    for i, ex in enumerate(stats['correction_examples'][:5], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {ex['question']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"Baseline: {ex['baseline_prediction']} ❌")
        print(f"Model:    {ex['model_prediction']} ✅")
    
    if stats['regression_examples']:
        print()
        print("=" * 70)
        print("SAMPLE REGRESSIONS (Baseline Correct → Model Wrong)")
        print("=" * 70)
        
        for i, ex in enumerate(stats['regression_examples'][:3], 1):
            print(f"\n--- Example {i} ---")
            print(f"Question: {ex['question']}")
            print(f"Ground Truth: {ex['ground_truth']}")
            print(f"Baseline: {ex['baseline_prediction']} ✅")
            print(f"Model:    {ex['model_prediction']} ❌")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze pattern improvements")
    parser.add_argument('--baseline', type=str, 
                       default='./evaluation/adversarial_squad/eval_predictions.jsonl',
                       help='Baseline predictions file')
    parser.add_argument('--model', type=str,
                       default='./evaluation/adversarial_70_30/eval_predictions.jsonl',
                       help='Model predictions file (default: 70-30)')
    parser.add_argument('--output', type=str,
                       default='./evaluation/pattern_improvements.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    analyze_improvements(args.baseline, args.model, args.output)
