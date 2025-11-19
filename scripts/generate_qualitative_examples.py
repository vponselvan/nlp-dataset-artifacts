#!/usr/bin/env python3
"""
Generate qualitative before-and-after examples for paper.
Shows specific cases where adversarial training helped or failed.
Can compare all 5 models at once or a single model.
"""

import json
import random
from pathlib import Path

def load_predictions(pred_file):
    """Load predictions"""
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions

def is_correct(pred, gt):
    """Check correctness"""
    return pred.lower().strip() == gt.lower().strip()

def extract_model_name(file_path):
    """Extract model name from file path"""
    if 'adversarial_squad' in file_path or 'squad' in file_path:
        return 'Baseline'
    elif '90_10' in file_path:
        return '90-10'
    elif '80_20' in file_path:
        return '80-20'
    elif '70_30' in file_path:
        return '70-30'
    elif '60_40' in file_path:
        return '60-40'
    elif '50_50' in file_path:
        return '50-50'
    return 'Model'

def find_interesting_examples(baseline_file, model_file, output_file='./evaluation/qualitative_examples.json', model_name=None):
    """Find interesting before/after examples"""
    
    if model_name is None:
        model_name = extract_model_name(model_file)
    
    print(f"Comparing: Baseline vs {model_name}")
    print("Loading predictions...")
    baseline = {p['id']: p for p in load_predictions(baseline_file)}
    model = {p['id']: p for p in load_predictions(model_file)}
    
    common_ids = set(baseline.keys()) & set(model.keys())
    
    examples = {
        'model_name': model_name,
        'corrections': [],  # baseline wrong → model correct
        'regressions': [],  # baseline correct → model wrong
        'persistent_errors': [],  # both wrong, different predictions
        'persistent_correct': []  # both correct
    }
    
    for ex_id in common_ids:
        b = baseline[ex_id]
        m = model[ex_id]
        
        gt = b['answers']['text'][0]
        b_pred = b.get('predicted_answer', b.get('prediction_text', ''))
        m_pred = m.get('predicted_answer', m.get('prediction_text', ''))
        
        b_correct = is_correct(b_pred, gt)
        m_correct = is_correct(m_pred, gt)
        
        example = {
            'id': ex_id,
            'question': b['question'],
            'context': b['context'],
            'ground_truth': gt,
            'baseline_prediction': b_pred,
            'model_prediction': m_pred
        }
        
        if not b_correct and m_correct:
            examples['corrections'].append(example)
        elif b_correct and not m_correct:
            examples['regressions'].append(example)
        elif not b_correct and not m_correct and b_pred != m_pred:
            examples['persistent_errors'].append(example)
        elif b_correct and m_correct:
            examples['persistent_correct'].append(example)
    
    # Sample interesting examples
    print(f"\nFound:")
    print(f"  Corrections: {len(examples['corrections'])}")
    print(f"  Regressions: {len(examples['regressions'])}")
    print(f"  Persistent errors: {len(examples['persistent_errors'])}")
    print(f"  Persistent correct: {len(examples['persistent_correct'])}")
    
    # Save all
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Display samples
    print("\n" + "=" * 70)
    print(f"SAMPLE CORRECTIONS (Baseline → {model_name})")
    print("=" * 70)
    
    for i, ex in enumerate(random.sample(examples['corrections'], min(5, len(examples['corrections']))), 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {ex['question']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"Baseline: {ex['baseline_prediction']} ❌")
        print(f"{model_name}: {ex['model_prediction']} ✅")
        print(f"Context: {ex['context'][:150]}...")
    
    return examples

def compare_all_models(baseline_file, output_dir='./evaluation'):
    """Compare all 5 models against baseline"""
    
    models = [
        ('90-10', './evaluation/adversarial_90_10/eval_predictions.jsonl'),
        ('80-20', './evaluation/adversarial_80_20/eval_predictions.jsonl'),
        ('70-30', './evaluation/adversarial_70_30/eval_predictions.jsonl'),
        ('60-40', './evaluation/adversarial_60_40/eval_predictions.jsonl'),
        ('50-50', './evaluation/adversarial_50_50/eval_predictions.jsonl'),
    ]
    
    all_results = {}
    
    print("=" * 70)
    print("COMPARING ALL 5 MODELS")
    print("=" * 70)
    print()
    
    for model_name, model_file in models:
        if not Path(model_file).exists():
            print(f"⚠️  Skipping {model_name}: file not found")
            continue
        
        output_file = f"{output_dir}/qualitative_examples_{model_name.replace('-', '_')}.json"
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        examples = find_interesting_examples(baseline_file, model_file, output_file, model_name)
        
        all_results[model_name] = {
            'corrections': len(examples['corrections']),
            'regressions': len(examples['regressions']),
            'persistent_errors': len(examples['persistent_errors']),
            'persistent_correct': len(examples['persistent_correct']),
            'output_file': output_file
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ALL MODELS")
    print("=" * 70)
    print()
    print(f"{'Model':<10} {'Corrections':<15} {'Regressions':<15} {'Net Gain':<10}")
    print("-" * 70)
    
    for model_name in ['90-10', '80-20', '70-30', '60-40', '50-50']:
        if model_name in all_results:
            r = all_results[model_name]
            net_gain = r['corrections'] - r['regressions']
            print(f"{model_name:<10} {r['corrections']:<15} {r['regressions']:<15} {net_gain:<10}")
    
    # Save summary
    summary_file = f"{output_dir}/qualitative_examples_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print(f"💾 Summary saved to: {summary_file}")
    print()
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate qualitative examples for paper")
    parser.add_argument('--baseline', type=str,
                       default='./evaluation/adversarial_squad/eval_predictions.jsonl',
                       help='Baseline predictions file')
    parser.add_argument('--model', type=str,
                       default='./evaluation/adversarial_70_30/eval_predictions.jsonl',
                       help='Model predictions file (ignored if --all is used)')
    parser.add_argument('--output', type=str,
                       default='./evaluation/qualitative_examples.json',
                       help='Output file for results')
    parser.add_argument('--all', action='store_true',
                       help='Compare all 5 models at once')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models(args.baseline)
    else:
        find_interesting_examples(args.baseline, args.model, args.output)
