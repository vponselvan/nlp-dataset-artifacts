#!/usr/bin/env python3
"""
Generate qualitative before-and-after examples for paper.
Shows specific cases where adversarial training helped or failed.
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

def find_interesting_examples(baseline_file, model_file, output_file='./evaluation/qualitative_examples.json'):
    """Find interesting before/after examples"""
    
    print("Loading predictions...")
    baseline = {p['id']: p for p in load_predictions(baseline_file)}
    model = {p['id']: p for p in load_predictions(model_file)}
    
    common_ids = set(baseline.keys()) & set(model.keys())
    
    examples = {
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
    print("SAMPLE CORRECTIONS (for paper)")
    print("=" * 70)
    
    for i, ex in enumerate(random.sample(examples['corrections'], min(5, len(examples['corrections']))), 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {ex['question']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"Baseline: {ex['baseline_prediction']} ❌")
        print(f"70-30 Model: {ex['model_prediction']} ✅")
        print(f"Context: {ex['context'][:150]}...")
    
    return examples

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str,
                       default='./evaluation/adversarial_squad/eval_predictions.jsonl')
    parser.add_argument('--model', type=str,
                       default='./evaluation/adversarial_70_30/eval_predictions.jsonl')
    parser.add_argument('--output', type=str,
                       default='./evaluation/qualitative_examples.json')
    
    args = parser.parse_args()
    
    find_interesting_examples(args.baseline, args.model, args.output)
