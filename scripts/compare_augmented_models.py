#!/usr/bin/env python3
"""
Compare original vs augmented models across all 5 ratios.
Shows the impact of data augmentation on adversarial robustness.
"""

import json
from pathlib import Path


def load_metrics(path):
    """Load evaluation metrics"""
    if not Path(path).exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def compare_models():
    """Compare original vs augmented models"""
    
    print("=" * 80)
    print("ORIGINAL vs AUGMENTED MODELS COMPARISON")
    print("=" * 80)
    print()
    
    ratios = ["90_10", "80_20", "70_30", "60_40", "50_50"]
    
    # Load all results
    results = {}
    for ratio in ratios:
        results[ratio] = {
            'original': {
                'addsent': load_metrics(f'./evaluation/adversarial_{ratio}/eval_metrics.json'),
                'squad': load_metrics(f'./evaluation/adversarial_{ratio}/squad/eval_metrics.json')
            },
            'augmented': {
                'addsent': load_metrics(f'./evaluation/adversarial_{ratio}_augmented/eval_metrics.json'),
                'squad': load_metrics(f'./evaluation/adversarial_{ratio}_augmented/squad/eval_metrics.json')
            }
        }
    
    # Comparison table
    print("\n" + "=" * 80)
    print("ADVERSARIAL PERFORMANCE (AddSent EM)")
    print("=" * 80)
    print()
    print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(
        "Ratio", "Original", "Augmented", "Improvement", "Status"
    ))
    print("-" * 80)
    
    for ratio in ratios:
        orig = results[ratio]['original']['addsent']
        aug = results[ratio]['augmented']['addsent']
        
        if orig and aug:
            orig_em = orig['eval_exact_match']
            aug_em = aug['eval_exact_match']
            improvement = aug_em - orig_em
            
            status = "✅" if improvement > 0 else "⚠️"
            
            print("{:<15} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>12}".format(
                ratio.replace('_', '-'),
                orig_em,
                aug_em,
                improvement,
                status
            ))
        else:
            print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(
                ratio.replace('_', '-'),
                "N/A" if not orig else f"{orig['eval_exact_match']:.2f}%",
                "N/A" if not aug else f"{aug['eval_exact_match']:.2f}%",
                "N/A",
                "❌"
            ))
    
    # Clean performance
    print("\n" + "=" * 80)
    print("CLEAN PERFORMANCE (SQuAD EM)")
    print("=" * 80)
    print()
    print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(
        "Ratio", "Original", "Augmented", "Difference", "Status"
    ))
    print("-" * 80)
    
    for ratio in ratios:
        orig = results[ratio]['original']['squad']
        aug = results[ratio]['augmented']['squad']
        
        if orig and aug:
            orig_em = orig['eval_exact_match']
            aug_em = aug['eval_exact_match']
            diff = aug_em - orig_em
            
            status = "✅" if diff >= -2 else "⚠️"  # Allow small degradation
            
            print("{:<15} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>12}".format(
                ratio.replace('_', '-'),
                orig_em,
                aug_em,
                diff,
                status
            ))
        else:
            print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(
                ratio.replace('_', '-'),
                "N/A" if not orig else f"{orig['eval_exact_match']:.2f}%",
                "N/A" if not aug else f"{aug['eval_exact_match']:.2f}%",
                "N/A",
                "❌"
            ))
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    # Calculate average improvement
    improvements = []
    for ratio in ratios:
        orig = results[ratio]['original']['addsent']
        aug = results[ratio]['augmented']['addsent']
        if orig and aug:
            improvements.append(aug['eval_exact_match'] - orig['eval_exact_match'])
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"Average adversarial improvement: +{avg_improvement:.2f}%")
        print(f"Best improvement: +{max(improvements):.2f}%")
        print(f"Worst improvement: +{min(improvements):.2f}%")
    
    # Check if failed ratios now work
    print()
    print("Failed ratios (original < 55% EM) status:")
    for ratio in ["70_30", "60_40", "50_50"]:
        orig = results[ratio]['original']['addsent']
        aug = results[ratio]['augmented']['addsent']
        
        if orig and aug:
            orig_em = orig['eval_exact_match']
            aug_em = aug['eval_exact_match']
            
            if orig_em < 55:
                status = "✅ FIXED" if aug_em >= 60 else "⚠️ IMPROVED" if aug_em > orig_em else "❌ STILL FAILED"
                print(f"  {ratio.replace('_', '-')}: {orig_em:.2f}% → {aug_em:.2f}% ({status})")
    
    # Best model
    print()
    print("Best augmented model:")
    best_ratio = None
    best_em = 0
    
    for ratio in ratios:
        aug = results[ratio]['augmented']['addsent']
        if aug and aug['eval_exact_match'] > best_em:
            best_em = aug['eval_exact_match']
            best_ratio = ratio
    
    if best_ratio:
        aug_squad = results[best_ratio]['augmented']['squad']
        print(f"  {best_ratio.replace('_', '-')}: {best_em:.2f}% EM on AddSent")
        if aug_squad:
            print(f"  Clean performance: {aug_squad['eval_exact_match']:.2f}% EM on SQuAD")
    
    print()
    print("=" * 80)
    
    # Save comparison results
    save_comparison_results(results)


def save_comparison_results(results, output_file='./evaluation/augmented_comparison_results.json'):
    """Save comparison results to JSON"""
    from pathlib import Path
    
    comparison = {}
    
    for ratio, data in results.items():
        orig_addsent = data['original']['addsent']
        orig_squad = data['original']['squad']
        aug_addsent = data['augmented']['addsent']
        aug_squad = data['augmented']['squad']
        
        comparison[ratio] = {
            'original': {
                'addsent_em': orig_addsent['eval_exact_match'] if orig_addsent else None,
                'addsent_f1': orig_addsent['eval_f1'] if orig_addsent else None,
                'squad_em': orig_squad['eval_exact_match'] if orig_squad else None,
                'squad_f1': orig_squad['eval_f1'] if orig_squad else None,
            },
            'augmented': {
                'addsent_em': aug_addsent['eval_exact_match'] if aug_addsent else None,
                'addsent_f1': aug_addsent['eval_f1'] if aug_addsent else None,
                'squad_em': aug_squad['eval_exact_match'] if aug_squad else None,
                'squad_f1': aug_squad['eval_f1'] if aug_squad else None,
            }
        }
        
        # Calculate improvements
        if orig_addsent and aug_addsent:
            comparison[ratio]['improvement'] = {
                'addsent_em': aug_addsent['eval_exact_match'] - orig_addsent['eval_exact_match'],
                'addsent_f1': aug_addsent['eval_f1'] - orig_addsent['eval_f1'],
            }
        
        if orig_squad and aug_squad:
            comparison[ratio]['improvement']['squad_em'] = aug_squad['eval_exact_match'] - orig_squad['eval_exact_match']
            comparison[ratio]['improvement']['squad_f1'] = aug_squad['eval_f1'] - orig_squad['eval_f1']
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {output_file}")


if __name__ == "__main__":
    compare_models()
