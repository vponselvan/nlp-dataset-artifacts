#!/usr/bin/env python3
"""
Compare ELECTRA-small vs ELECTRA-base models
Shows the impact of model scaling on adversarial robustness
"""

import json
import os
from pathlib import Path

def load_metrics(eval_dir):
    """Load evaluation metrics from a directory"""
    metrics_file = Path(eval_dir) / "eval_metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def main():
    print("=" * 80)
    print("ELECTRA-small vs ELECTRA-base Comparison")
    print("=" * 80)
    print()
    
    # Define model configurations to compare
    models = {
        "ELECTRA-small (baseline)": {
            "addsent": "evaluation/adversarial_squad",
            "squad": "evaluation/squad"
        },
        "ELECTRA-small (80-20 original)": {
            "addsent": "evaluation/adversarial_80_20",
            "squad": "evaluation/adversarial_80_20"
        },
        "ELECTRA-small (80-20 augmented)": {
            "addsent": "evaluation/adversarial_80_20_augmented",
            "squad": "evaluation/adversarial_80_20_augmented"
        },
        "ELECTRA-base (80-20 augmented)": {
            "addsent": "evaluation/electra_base_80_20_augmented/addsent",
            "squad": "evaluation/electra_base_80_20_augmented/squad"
        }
    }
    
    results = {}
    
    # Load all results
    for model_name, paths in models.items():
        addsent_metrics = load_metrics(paths["addsent"])
        squad_metrics = load_metrics(paths["squad"])
        
        if addsent_metrics and squad_metrics:
            results[model_name] = {
                "addsent_em": addsent_metrics.get("eval_exact_match", 0),
                "addsent_f1": addsent_metrics.get("eval_f1", 0),
                "squad_em": squad_metrics.get("eval_exact_match", 0),
                "squad_f1": squad_metrics.get("eval_f1", 0)
            }
        else:
            print(f"⚠️  Warning: Could not load metrics for {model_name}")
            print(f"    AddSent: {paths['addsent']}")
            print(f"    SQuAD: {paths['squad']}")
            print()
    
    if not results:
        print("❌ No results found. Please run evaluations first.")
        return
    
    # Print comparison table
    print("\n📊 Performance Comparison")
    print("-" * 80)
    print(f"{'Model':<40} {'AddSent EM':<12} {'AddSent F1':<12} {'SQuAD EM':<12} {'SQuAD F1':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<40} "
              f"{metrics['addsent_em']:>10.2f}%  "
              f"{metrics['addsent_f1']:>10.2f}%  "
              f"{metrics['squad_em']:>10.2f}%  "
              f"{metrics['squad_f1']:>10.2f}%")
    
    print("-" * 80)
    print()
    
    # Calculate improvements
    if "ELECTRA-small (baseline)" in results and "ELECTRA-base (80-20 augmented)" in results:
        baseline = results["ELECTRA-small (baseline)"]
        electra_base = results["ELECTRA-base (80-20 augmented)"]
        
        print("\n🎯 ELECTRA-base vs Baseline")
        print("-" * 80)
        print(f"AddSent EM: {baseline['addsent_em']:.2f}% → {electra_base['addsent_em']:.2f}% "
              f"(+{electra_base['addsent_em'] - baseline['addsent_em']:.2f}%)")
        print(f"SQuAD EM:   {baseline['squad_em']:.2f}% → {electra_base['squad_em']:.2f}% "
              f"({electra_base['squad_em'] - baseline['squad_em']:+.2f}%)")
        print()
    
    if "ELECTRA-small (80-20 augmented)" in results and "ELECTRA-base (80-20 augmented)" in results:
        small_aug = results["ELECTRA-small (80-20 augmented)"]
        base_aug = results["ELECTRA-base (80-20 augmented)"]
        
        print("\n🚀 ELECTRA-base vs ELECTRA-small (both with augmentation)")
        print("-" * 80)
        print(f"AddSent EM: {small_aug['addsent_em']:.2f}% → {base_aug['addsent_em']:.2f}% "
              f"(+{base_aug['addsent_em'] - small_aug['addsent_em']:.2f}%)")
        print(f"SQuAD EM:   {small_aug['squad_em']:.2f}% → {base_aug['squad_em']:.2f}% "
              f"(+{base_aug['squad_em'] - small_aug['squad_em']:.2f}%)")
        print()
        
        # Check if we hit the target
        target_min = 72.0
        target_max = 76.0
        actual = base_aug['addsent_em']
        
        print("\n🎯 Target Achievement")
        print("-" * 80)
        print(f"Target Range: {target_min:.0f}-{target_max:.0f}% EM on AddSent")
        print(f"Actual:       {actual:.2f}% EM on AddSent")
        
        if actual >= target_min and actual <= target_max:
            print(f"✅ SUCCESS! Hit the target range!")
        elif actual > target_max:
            print(f"🎉 EXCEEDED! Beat the target by {actual - target_max:.2f}%!")
        else:
            print(f"⚠️  Below target by {target_min - actual:.2f}%")
        print()
    
    # Model capacity analysis
    if "ELECTRA-base (80-20 augmented)" in results:
        print("\n💡 Model Capacity Analysis")
        print("-" * 80)
        print("ELECTRA-small: 14M parameters")
        print("ELECTRA-base:  110M parameters (8x larger)")
        print()
        
        base_aug = results["ELECTRA-base (80-20 augmented)"]
        print(f"With 8x more capacity:")
        print(f"  - Can handle both adversarial patterns AND generalization")
        print(f"  - Achieved {base_aug['addsent_em']:.2f}% on AddSent")
        print(f"  - Maintained {base_aug['squad_em']:.2f}% on SQuAD")
        print()
    
    # Save comparison results
    output_file = "evaluation/electra_base_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
