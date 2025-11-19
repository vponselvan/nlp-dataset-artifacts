#!/usr/bin/env python3
"""
Compare baseline vs both adversarially trained models (50-50 and 80-20).
"""

import json
from pathlib import Path

def load_metrics(path):
    """Load evaluation metrics"""
    if not Path(path).exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def print_model_results(name, addsent_metrics, squad_metrics):
    """Print results for a single model"""
    if addsent_metrics and squad_metrics:
        print(f"\n{name}:")
        print(f"  AddSent - EM: {addsent_metrics['eval_exact_match']:6.2f}%  F1: {addsent_metrics['eval_f1']:6.2f}%")
        print(f"  SQuAD   - EM: {squad_metrics['eval_exact_match']:6.2f}%  F1: {squad_metrics['eval_f1']:6.2f}%")
    else:
        print(f"\n{name}: ❌ Not found")

def compare_all_models():
    """Compare all models"""
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON - 5 RATIOS")
    print("=" * 80)
    
    # Load all results from evaluation/ folder
    baseline_addsent = load_metrics('./evaluation/adversarial_squad/eval_metrics.json')
    baseline_squad = load_metrics('./evaluation/squad/eval_metrics.json')
    
    model_90_10_addsent = load_metrics('./evaluation/adversarial_90_10/eval_metrics.json')
    model_90_10_squad = load_metrics('./evaluation/adversarial_90_10/squad/eval_metrics.json')
    
    model_80_20_addsent = load_metrics('./evaluation/adversarial_80_20/eval_metrics.json')
    model_80_20_squad = load_metrics('./evaluation/adversarial_80_20/squad/eval_metrics.json')
    
    model_70_30_addsent = load_metrics('./evaluation/adversarial_70_30/eval_metrics.json')
    model_70_30_squad = load_metrics('./evaluation/adversarial_70_30/squad/eval_metrics.json')
    
    model_60_40_addsent = load_metrics('./evaluation/adversarial_60_40/eval_metrics.json')
    model_60_40_squad = load_metrics('./evaluation/adversarial_60_40/squad/eval_metrics.json')
    
    model_50_50_addsent = load_metrics('./evaluation/adversarial_50_50/eval_metrics.json')
    model_50_50_squad = load_metrics('./evaluation/adversarial_50_50/squad/eval_metrics.json')
    
    # Print all results
    print("\n" + "=" * 80)
    print("RAW RESULTS")
    print("=" * 80)
    
    print_model_results("Baseline (Original Model)", baseline_addsent, baseline_squad)
    print_model_results("90-10 Split (90% SQuAD + 10% AddSent)", model_90_10_addsent, model_90_10_squad)
    print_model_results("80-20 Split (80% SQuAD + 20% AddSent)", model_80_20_addsent, model_80_20_squad)
    print_model_results("70-30 Split (70% SQuAD + 30% AddSent)", model_70_30_addsent, model_70_30_squad)
    print_model_results("60-40 Split (60% SQuAD + 40% AddSent)", model_60_40_addsent, model_60_40_squad)
    print_model_results("50-50 Split (50% SQuAD + 50% AddSent)", model_50_50_addsent, model_50_50_squad)
    
    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "AddSent EM", "AddSent F1", "SQuAD EM", "SQuAD F1"
    ))
    print("-" * 80)
    
    if baseline_addsent and baseline_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "Baseline",
            baseline_addsent['eval_exact_match'],
            baseline_addsent['eval_f1'],
            baseline_squad['eval_exact_match'],
            baseline_squad['eval_f1']
        ))
    
    if model_90_10_addsent and model_90_10_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "90-10 Split",
            model_90_10_addsent['eval_exact_match'],
            model_90_10_addsent['eval_f1'],
            model_90_10_squad['eval_exact_match'],
            model_90_10_squad['eval_f1']
        ))
    
    if model_80_20_addsent and model_80_20_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "80-20 Split",
            model_80_20_addsent['eval_exact_match'],
            model_80_20_addsent['eval_f1'],
            model_80_20_squad['eval_exact_match'],
            model_80_20_squad['eval_f1']
        ))
    
    if model_70_30_addsent and model_70_30_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "70-30 Split",
            model_70_30_addsent['eval_exact_match'],
            model_70_30_addsent['eval_f1'],
            model_70_30_squad['eval_exact_match'],
            model_70_30_squad['eval_f1']
        ))
    
    if model_60_40_addsent and model_60_40_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "60-40 Split",
            model_60_40_addsent['eval_exact_match'],
            model_60_40_addsent['eval_f1'],
            model_60_40_squad['eval_exact_match'],
            model_60_40_squad['eval_f1']
        ))
    
    if model_50_50_addsent and model_50_50_squad:
        print("{:<25} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            "50-50 Split",
            model_50_50_addsent['eval_exact_match'],
            model_50_50_addsent['eval_f1'],
            model_50_50_squad['eval_exact_match'],
            model_50_50_squad['eval_f1']
        ))
    
    # Improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENTS vs BASELINE")
    print("=" * 80)
    
    print("\n{:<25} {:>15} {:>15} {:>15} {:>15}".format(
        "Model", "AddSent EM Δ", "AddSent F1 Δ", "SQuAD EM Δ", "SQuAD F1 Δ"
    ))
    print("-" * 80)
    
    if baseline_addsent and baseline_squad:
        models = [
            ("90-10 Split", model_90_10_addsent, model_90_10_squad),
            ("80-20 Split", model_80_20_addsent, model_80_20_squad),
            ("70-30 Split", model_70_30_addsent, model_70_30_squad),
            ("60-40 Split", model_60_40_addsent, model_60_40_squad),
            ("50-50 Split", model_50_50_addsent, model_50_50_squad),
        ]
        
        for name, addsent_metrics, squad_metrics in models:
            if addsent_metrics and squad_metrics:
                addsent_em_delta = addsent_metrics['eval_exact_match'] - baseline_addsent['eval_exact_match']
                addsent_f1_delta = addsent_metrics['eval_f1'] - baseline_addsent['eval_f1']
                squad_em_delta = squad_metrics['eval_exact_match'] - baseline_squad['eval_exact_match']
                squad_f1_delta = squad_metrics['eval_f1'] - baseline_squad['eval_f1']
                
                print("{:<25} {:>14.2f}% {:>14.2f}% {:>14.2f}% {:>14.2f}%".format(
                    name,
                    addsent_em_delta,
                    addsent_f1_delta,
                    squad_em_delta,
                    squad_f1_delta
                ))
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS - TRADE-OFF FOR EACH RATIO")
    print("=" * 80)
    
    if baseline_addsent and baseline_squad:
        models = [
            ("90-10", model_90_10_addsent, model_90_10_squad),
            ("80-20", model_80_20_addsent, model_80_20_squad),
            ("70-30", model_70_30_addsent, model_70_30_squad),
            ("60-40", model_60_40_addsent, model_60_40_squad),
            ("50-50", model_50_50_addsent, model_50_50_squad),
        ]
        
        for name, addsent_metrics, squad_metrics in models:
            if addsent_metrics and squad_metrics:
                addsent_gain = addsent_metrics['eval_exact_match'] - baseline_addsent['eval_exact_match']
                squad_cost = baseline_squad['eval_exact_match'] - squad_metrics['eval_exact_match']
                ratio = addsent_gain / max(squad_cost, 0.01)
                
                print(f"\n{name} Split:")
                print(f"  Robustness Gain: +{addsent_gain:.2f}% on adversarial")
                print(f"  Clean Cost:      {'-' if squad_cost > 0 else '+'}{abs(squad_cost):.2f}% on clean")
                print(f"  Trade-off Ratio: {ratio:.2f}x")
                
                if addsent_gain > 20 and squad_cost < 5:
                    print(f"  Assessment: ✅ Excellent - High robustness gain with minimal cost")
                elif addsent_gain > 15 and squad_cost < 10:
                    print(f"  Assessment: ✅ Good - Strong robustness gain with acceptable cost")
                elif addsent_gain > 10:
                    print(f"  Assessment: ⚠️ High robustness but significant clean performance cost")
                else:
                    print(f"  Assessment: ⚠️ Limited improvement")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION - BEST TRADE-OFF")
    print("=" * 80)
    
    if baseline_addsent and baseline_squad:
        models_data = [
            ("90-10", model_90_10_addsent, model_90_10_squad),
            ("80-20", model_80_20_addsent, model_80_20_squad),
            ("70-30", model_70_30_addsent, model_70_30_squad),
            ("60-40", model_60_40_addsent, model_60_40_squad),
            ("50-50", model_50_50_addsent, model_50_50_squad),
        ]
        
        # Calculate trade-off ratios for all models
        ratios = []
        for name, addsent_metrics, squad_metrics in models_data:
            if addsent_metrics and squad_metrics:
                addsent_gain = addsent_metrics['eval_exact_match'] - baseline_addsent['eval_exact_match']
                squad_cost = baseline_squad['eval_exact_match'] - squad_metrics['eval_exact_match']
                ratio = addsent_gain / max(squad_cost, 0.01)
                ratios.append((name, ratio, addsent_gain, squad_cost, addsent_metrics['eval_exact_match']))
        
        if ratios:
            # Sort by trade-off ratio (descending)
            ratios.sort(key=lambda x: x[1], reverse=True)
            best = ratios[0]
            
            print(f"\n✅ {best[0]} Split is RECOMMENDED")
            print(f"   Best trade-off ratio: {best[1]:.2f}x")
            print(f"   Robustness gain: +{best[2]:.1f}% on adversarial")
            print(f"   Clean cost: -{best[3]:.1f}% on clean")
            print(f"   Final adversarial performance: {best[4]:.2f}% EM")
            
            print(f"\n📊 Trade-off Ranking:")
            for i, (name, ratio, gain, cost, _) in enumerate(ratios, 1):
                print(f"   {i}. {name}: {ratio:.2f}x (gain: +{gain:.1f}%, cost: -{cost:.1f}%)")
    
    print("\n" + "=" * 80)
    
    # Save results to JSON file for visualization
    if baseline_addsent and baseline_squad:
        models_data = [
            ("90-10", model_90_10_addsent, model_90_10_squad),
            ("80-20", model_80_20_addsent, model_80_20_squad),
            ("70-30", model_70_30_addsent, model_70_30_squad),
            ("60-40", model_60_40_addsent, model_60_40_squad),
            ("50-50", model_50_50_addsent, model_50_50_squad),
        ]
        save_comparison_results(baseline_addsent, baseline_squad, models_data)

def save_comparison_results(baseline_addsent, baseline_squad, models_data, output_file='./evaluation/comparison_results.json'):
    """Save comparison results to JSON file for visualization"""
    import json
    from pathlib import Path
    
    results = {
        "baseline": {
            "addsent_em": baseline_addsent['eval_exact_match'] if baseline_addsent else None,
            "addsent_f1": baseline_addsent['eval_f1'] if baseline_addsent else None,
            "squad_em": baseline_squad['eval_exact_match'] if baseline_squad else None,
            "squad_f1": baseline_squad['eval_f1'] if baseline_squad else None,
        },
        "experiments": []
    }
    
    for name, addsent_metrics, squad_metrics in models_data:
        if addsent_metrics and squad_metrics and baseline_addsent and baseline_squad:
            addsent_gain = addsent_metrics['eval_exact_match'] - baseline_addsent['eval_exact_match']
            squad_cost = baseline_squad['eval_exact_match'] - squad_metrics['eval_exact_match']
            trade_off_ratio = addsent_gain / max(squad_cost, 0.01)
            
            results["experiments"].append({
                "name": name,
                "adversarial_pct": int(name.split("-")[1]),
                "addsent_em": addsent_metrics['eval_exact_match'],
                "addsent_f1": addsent_metrics['eval_f1'],
                "squad_em": squad_metrics['eval_exact_match'],
                "squad_f1": squad_metrics['eval_f1'],
                "addsent_gain": addsent_gain,
                "squad_cost": squad_cost,
                "trade_off_ratio": trade_off_ratio
            })
    
    # Sort by adversarial percentage
    results["experiments"].sort(key=lambda x: x["adversarial_pct"])
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {output_file}")
    print("   Use this file for creating visualizations!")
    
    return results

if __name__ == "__main__":
    compare_all_models()
