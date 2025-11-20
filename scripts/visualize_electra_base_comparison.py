#!/usr/bin/env python3
"""
Visualize ELECTRA-small vs ELECTRA-base comparison
Creates publication-quality plots showing the impact of model scaling
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_metrics(eval_dir):
    """Load evaluation metrics from a directory"""
    metrics_file = Path(eval_dir) / "eval_metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_comparison_plots():
    """Create comprehensive comparison visualizations"""
    
    # Load all results
    models_data = {
        "Baseline\n(ELECTRA-small)": {
            "addsent": "evaluation/adversarial_squad",
            "squad": "evaluation/squad",
            "params": "14M",
            "color": "#e74c3c"
        },
        "80-20 Original\n(ELECTRA-small)": {
            "addsent": "evaluation/adversarial_80_20",
            "squad": "evaluation/adversarial_80_20",
            "params": "14M",
            "color": "#3498db"
        },
        "80-20 Augmented\n(ELECTRA-small)": {
            "addsent": "evaluation/adversarial_80_20_augmented",
            "squad": "evaluation/adversarial_80_20_augmented",
            "params": "14M",
            "color": "#9b59b6"
        },
        "80-20 Augmented\n(ELECTRA-base)": {
            "addsent": "evaluation/electra_base_80_20_augmented/addsent",
            "squad": "evaluation/electra_base_80_20_augmented/squad",
            "params": "110M",
            "color": "#27ae60"
        }
    }
    
    results = {}
    for model_name, config in models_data.items():
        addsent_metrics = load_metrics(config["addsent"])
        squad_metrics = load_metrics(config["squad"])
        
        if addsent_metrics and squad_metrics:
            results[model_name] = {
                "addsent_em": addsent_metrics.get("eval_exact_match", 0),
                "addsent_f1": addsent_metrics.get("eval_f1", 0),
                "squad_em": squad_metrics.get("eval_exact_match", 0),
                "squad_f1": squad_metrics.get("eval_f1", 0),
                "params": config["params"],
                "color": config["color"]
            }
    
    if not results:
        print("❌ No results found. Please run evaluations first.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Exact Match Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    models = list(results.keys())
    addsent_scores = [results[m]["addsent_em"] for m in models]
    squad_scores = [results[m]["squad_em"] for m in models]
    colors = [results[m]["color"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, addsent_scores, width, label='AddSent (Adversarial)', 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, squad_scores, width, label='SQuAD (Clean)',
                    color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Exact Match (%)', fontweight='bold')
    ax1.set_title('Exact Match Performance Comparison', fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: F1 Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    addsent_f1 = [results[m]["addsent_f1"] for m in models]
    squad_f1 = [results[m]["squad_f1"] for m in models]
    
    bars1 = ax2.bar(x - width/2, addsent_f1, width, label='AddSent (Adversarial)',
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, squad_f1, width, label='SQuAD (Clean)',
                    color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('F1 Score (%)', fontweight='bold')
    ax2.set_title('F1 Score Performance Comparison', fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Improvement Over Baseline
    ax3 = plt.subplot(2, 3, 3)
    baseline_addsent = results["Baseline\n(ELECTRA-small)"]["addsent_em"]
    baseline_squad = results["Baseline\n(ELECTRA-small)"]["squad_em"]
    
    improvements_addsent = [results[m]["addsent_em"] - baseline_addsent for m in models[1:]]
    improvements_squad = [results[m]["squad_em"] - baseline_squad for m in models[1:]]
    
    x_imp = np.arange(len(models[1:]))
    bars1 = ax3.bar(x_imp - width/2, improvements_addsent, width, 
                    label='AddSent Improvement', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x_imp + width/2, improvements_squad, width,
                    label='SQuAD Improvement', color='#3498db', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Improvement over Baseline (%)', fontweight='bold')
    ax3.set_title('Improvement Over Baseline', fontweight='bold', pad=20)
    ax3.set_xticks(x_imp)
    ax3.set_xticklabels(models[1:], rotation=15, ha='right')
    ax3.legend(loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    # Plot 4: Trade-off Analysis (Scatter Plot)
    ax4 = plt.subplot(2, 3, 4)
    
    for model in models:
        ax4.scatter(results[model]["squad_em"], results[model]["addsent_em"],
                   s=300, color=results[model]["color"], alpha=0.7,
                   edgecolors='black', linewidth=2, label=model)
        ax4.annotate(model, 
                    (results[model]["squad_em"], results[model]["addsent_em"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('SQuAD EM (Clean Performance) %', fontweight='bold')
    ax4.set_ylabel('AddSent EM (Adversarial Robustness) %', fontweight='bold')
    ax4.set_title('Robustness vs Clean Performance Trade-off', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Balance')
    
    # Plot 5: Model Capacity Impact
    ax5 = plt.subplot(2, 3, 5)
    
    small_models = [m for m in models if "ELECTRA-small" in m]
    base_models = [m for m in models if "ELECTRA-base" in m]
    
    small_addsent_avg = np.mean([results[m]["addsent_em"] for m in small_models])
    small_squad_avg = np.mean([results[m]["squad_em"] for m in small_models])
    
    if base_models:
        base_addsent_avg = np.mean([results[m]["addsent_em"] for m in base_models])
        base_squad_avg = np.mean([results[m]["squad_em"] for m in base_models])
        
        categories = ['AddSent\n(Adversarial)', 'SQuAD\n(Clean)']
        small_scores = [small_addsent_avg, small_squad_avg]
        base_scores = [base_addsent_avg, base_squad_avg]
        
        x_cap = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax5.bar(x_cap - width/2, small_scores, width, 
                       label='ELECTRA-small (14M)', color='#3498db', 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax5.bar(x_cap + width/2, base_scores, width,
                       label='ELECTRA-base (110M)', color='#27ae60',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax5.set_ylabel('Average Exact Match (%)', fontweight='bold')
        ax5.set_title('Model Capacity Impact (Average Performance)', fontweight='bold', pad=20)
        ax5.set_xticks(x_cap)
        ax5.set_xticklabels(categories)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim([0, 100])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 6: Progressive Improvement Timeline
    ax6 = plt.subplot(2, 3, 6)
    
    stages = ['Baseline', '80-20\nOriginal', '80-20\nAugmented\n(Small)', '80-20\nAugmented\n(Base)']
    addsent_progression = [
        results["Baseline\n(ELECTRA-small)"]["addsent_em"],
        results["80-20 Original\n(ELECTRA-small)"]["addsent_em"],
        results["80-20 Augmented\n(ELECTRA-small)"]["addsent_em"],
        results["80-20 Augmented\n(ELECTRA-base)"]["addsent_em"]
    ]
    squad_progression = [
        results["Baseline\n(ELECTRA-small)"]["squad_em"],
        results["80-20 Original\n(ELECTRA-small)"]["squad_em"],
        results["80-20 Augmented\n(ELECTRA-small)"]["squad_em"],
        results["80-20 Augmented\n(ELECTRA-base)"]["squad_em"]
    ]
    
    x_prog = np.arange(len(stages))
    ax6.plot(x_prog, addsent_progression, marker='o', linewidth=3, markersize=10,
            label='AddSent (Adversarial)', color='#e74c3c')
    ax6.plot(x_prog, squad_progression, marker='s', linewidth=3, markersize=10,
            label='SQuAD (Clean)', color='#3498db')
    
    ax6.set_ylabel('Exact Match (%)', fontweight='bold')
    ax6.set_title('Progressive Improvement Timeline', fontweight='bold', pad=20)
    ax6.set_xticks(x_prog)
    ax6.set_xticklabels(stages, rotation=15, ha='right')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([40, 95])
    
    # Add value labels
    for i, (addsent, squad) in enumerate(zip(addsent_progression, squad_progression)):
        ax6.text(i, addsent + 1, f'{addsent:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax6.text(i, squad - 2, f'{squad:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("evaluation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "electra_base_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive comparison plot to: {output_file}")
    
    # Create individual high-quality plots for paper
    create_paper_plots(results, models, output_dir)
    
    plt.show()

def create_paper_plots(results, models, output_dir):
    """Create individual publication-quality plots for paper"""
    
    # Plot 1: Simple bar chart for paper
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    addsent_scores = [results[m]["addsent_em"] for m in models]
    squad_scores = [results[m]["squad_em"] for m in models]
    
    bars1 = ax.bar(x - width/2, addsent_scores, width, label='AddSent (Adversarial)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, squad_scores, width, label='SQuAD (Clean)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Exact Match (%)', fontweight='bold', fontsize=14)
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "paper_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved paper plot to: {output_dir / 'paper_performance_comparison.png'}")
    plt.close()
    
    # Plot 2: Improvement plot for paper
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_addsent = results["Baseline\n(ELECTRA-small)"]["addsent_em"]
    baseline_squad = results["Baseline\n(ELECTRA-small)"]["squad_em"]
    
    improvements_addsent = [results[m]["addsent_em"] - baseline_addsent for m in models[1:]]
    improvements_squad = [results[m]["squad_em"] - baseline_squad for m in models[1:]]
    
    x_imp = np.arange(len(models[1:]))
    width = 0.35
    
    bars1 = ax.bar(x_imp - width/2, improvements_addsent, width,
                   label='AddSent Improvement', color='#e74c3c', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_imp + width/2, improvements_squad, width,
                   label='SQuAD Improvement', color='#3498db', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_ylabel('Improvement over Baseline (%)', fontweight='bold', fontsize=14)
    ax.set_title('Performance Improvement Analysis', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x_imp)
    ax.set_xticklabels(models[1:], rotation=15, ha='right', fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "paper_improvement_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved improvement plot to: {output_dir / 'paper_improvement_analysis.png'}")
    plt.close()

def main():
    print("=" * 80)
    print("Creating ELECTRA-base Comparison Visualizations")
    print("=" * 80)
    print()
    
    create_comparison_plots()
    
    print()
    print("=" * 80)
    print("✅ All visualizations created successfully!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  1. evaluation/plots/electra_base_comparison.png (comprehensive)")
    print("  2. evaluation/plots/paper_performance_comparison.png (for paper)")
    print("  3. evaluation/plots/paper_improvement_analysis.png (for paper)")
    print()

if __name__ == "__main__":
    main()
