#!/usr/bin/env python3
"""
Visualize complete comparison including baseline ELECTRA-base
Creates publication-quality plots showing all models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_comparison_data():
    """Load comparison data from JSON"""
    json_file = Path("evaluation/complete_comparison_results.json")
    if not json_file.exists():
        print(f"❌ File not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        return json.load(f)

def create_comprehensive_plots(data):
    """Create comprehensive comparison visualizations"""
    
    models = data["models"]
    model_names = [m["name"] for m in models]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Exact Match Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    addsent_scores = [m["addsent_em"] for m in models]
    squad_scores = [m["squad_em"] for m in models]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Use consistent colors: red for AddSent, blue for SQuAD
    bars1 = ax1.bar(x - width/2, addsent_scores, width, label='AddSent (Adversarial)', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, squad_scores, width, label='SQuAD (Clean)',
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Exact Match (%)', fontweight='bold')
    ax1.set_title('Exact Match Performance - All Models', fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 2: F1 Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    addsent_f1 = [m["addsent_f1"] for m in models]
    squad_f1 = [m["squad_f1"] for m in models]
    
    bars1 = ax2.bar(x - width/2, addsent_f1, width, label='AddSent (Adversarial)',
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, squad_f1, width, label='SQuAD (Clean)',
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('F1 Score (%)', fontweight='bold')
    ax2.set_title('F1 Score Performance - All Models', fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 3: Trade-off Analysis (Scatter Plot)
    ax3 = plt.subplot(2, 3, 3)
    
    for model in models:
        ax3.scatter(model["squad_em"], model["addsent_em"],
                   s=400, color=model["color"], alpha=0.7,
                   marker=model["marker"], edgecolors='black', linewidth=2,
                   label=model["name"])
    
    ax3.set_xlabel('SQuAD EM (Clean Performance) %', fontweight='bold')
    ax3.set_ylabel('AddSent EM (Adversarial Robustness) %', fontweight='bold')
    ax3.set_title('Robustness vs Clean Performance', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8)
    ax3.set_xlim([55, 95])
    ax3.set_ylim([45, 95])
    
    # Add diagonal line for reference
    ax3.plot([45, 95], [45, 95], 'k--', alpha=0.3, linewidth=1)
    
    # Plot 4: Improvement Over ELECTRA-small Baseline
    ax4 = plt.subplot(2, 3, 4)
    baseline_addsent = models[0]["addsent_em"]
    baseline_squad = models[0]["squad_em"]
    
    improvements_addsent = [m["addsent_em"] - baseline_addsent for m in models[1:]]
    improvements_squad = [m["squad_em"] - baseline_squad for m in models[1:]]
    
    x_imp = np.arange(len(models[1:]))
    bars1 = ax4.bar(x_imp - width/2, improvements_addsent, width, 
                    label='AddSent Improvement', color='#e74c3c', alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x_imp + width/2, improvements_squad, width,
                    label='SQuAD Improvement', color='#3498db', alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Improvement over Baseline (%)', fontweight='bold')
    ax4.set_title('Improvement Over ELECTRA-small Baseline', fontweight='bold', pad=20)
    ax4.set_xticks(x_imp)
    ax4.set_xticklabels([m["name"] for m in models[1:]], rotation=20, ha='right', fontsize=9)
    ax4.legend(loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    # Plot 5: Model Size Comparison
    ax5 = plt.subplot(2, 3, 5)
    
    small_models = [m for m in models if m["model_size"] == "14M"]
    base_models = [m for m in models if m["model_size"] == "110M"]
    
    categories = ['AddSent\n(Adversarial)', 'SQuAD\n(Clean)']
    
    # Average for small models
    small_addsent = np.mean([m["addsent_em"] for m in small_models])
    small_squad = np.mean([m["squad_em"] for m in small_models])
    
    # Average for base models
    base_addsent = np.mean([m["addsent_em"] for m in base_models])
    base_squad = np.mean([m["squad_em"] for m in base_models])
    
    x_size = np.arange(len(categories))
    bars1 = ax5.bar(x_size - width/2, [small_addsent, small_squad], width,
                   label='ELECTRA-small (14M)', color='#e74c3c',
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax5.bar(x_size + width/2, [base_addsent, base_squad], width,
                   label='ELECTRA-base (110M)', color='#27ae60',
                   alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax5.set_ylabel('Average Exact Match (%)', fontweight='bold')
    ax5.set_title('Model Capacity Impact (Average)', fontweight='bold', pad=20)
    ax5.set_xticks(x_size)
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
    
    # Plot 6: Progressive Improvement
    ax6 = plt.subplot(2, 3, 6)
    
    # Select key models for progression
    progression_models = [
        models[0],  # Baseline small
        models[1],  # Baseline base
        models[3],  # 80-20 aug small
        models[4]   # 80-20 aug base
    ]
    
    stages = [m["name"].split('(')[0].strip() for m in progression_models]
    addsent_prog = [m["addsent_em"] for m in progression_models]
    squad_prog = [m["squad_em"] for m in progression_models]
    
    x_prog = np.arange(len(stages))
    prog_colors = [progression_models[i]["color"] for i in range(len(stages))]
    
    # Plot lines with gradient colors
    for i in range(len(stages) - 1):
        ax6.plot([x_prog[i], x_prog[i+1]], [addsent_prog[i], addsent_prog[i+1]], 
                linewidth=3, color='#e74c3c', alpha=0.7)
        ax6.plot([x_prog[i], x_prog[i+1]], [squad_prog[i], squad_prog[i+1]], 
                linewidth=3, color='#3498db', alpha=0.7)
    
    # Plot markers with model-specific colors
    for i, (x, addsent, squad, color) in enumerate(zip(x_prog, addsent_prog, squad_prog, prog_colors)):
        ax6.scatter(x, addsent, s=200, color=color, marker='o', 
                   edgecolors='black', linewidth=2, zorder=5)
        ax6.scatter(x, squad, s=200, color=color, marker='s', 
                   edgecolors='black', linewidth=2, zorder=5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#e74c3c', label='AddSent (Adversarial)',
               markersize=10, linewidth=3, markerfacecolor='#e74c3c'),
        Line2D([0], [0], marker='s', color='#3498db', label='SQuAD (Clean)',
               markersize=10, linewidth=3, markerfacecolor='#3498db')
    ]
    
    ax6.set_ylabel('Exact Match (%)', fontweight='bold')
    ax6.set_title('Progressive Improvement Path', fontweight='bold', pad=20)
    ax6.set_xticks(x_prog)
    ax6.set_xticklabels(stages, rotation=20, ha='right', fontsize=9)
    ax6.legend(handles=legend_elements, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([45, 95])
    
    # Add value labels
    for i, (addsent, squad) in enumerate(zip(addsent_prog, squad_prog)):
        ax6.text(i, addsent + 1.5, f'{addsent:.1f}%', ha='center', 
                fontsize=9, fontweight='bold', color='#e74c3c')
        ax6.text(i, squad - 2.5, f'{squad:.1f}%', ha='center',
                fontsize=9, fontweight='bold', color='#3498db')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("evaluation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "complete_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive comparison plot to: {output_file}")
    
    plt.show()


def create_paper_plots(data):
    """Create individual publication-quality plots for paper"""
    
    models = data["models"]
    output_dir = Path("evaluation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paper Plot 1: Main Performance Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = [m["name"] for m in models]
    addsent_scores = [m["addsent_em"] for m in models]
    squad_scores = [m["squad_em"] for m in models]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, addsent_scores, width, label='AddSent (Adversarial)',
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, squad_scores, width, label='SQuAD (Clean)',
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Exact Match (%)', fontweight='bold', fontsize=14)
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=11)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "paper_complete_performance.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved paper plot to: {output_dir / 'paper_complete_performance.png'}")
    plt.close()
    
    # Paper Plot 2: Baseline Comparison (Small vs Base)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    baseline_small = models[0]
    baseline_base = models[1]
    
    categories = ['AddSent\n(Adversarial)', 'SQuAD\n(Clean)']
    small_scores = [baseline_small["addsent_em"], baseline_small["squad_em"]]
    base_scores = [baseline_base["addsent_em"], baseline_base["squad_em"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, small_scores, width,
                   label='ELECTRA-small (14M)', color='#e74c3c',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, base_scores, width,
                   label='ELECTRA-base (110M)', color='#f39c12',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Exact Match (%)', fontweight='bold', fontsize=14)
    ax.set_title('Impact of Model Scaling (Baseline Models)', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotations
    for i, cat in enumerate(categories):
        improvement = base_scores[i] - small_scores[i]
        y_pos = max(small_scores[i], base_scores[i]) + 3
        ax.text(i, y_pos, f'+{improvement:.1f}%',
               ha='center', fontsize=11, fontweight='bold',
               color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "paper_baseline_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved baseline comparison to: {output_dir / 'paper_baseline_comparison.png'}")
    plt.close()
    
    # Paper Plot 3: Final Achievement
    fig, ax = plt.subplots(figsize=(10, 7))
    
    final_model = models[4]  # 80-20 Augmented ELECTRA-base
    baseline_small = models[0]
    
    categories = ['AddSent\n(Adversarial)', 'SQuAD\n(Clean)']
    baseline_scores = [baseline_small["addsent_em"], baseline_small["squad_em"]]
    final_scores = [final_model["addsent_em"], final_model["squad_em"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_scores, width,
                   label='Baseline (ELECTRA-small)', color='#e74c3c',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, final_scores, width,
                   label='Final Model (ELECTRA-base + Adversarial)', color='#27ae60',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Exact Match (%)', fontweight='bold', fontsize=14)
    ax.set_title('Final Achievement: Baseline vs Best Model', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotations
    for i, cat in enumerate(categories):
        improvement = final_scores[i] - baseline_scores[i]
        y_pos = max(baseline_scores[i], final_scores[i]) + 3
        ax.text(i, y_pos, f'+{improvement:.1f}%',
               ha='center', fontsize=12, fontweight='bold',
               color='darkgreen', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / "paper_final_achievement.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved final achievement plot to: {output_dir / 'paper_final_achievement.png'}")
    plt.close()

def print_summary(data):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    models = data["models"]
    
    print("\nModel Performance:")
    print("-" * 80)
    for model in models:
        print(f"\n{model['name']} ({model['model_size']} params)")
        print(f"  Training: {model['training_data']}")
        print(f"  AddSent EM: {model['addsent_em']:.2f}%  |  F1: {model['addsent_f1']:.2f}%")
        print(f"  SQuAD EM:   {model['squad_em']:.2f}%  |  F1: {model['squad_f1']:.2f}%")
    
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    
    for key, comp in data["key_comparisons"].items():
        print(f"\n{comp['description']}:")
        for metric, value in comp.items():
            if metric != 'description':
                print(f"  {metric}: {value:+.2f}%")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    for i, insight in enumerate(data["insights"], 1):
        print(f"{i}. {insight}")
    print()

def main():
    print("=" * 80)
    print("Creating Complete Comparison Visualizations")
    print("=" * 80)
    print()
    
    data = load_comparison_data()
    if not data:
        return
    
    print("✅ Loaded comparison data")
    print(f"   Models: {len(data['models'])}")
    print()
    
    print("Creating comprehensive plots...")
    create_comprehensive_plots(data)
    
    print("\nCreating paper-ready plots...")
    create_paper_plots(data)
    
    print_summary(data)
    
    print("=" * 80)
    print("✅ All visualizations created successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. evaluation/plots/complete_comparison.png (comprehensive 6-panel)")
    print("  2. evaluation/plots/paper_complete_performance.png (all models)")
    print("  3. evaluation/plots/paper_baseline_comparison.png (baseline scaling)")
    print("  4. evaluation/plots/paper_final_achievement.png (final vs baseline)")
    print()

if __name__ == "__main__":
    main()
