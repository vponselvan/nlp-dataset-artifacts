#!/usr/bin/env python3
"""
Create visualizations comparing original vs augmented models.
Shows the impact of data augmentation across all 5 ratios.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_augmented_comparison(results_file='./evaluation/augmented_comparison_results.json'):
    """Load augmented comparison results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_original_vs_augmented(results, output_dir='./evaluation/plots'):
    """Plot original vs augmented performance comparison"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ratios = list(results.keys())
    ratio_labels = [r.replace('_', '-') for r in ratios]
    
    # Extract data
    orig_addsent = [results[r]['original']['addsent_em'] for r in ratios]
    aug_addsent = [results[r]['augmented']['addsent_em'] for r in ratios]
    orig_squad = [results[r]['original']['squad_em'] for r in ratios]
    aug_squad = [results[r]['augmented']['squad_em'] for r in ratios]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(ratios))
    width = 0.35
    
    # Plot 1: AddSent (Adversarial) Performance
    bars1 = ax1.bar(x - width/2, orig_addsent, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, aug_addsent, width, label='Augmented', color='#27ae60', alpha=0.8)
    
    ax1.set_xlabel('Training Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax1.set_title('AddSent (Adversarial) Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ratio_labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: SQuAD (Clean) Performance
    bars3 = ax2.bar(x - width/2, orig_squad, width, label='Original', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, aug_squad, width, label='Augmented', color='#9b59b6', alpha=0.8)
    
    ax2.set_xlabel('Training Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax2.set_title('SQuAD (Clean) Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ratio_labels)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = f"{output_dir}/original_vs_augmented_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_improvement_breakdown(results, output_dir='./evaluation/plots'):
    """Plot improvement breakdown for each ratio"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ratios = list(results.keys())
    ratio_labels = [r.replace('_', '-') for r in ratios]
    
    # Extract improvements
    addsent_improvements = []
    squad_improvements = []
    
    for r in ratios:
        if 'improvement' in results[r]:
            addsent_improvements.append(results[r]['improvement']['addsent_em'])
            squad_improvements.append(results[r]['improvement']['squad_em'])
        else:
            addsent_improvements.append(0)
            squad_improvements.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ratios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, addsent_improvements, width, 
                   label='AddSent (Adversarial)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, squad_improvements, width, 
                   label='SQuAD (Clean)', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Training Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Data Augmentation by Ratio', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label_y = height + 0.3 if height > 0 else height - 0.3
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = f"{output_dir}/augmentation_improvements.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_combined_curves(results, output_dir='./evaluation/plots'):
    """Plot original and augmented curves together"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ratios = list(results.keys())
    adversarial_pct = [int(r.split('_')[1]) for r in ratios]
    
    # Extract data
    orig_addsent = [results[r]['original']['addsent_em'] for r in ratios]
    aug_addsent = [results[r]['augmented']['addsent_em'] for r in ratios]
    orig_squad = [results[r]['original']['squad_em'] for r in ratios]
    aug_squad = [results[r]['augmented']['squad_em'] for r in ratios]
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot lines
    plt.plot(adversarial_pct, orig_addsent, 'o-', linewidth=2, markersize=8, 
             label='Original - AddSent', color='#e74c3c', linestyle='--')
    plt.plot(adversarial_pct, aug_addsent, 'o-', linewidth=2, markersize=8, 
             label='Augmented - AddSent', color='#27ae60')
    plt.plot(adversarial_pct, orig_squad, 's-', linewidth=2, markersize=8, 
             label='Original - SQuAD', color='#3498db', linestyle='--')
    plt.plot(adversarial_pct, aug_squad, 's-', linewidth=2, markersize=8, 
             label='Augmented - SQuAD', color='#9b59b6')
    
    plt.xlabel('% Adversarial Training Data', fontsize=12, fontweight='bold')
    plt.ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    plt.title('Original vs Augmented: Performance Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(adversarial_pct)
    
    plt.tight_layout()
    output_file = f"{output_dir}/combined_performance_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("Creating Augmented Comparison Visualizations")
    print("=" * 80)
    print()
    
    # Load results
    try:
        results = load_augmented_comparison()
        print(f"Loaded comparison for {len(results)} ratios")
        print()
    except FileNotFoundError:
        print("❌ Error: augmented_comparison_results.json not found!")
        print("   Please run: python3 scripts/compare_augmented_models.py first")
        return
    
    # Generate plots
    print("Generating plots...")
    plot_original_vs_augmented(results)
    plot_improvement_breakdown(results)
    plot_combined_curves(results)
    
    print()
    print("=" * 80)
    print("✅ All visualizations created!")
    print("=" * 80)
    print()
    print("Plots saved to: ./evaluation/plots/")
    print("  - original_vs_augmented_comparison.png")
    print("  - augmentation_improvements.png")
    print("  - combined_performance_curves.png")
    print()


if __name__ == "__main__":
    main()
