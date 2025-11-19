#!/usr/bin/env python3
"""
Create visualizations from comparison results.
Generates publication-quality plots showing trade-offs.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_comparison_results(results_file='./evaluation/comparison_results.json'):
    """Load comparison results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_trade_off_curve(results, output_dir='./evaluation/plots'):
    """Plot adversarial robustness vs clean performance trade-off"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    adversarial_pct = [exp['adversarial_pct'] for exp in results['experiments']]
    addsent_em = [exp['addsent_em'] for exp in results['experiments']]
    squad_em = [exp['squad_em'] for exp in results['experiments']]
    
    baseline_addsent = results['baseline']['addsent_em']
    baseline_squad = results['baseline']['squad_em']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot([0] + adversarial_pct, [baseline_addsent] + addsent_em, 
             'o-', linewidth=2, markersize=8, label='AddSent (Adversarial)', color='#e74c3c')
    plt.plot([0] + adversarial_pct, [baseline_squad] + squad_em, 
             's-', linewidth=2, markersize=8, label='SQuAD (Clean)', color='#3498db')
    
    # Formatting
    plt.xlabel('% Adversarial Training Data', fontsize=12)
    plt.ylabel('Exact Match (%)', fontsize=12)
    plt.title('Adversarial Fine-Tuning Trade-off Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks([0, 10, 20, 30, 40, 50])
    
    # Save
    output_file = f"{output_dir}/trade_off_curve.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_trade_off_ratio(results, output_dir='./evaluation/plots'):
    """Plot trade-off ratio (gain/cost) for each experiment"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    names = [exp['name'] for exp in results['experiments']]
    ratios = [exp['trade_off_ratio'] for exp in results['experiments']]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = plt.bar(names, ratios, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    plt.xlabel('Training Data Ratio (SQuAD-AddSent)', fontsize=12)
    plt.ylabel('Trade-off Ratio (Gain/Cost)', fontsize=12)
    plt.title('Trade-off Efficiency by Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save
    output_file = f"{output_dir}/trade_off_ratio.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_performance_comparison(results, output_dir='./evaluation/plots'):
    """Plot side-by-side comparison of all models"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    names = ['Baseline'] + [exp['name'] for exp in results['experiments']]
    addsent_scores = [results['baseline']['addsent_em']] + \
                     [exp['addsent_em'] for exp in results['experiments']]
    squad_scores = [results['baseline']['squad_em']] + \
                   [exp['squad_em'] for exp in results['experiments']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, addsent_scores, width, label='AddSent (Adversarial)', 
                   color='#e74c3c', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, squad_scores, width, label='SQuAD (Clean)', 
                   color='#3498db', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Exact Match (%)', fontsize=12)
    ax.set_title('Performance Comparison Across All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Save
    output_file = f"{output_dir}/performance_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_gains_and_costs(results, output_dir='./evaluation/plots'):
    """Plot gains and costs separately"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    names = [exp['name'] for exp in results['experiments']]
    gains = [exp['addsent_gain'] for exp in results['experiments']]
    costs = [exp['squad_cost'] for exp in results['experiments']]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot gains
    bars1 = ax1.bar(names, gains, color='#27ae60', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Training Ratio', fontsize=11)
    ax1.set_ylabel('Robustness Gain (%)', fontsize=11)
    ax1.set_title('Adversarial Robustness Gain', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, gain in zip(bars1, gains):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'+{gain:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot costs
    bars2 = ax2.bar(names, costs, color='#e67e22', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Training Ratio', fontsize=11)
    ax2.set_ylabel('Clean Performance Cost (%)', fontsize=11)
    ax2.set_title('Clean Data Performance Cost', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'-{cost:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Save
    output_file = f"{output_dir}/gains_and_costs.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def create_all_plots(results_file='./evaluation/comparison_results.json'):
    """Create all visualization plots"""
    print("=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    print()
    
    # Load results
    print(f"Loading results from: {results_file}")
    results = load_comparison_results(results_file)
    print(f"Found {len(results['experiments'])} experiments")
    print()
    
    # Create plots
    print("Generating plots...")
    plot_trade_off_curve(results)
    plot_trade_off_ratio(results)
    plot_performance_comparison(results)
    plot_gains_and_costs(results)
    
    print()
    print("=" * 60)
    print("✅ All visualizations created!")
    print("=" * 60)
    print()
    print("Plots saved to: ./evaluation/plots/")
    print("  - trade_off_curve.png")
    print("  - trade_off_ratio.png")
    print("  - performance_comparison.png")
    print("  - gains_and_costs.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations from comparison results")
    parser.add_argument('--results_file', type=str, 
                       default='./evaluation/comparison_results.json',
                       help='Path to comparison results JSON file')
    
    args = parser.parse_args()
    
    try:
        create_all_plots(args.results_file)
    except FileNotFoundError:
        print("❌ Error: comparison_results.json not found!")
        print("   Please run: python3 scripts/compare_adversarial_models.py first")
    except ImportError:
        print("❌ Error: matplotlib not installed!")
        print("   Install with: pip install matplotlib")
