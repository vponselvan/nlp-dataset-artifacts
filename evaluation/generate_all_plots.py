#!/usr/bin/env python3
"""
Generate all plots for the project report with consistent, clear, and rich visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for consistent, professional plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# Load data
with open("comparison_results.json", "r") as f:
    comparison_data = json.load(f)

with open("complete_comparison_results.json", "r") as f:
    complete_data = json.load(f)

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Color scheme
COLORS = {
    "baseline_small": "#e74c3c",
    "baseline_base": "#f39c12",
    "adv_small": "#3498db",
    "adv_base": "#27ae60",
    "aug_small": "#9b59b6",
    "aug_base": "#16a085",
    "addsent": "#e74c3c",
    "squad": "#3498db",
}


def plot_1_baseline_comparison():
    """Plot 1: ELECTRA-small baseline performance"""
    baseline = comparison_data["baseline"]

    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = ["SQuAD Dev", "AddSent Adv"]
    em_scores = [baseline["squad_em"], baseline["addsent_em"]]
    f1_scores = [baseline["squad_f1"], baseline["addsent_f1"]]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        em_scores,
        width,
        label="EM",
        color=COLORS["addsent"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x + width / 2,
        f1_scores,
        width,
        label="F1",
        color=COLORS["squad"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    # Add performance drop annotation
    drop = baseline["squad_em"] - baseline["addsent_em"]
    ax.annotate(
        f"Drop: {drop:.1f}%",
        xy=(0.5, (baseline["squad_em"] + baseline["addsent_em"]) / 2),
        xytext=(1.5, 70),
        arrowprops=dict(arrowstyle="->", lw=2, color="red"),
        fontsize=12,
        fontweight="bold",
        color="red",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

    ax.set_ylabel("Performance (%)", fontweight="bold")
    ax.set_title(
        "Baseline Performance: ELECTRA-small on Clean vs Adversarial Data",
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim([0, 95])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated: baseline_performance.png")


def plot_2_adversarial_training_results():
    """Plot 2: Performance across different mixing ratios (ELECTRA-small)"""
    experiments = comparison_data["experiments"]
    baseline = comparison_data["baseline"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ratios = [f"{exp['name']}" for exp in experiments]
    addsent_scores = [exp["addsent_em"] for exp in experiments]
    squad_scores = [exp["squad_em"] for exp in experiments]

    # Left plot: AddSent performance
    line1 = ax1.plot(
        ratios,
        addsent_scores,
        marker="o",
        linewidth=3,
        markersize=10,
        color=COLORS["addsent"],
        label="AddSent EM",
    )
    ax1.axhline(
        y=baseline["addsent_em"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Baseline ({baseline["addsent_em"]:.1f}%)',
        alpha=0.7,
    )

    # Highlight best performer
    best_idx = addsent_scores.index(max(addsent_scores))
    ax1.scatter(
        best_idx,
        addsent_scores[best_idx],
        s=300,
        color="gold",
        edgecolor="black",
        linewidth=3,
        zorder=5,
        marker="*",
    )
    ax1.text(
        best_idx,
        addsent_scores[best_idx] + 2,
        "BEST",
        ha="center",
        fontweight="bold",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.8),
    )

    ax1.set_xlabel("Training Ratio (SQuAD-AddSent)", fontweight="bold")
    ax1.set_ylabel("Exact Match (%)", fontweight="bold")
    ax1.set_title("Adversarial Performance (AddSent)", fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_ylim([40, 75])

    # Right plot: SQuAD performance
    line2 = ax2.plot(
        ratios,
        squad_scores,
        marker="s",
        linewidth=3,
        markersize=10,
        color=COLORS["squad"],
        label="SQuAD EM",
    )
    ax2.axhline(
        y=baseline["squad_em"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Baseline ({baseline["squad_em"]:.1f}%)',
        alpha=0.7,
    )

    ax2.set_xlabel("Training Ratio (SQuAD-AddSent)", fontweight="bold")
    ax2.set_ylabel("Exact Match (%)", fontweight="bold")
    ax2.set_title("Clean Performance (SQuAD)", fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim([40, 85])

    plt.suptitle(
        "ELECTRA-small: Impact of Adversarial Training Ratios",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "adversarial_training_ratios.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Generated: adversarial_training_ratios.png")


def plot_3_tradeoff_analysis():
    """Plot 3: Trade-off analysis"""
    experiments = comparison_data["experiments"]

    fig, ax = plt.subplots(figsize=(10, 7))

    addsent_gains = [exp["addsent_gain"] for exp in experiments]
    squad_costs = [exp["squad_cost"] for exp in experiments]
    ratios = [exp["name"] for exp in experiments]

    # Create scatter plot with different sizes based on trade-off ratio
    for i, (gain, cost, ratio) in enumerate(zip(addsent_gains, squad_costs, ratios)):
        size = 300 + i * 100
        color = COLORS["addsent"] if gain > 0 else "gray"
        ax.scatter(
            cost, gain, s=size, alpha=0.6, color=color, edgecolor="black", linewidth=2
        )
        ax.annotate(
            ratio,
            (cost, gain),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Add quadrant lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)

    # Add arrows and labels for quadrants
    ax.text(
        35,
        13,
        "Trade-off\nRegion",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color="red",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    ax.text(
        10,
        -7,
        "Both\nDecline",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("SQuAD Performance Cost (%)", fontweight="bold", fontsize=13)
    ax.set_ylabel("AddSent Performance Gain (%)", fontweight="bold", fontsize=13)
    ax.set_title(
        "Robustness-Accuracy Trade-off Analysis", fontweight="bold", fontsize=15, pad=20
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([10, 38])
    ax.set_ylim([-10, 15])

    plt.tight_layout()
    plt.savefig(output_dir / "tradeoff_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated: tradeoff_analysis.png")


def plot_4_model_progression():
    """Plot 4: Complete model progression"""
    models = complete_data["models"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Prepare data
    model_names = [m["name"] for m in models]
    model_names_short = [
        "Baseline\nSmall",
        "Baseline\nBase",
        "80-20\nSmall",
        "80-20\nBase",
        "80-20Aug\nSmall",
        "80-20Aug\nBase",
    ]
    addsent_scores = [m["addsent_em"] for m in models]
    squad_scores = [m["squad_em"] for m in models]
    colors = [m["color"] for m in models]

    x = np.arange(len(model_names))

    # Left: AddSent scores
    bars1 = ax1.bar(
        x, addsent_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    for i, (bar, score) in enumerate(zip(bars1, addsent_scores)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Highlight best
    best_idx = addsent_scores.index(max(addsent_scores))
    bars1[best_idx].set_edgecolor("gold")
    bars1[best_idx].set_linewidth(4)
    ax1.text(
        best_idx,
        addsent_scores[best_idx] + 4,
        "★ BEST",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="gold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.8),
    )

    ax1.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=12)
    ax1.set_title("Adversarial Robustness (AddSent)", fontweight="bold", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names_short, fontsize=9, fontweight="bold")
    ax1.set_ylim([0, 100])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: SQuAD scores
    bars2 = ax2.bar(
        x, squad_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    for i, (bar, score) in enumerate(zip(bars2, squad_scores)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax2.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=12)
    ax2.set_title("Clean Performance (SQuAD)", fontweight="bold", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names_short, fontsize=9, fontweight="bold")
    ax2.set_ylim([0, 100])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "Model Progression: From ELECTRA-small Baseline to ELECTRA-base with Adversarial Training",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "model_progression.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated: model_progression.png")


def plot_5_improvement_breakdown():
    """Plot 5: Improvement breakdown showing each step"""
    fig, ax = plt.subplots(figsize=(12, 7))

    steps = [
        "ELECTRA-small\nBaseline",
        "Add Adversarial\nTraining (80-20)",
        "Switch to\nELECTRA-base",
        "Apply to\nELECTRA-base",
    ]

    addsent_values = [53.99, 66.57, 68.90, 88.43]
    squad_values = [78.16, 62.85, 85.46, 89.97]

    x = np.arange(len(steps))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        addsent_values,
        width,
        label="AddSent EM",
        color=COLORS["addsent"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        squad_values,
        width,
        label="SQuAD EM",
        color=COLORS["squad"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    # Add improvement arrows
    improvements = [
        (0, 1, +12.58, -15.31),  # Baseline to 80-20
        (1, 2, +2.33, +22.61),  # Small to Base baseline
        (2, 3, +19.53, +4.51),  # Base baseline to 80-20 Base
    ]

    for start, end, addsent_imp, squad_imp in improvements:
        # AddSent improvement
        ax.annotate(
            "",
            xy=(end - width / 2, addsent_values[end] - 2),
            xytext=(start - width / 2, addsent_values[start] + 2),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"),
        )
        mid_x = (start + end) / 2 - width / 2
        mid_y = (addsent_values[start] + addsent_values[end]) / 2
        ax.text(
            mid_x,
            mid_y,
            f"{addsent_imp:+.1f}%",
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=13)
    ax.set_title(
        "Step-by-Step Improvement: From Baseline to Final Model",
        fontweight="bold",
        fontsize=15,
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated: improvement_breakdown.png")


def plot_6_final_comparison():
    """Plot 6: Final model vs baselines scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    models = complete_data["models"]

    for model in models:
        size = 300 if "Baseline" in model["name"] else 500
        marker = model.get("marker", "o")

        ax.scatter(
            model["squad_em"],
            model["addsent_em"],
            s=size,
            alpha=0.7,
            color=model["color"],
            marker=marker,
            edgecolor="black",
            linewidth=2,
            label=model["name"],
        )

    # Add diagonal line for reference (where AddSent = SQuAD)
    ax.plot([40, 95], [40, 95], "k--", alpha=0.3, linewidth=2, label="Perfect Parity")

    # Annotate best model
    best_model = models[3]  # 80-20 Base
    ax.annotate(
        "BEST MODEL\n88.43% / 89.97%",
        xy=(best_model["squad_em"], best_model["addsent_em"]),
        xytext=(best_model["squad_em"] - 10, best_model["addsent_em"] - 10),
        arrowprops=dict(arrowstyle="->", lw=3, color="gold"),
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="gold",
            alpha=0.9,
            edgecolor="black",
            linewidth=2,
        ),
    )

    ax.set_xlabel("SQuAD Performance (EM %)", fontweight="bold", fontsize=13)
    ax.set_ylabel("AddSent Performance (EM %)", fontweight="bold", fontsize=13)
    ax.set_title(
        "Adversarial Robustness vs Clean Performance: All Models",
        fontweight="bold",
        fontsize=15,
        pad=20,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.95,
        fontsize=9,
        ncol=2,
        markerscale=0.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([45, 95])
    ax.set_ylim([45, 95])

    plt.tight_layout()
    plt.savefig(
        output_dir / "final_comparison_scatter.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Generated: final_comparison_scatter.png")


def plot_7_capacity_bottleneck():
    """Plot 7: Demonstrating capacity bottleneck"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data for ELECTRA-small
    small_configs = ["Baseline", "80-20 Original", "80-20 Augmented"]
    small_addsent = [53.99, 66.57, 63.48]
    small_squad = [78.16, 62.85, 66.60]

    # Data for ELECTRA-base
    base_configs = ["Baseline", "80-20 Original", "80-20 Augmented"]
    base_addsent = [68.90, 88.43, 86.12]
    base_squad = [85.46, 89.97, 87.92]

    x = np.arange(len(small_configs))
    width = 0.35

    # Plot ELECTRA-small
    ax.plot(
        x,
        small_addsent,
        "o-",
        linewidth=3,
        markersize=10,
        label="ELECTRA-small AddSent",
        color="#e74c3c",
        alpha=0.7,
    )
    ax.plot(
        x,
        small_squad,
        "s--",
        linewidth=3,
        markersize=10,
        label="ELECTRA-small SQuAD",
        color="#3498db",
        alpha=0.7,
    )

    # Plot ELECTRA-base
    ax.plot(
        x,
        base_addsent,
        "o-",
        linewidth=3,
        markersize=12,
        label="ELECTRA-base AddSent",
        color="#c0392b",
    )
    ax.plot(
        x,
        base_squad,
        "s--",
        linewidth=3,
        markersize=12,
        label="ELECTRA-base SQuAD",
        color="#2980b9",
    )

    # Highlight the capacity bottleneck
    ax.annotate(
        "Capacity\nBottleneck!",
        xy=(2, small_addsent[2]),
        xytext=(2.3, 55),
        arrowprops=dict(arrowstyle="->", lw=3, color="red"),
        fontsize=13,
        fontweight="bold",
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="yellow",
            alpha=0.9,
            edgecolor="red",
            linewidth=2,
        ),
    )

    ax.annotate(
        "Eliminated by\nModel Scaling!",
        xy=(2, base_addsent[2]),
        xytext=(1.5, 95),
        arrowprops=dict(arrowstyle="->", lw=3, color="green"),
        fontsize=13,
        fontweight="bold",
        color="green",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="lightgreen",
            alpha=0.9,
            edgecolor="green",
            linewidth=2,
        ),
    )

    ax.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=13)
    ax.set_title(
        "Capacity Bottleneck: ELECTRA-small vs ELECTRA-base",
        fontweight="bold",
        fontsize=15,
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(small_configs, fontsize=11, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        framealpha=0.95,
        fontsize=10,
        ncol=2,
    )
    ax.set_ylim([50, 100])
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "capacity_bottleneck.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated: capacity_bottleneck.png")


def main():
    print("\n" + "=" * 60)
    print("Generating All Plots for Project Report")
    print("=" * 60 + "\n")

    plot_1_baseline_comparison()
    plot_2_adversarial_training_results()
    plot_3_tradeoff_analysis()
    plot_4_model_progression()
    plot_5_improvement_breakdown()
    plot_6_final_comparison()
    plot_7_capacity_bottleneck()

    print("\n" + "=" * 60)
    print("✓ All plots generated successfully!")
    print(f"✓ Saved to: {output_dir.absolute()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
