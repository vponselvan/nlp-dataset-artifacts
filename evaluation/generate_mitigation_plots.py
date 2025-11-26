#!/usr/bin/env python3
"""
Generate mitigation strategy comparison plots with actual results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# Load data
with open("complete_comparison_results.json", "r") as f:
    data = json.load(f)

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)


def plot_mitigation_comparison():
    """Plot comparing all mitigation strategies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    models_data = {m["name"]: m for m in data["models"]}

    strategies = ["80-20 Original\n(Baseline)", "Negation-Aware", "Entity-Aware"]

    addsent_scores = [
        models_data["80-20 Original (ELECTRA-base)"]["addsent_em"],
        models_data["Negation-Aware (ELECTRA-base)"]["addsent_em"],
        models_data["Entity-Aware (ELECTRA-base)"]["addsent_em"],
    ]

    squad_scores = [
        models_data["80-20 Original (ELECTRA-base)"]["squad_em"],
        models_data["Negation-Aware (ELECTRA-base)"]["squad_em"],
        models_data["Entity-Aware (ELECTRA-base)"]["squad_em"],
    ]

    colors = ["#3498db", "#e67e22", "#8e44ad"]

    x = np.arange(len(strategies))
    width = 0.35

    # AddSent plot
    bars1 = ax1.bar(
        x, addsent_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("Exact Match (%)", fontweight="bold")
    ax1.set_title("AddSent (Adversarial) Performance", fontweight="bold", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_ylim([86, 91])
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add improvement annotations
    for i in range(1, len(strategies)):
        improvement = addsent_scores[i] - addsent_scores[0]
        mid_x = i
        mid_y = (addsent_scores[0] + addsent_scores[i]) / 2
        ax1.annotate(
            f"+{improvement:.2f}%",
            xy=(mid_x, mid_y),
            fontsize=9,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # SQuAD plot
    bars2 = ax2.bar(
        x, squad_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("Exact Match (%)", fontweight="bold")
    ax2.set_title("SQuAD (Clean) Performance", fontweight="bold", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=10)
    ax2.set_ylim([89, 91.5])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add improvement annotations
    for i in range(1, len(strategies)):
        improvement = squad_scores[i] - squad_scores[0]
        mid_x = i
        mid_y = (squad_scores[0] + squad_scores[i]) / 2
        ax2.annotate(
            f"+{improvement:.2f}%",
            xy=(mid_x, mid_y),
            fontsize=9,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "mitigation_strategies_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Created mitigation_strategies_comparison.png")


def plot_performance_progression():
    """Plot showing progression from baseline to best model"""
    fig, ax = plt.subplots(figsize=(12, 7))

    models_data = {m["name"]: m for m in data["models"]}

    models = [
        "Baseline\n(ELECTRA-base)",
        "80-20 Original",
        "Negation-Aware",
        "Entity-Aware",
    ]

    addsent = [
        models_data["Baseline (ELECTRA-base)"]["addsent_em"],
        models_data["80-20 Original (ELECTRA-base)"]["addsent_em"],
        models_data["Negation-Aware (ELECTRA-base)"]["addsent_em"],
        models_data["Entity-Aware (ELECTRA-base)"]["addsent_em"],
    ]

    squad = [
        models_data["Baseline (ELECTRA-base)"]["squad_em"],
        models_data["80-20 Original (ELECTRA-base)"]["squad_em"],
        models_data["Negation-Aware (ELECTRA-base)"]["squad_em"],
        models_data["Entity-Aware (ELECTRA-base)"]["squad_em"],
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        addsent,
        width,
        label="AddSent (Adversarial)",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        squad,
        width,
        label="SQuAD (Clean)",
        color="#3498db",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=13)
    ax.set_title("Model Performance Progression", fontweight="bold", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_ylim([65, 92])
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
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
                fontsize=9,
            )

    # Add total improvement annotation
    total_addsent = addsent[-1] - addsent[0]
    ax.annotate(
        f"Total AddSent\nImprovement:\n+{total_addsent:.2f}%",
        xy=(3, addsent[-1]),
        xytext=(3.5, 80),
        arrowprops=dict(arrowstyle="->", lw=2, color="red"),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "performance_progression.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Created performance_progression.png")


def plot_improvement_breakdown():
    """Plot showing improvement breakdown by strategy"""
    fig, ax = plt.subplots(figsize=(10, 6))

    comparisons = data["key_comparisons"]

    strategies = [
        "Adversarial\nTraining\n(80-20)",
        "Negation-Aware\nMitigation",
        "Entity-Aware\nMitigation",
    ]

    # Calculate improvements from baseline
    baseline_addsent = 68.90
    adv_training_gain = 88.43 - baseline_addsent
    negation_gain = comparisons["negation_aware_mitigation"]["addsent_improvement"]
    entity_gain = comparisons["entity_aware_mitigation"]["addsent_improvement"]

    improvements = [adv_training_gain, negation_gain, entity_gain]
    colors = ["#27ae60", "#e67e22", "#8e44ad"]

    bars = ax.bar(
        strategies,
        improvements,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_ylabel("AddSent EM Improvement (percentage points)", fontweight="bold")
    ax.set_title("Improvement Breakdown by Strategy", fontweight="bold", fontsize=14)
    ax.set_ylim([0, 22])
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"+{height:.2f}pp",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add cumulative annotation
    cumulative = sum(improvements)
    ax.axhline(y=cumulative, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(
        2.5,
        cumulative + 0.5,
        f"Cumulative: +{cumulative:.2f}pp",
        fontsize=11,
        fontweight="bold",
        color="red",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Created improvement_breakdown.png")


def plot_scatter_comparison():
    """Scatter plot comparing all models"""
    fig, ax = plt.subplots(figsize=(11, 8))

    # Plot each model
    for model in data["models"]:
        if "ELECTRA-base" in model["name"]:
            ax.scatter(
                model["squad_em"],
                model["addsent_em"],
                s=200,
                color=model["color"],
                marker=model["marker"],
                edgecolors="black",
                linewidth=2,
                alpha=0.8,
                label=model["name"],
            )

    # Add diagonal reference line (perfect generalization)
    ax.plot(
        [65, 95],
        [65, 95],
        "k--",
        alpha=0.3,
        linewidth=1.5,
        label="Perfect Generalization",
    )

    # Add labels
    ax.set_xlabel("SQuAD EM (Clean Data)", fontweight="bold", fontsize=13)
    ax.set_ylabel("AddSent EM (Adversarial Data)", fontweight="bold", fontsize=13)
    ax.set_title(
        "Clean vs Adversarial Performance Comparison", fontweight="bold", fontsize=15
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([67, 92])
    ax.set_ylim([67, 92])

    # Add performance gap annotations
    models_data = {m["name"]: m for m in data["models"]}
    entity_model = models_data["Entity-Aware (ELECTRA-base)"]
    gap = entity_model["squad_em"] - entity_model["addsent_em"]
    ax.annotate(
        f"Best Model Gap:\n{gap:.2f}pp",
        xy=(entity_model["squad_em"], entity_model["addsent_em"]),
        xytext=(87, 85),
        arrowprops=dict(arrowstyle="->", lw=2, color="purple"),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "scatter_comparison_mitigation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Created scatter_comparison_mitigation.png")


def plot_error_pattern_impact():
    """Plot showing targeted error pattern reductions"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Based on the results
    strategies = ["Negation-Aware\n(40.4% errors)", "Entity-Aware\n(29.8% errors)"]
    improvements = [0.50, 1.46]  # AddSent improvements
    colors = ["#e67e22", "#8e44ad"]

    bars = ax.barh(
        strategies,
        improvements,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("AddSent EM Improvement (percentage points)", fontweight="bold")
    ax.set_title(
        "Impact of Targeted Error Pattern Mitigation", fontweight="bold", fontsize=14
    )
    ax.set_xlim([0, 2])
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"+{width:.2f}pp",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add insight
    ax.text(
        1.0,
        0.3,
        "Entity confusion has higher impact\ndespite lower frequency",
        fontsize=10,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "error_pattern_impact.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Created error_pattern_impact.png")


def generate_results_table():
    """Generate LaTeX table with actual results"""
    models_data = {m["name"]: m for m in data["models"]}

    latex = r"""
\begin{table}[h]
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{SQuAD EM} & \textbf{AddSent EM} & \textbf{Gap} \\
\midrule
Baseline (ELECTRA-base) & 85.46 & 68.90 & -16.56 \\
80-20 Original & 89.97 & 88.43 & -1.54 \\
+ Negation-Aware & 90.07 & 88.93 & -1.14 \\
+ Entity-Aware & \textbf{90.73} & \textbf{89.89} & \textbf{-0.84} \\
\midrule
\textbf{Total Improvement} & \textbf{+5.27} & \textbf{+20.99} & \textbf{+15.72} \\
\bottomrule
\end{tabular}
\caption{Progressive improvement from baseline to Entity-Aware model. The adversarial gap reduced from -16.56pp to -0.84pp, an 84.9\% reduction.}
\label{tab:mitigation_results}
\end{table}
"""

    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex)

    print("✓ Created results_table.tex")


if __name__ == "__main__":
    print("Generating mitigation strategy plots...")
    print("=" * 60)

    plot_mitigation_comparison()
    plot_performance_progression()
    plot_improvement_breakdown()
    plot_scatter_comparison()
    plot_error_pattern_impact()
    generate_results_table()

    print("=" * 60)
    print("✓ All plots generated successfully!")
    print(f"✓ Saved to: {output_dir.absolute()}")
