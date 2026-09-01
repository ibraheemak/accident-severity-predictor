#!/usr/bin/env python3
"""Visualize value-of-information curves.

Creates plots showing:
- Accuracy vs average acquisition cost
- Bundle value-of-information rankings
- Cumulative value by bundle acquisition order
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import BUNDLE_NAMES
from bundle_costs import BUNDLE_COSTS, BUNDLE_TIERS_COSTS


def plot_voi_summary(results_df, output_dir):
    """Plot summary of value-of-information."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs steps
    ax = axes[0]
    for strategy in results_df["strategy"].unique():
        subset = results_df[results_df["strategy"] == strategy]
        mean_steps = subset["n_steps"].mean()
        accuracy = subset["is_correct"].mean()
        ax.scatter(mean_steps, accuracy, s=100, alpha=0.7, label=strategy)

    ax.set_xlabel("Mean Acquisition Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Acquisition Cost", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Accuracy vs time
    ax = axes[1]
    for strategy in results_df["strategy"].unique():
        subset = results_df[results_df["strategy"] == strategy]
        # Estimate total cost in seconds
        total_cost = 0
        for idx, row in subset.iterrows():
            for bundle_str in str(row.get("acquired_bundles", "")).split(","):
                if bundle_str.strip() in BUNDLE_NAMES:
                    bundle_idx = BUNDLE_NAMES.index(bundle_str.strip())
                    total_cost += BUNDLE_COSTS.get(bundle_idx, 0)

        mean_cost = total_cost / len(subset) if len(subset) > 0 else 0
        accuracy = subset["is_correct"].mean()
        ax.scatter(mean_cost, accuracy, s=100, alpha=0.7, label=strategy)

    ax.set_xlabel("Mean Acquisition Time (seconds)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Acquisition Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "voi_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_bundle_costs(output_dir):
    """Plot bundle costs by tier."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bundles = list(range(10))
    costs = [BUNDLE_COSTS[i] * 1000 for i in bundles]  # Convert to ms for better scale
    colors = []

    for i in bundles:
        if i in BUNDLE_TIERS_COSTS["ask_caller"]:
            colors.append("#e74c3c")  # Red
        elif i in BUNDLE_TIERS_COSTS["map_lookup"]:
            colors.append("#3498db")  # Blue
        else:
            colors.append("#2ecc71")  # Green

    bars = ax.bar(range(10), costs, color=colors, alpha=0.7, edgecolor="black")

    # Add labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        if cost < 100:
            label = f"{cost:.0f}ms"
        else:
            label = f"{cost/1000:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(10))
    ax.set_xticklabels([BUNDLE_NAMES[i] for i in range(10)], rotation=45, ha="right")
    ax.set_ylabel("Acquisition Cost (ms)", fontsize=12)
    ax.set_title("Bundle Acquisition Costs", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Ask Caller"),
        Patch(facecolor="#3498db", alpha=0.7, label="Map Lookup"),
        Patch(facecolor="#2ecc71", alpha=0.7, label="External API"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    path = output_dir / "bundle_costs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_bundle_value(bundle_value_means, output_dir):
    """Plot value-of-information per bundle."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bundles = sorted(bundle_value_means.items(), key=lambda x: x[1], reverse=True)
    names = [b[0] for b in bundles]
    values = [b[1] for b in bundles]

    # Color by tier
    colors = []
    for name in names:
        idx = BUNDLE_NAMES.index(name)
        if idx in BUNDLE_TIERS_COSTS["ask_caller"]:
            colors.append("#e74c3c")
        elif idx in BUNDLE_TIERS_COSTS["map_lookup"]:
            colors.append("#3498db")
        else:
            colors.append("#2ecc71")

    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.7, edgecolor="black")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg Confidence Improvement", fontsize=12)
    ax.set_title("Value-of-Information per Bundle", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "bundle_value.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path,
                        help="Path to strategy comparison results CSV")
    parser.add_argument("--policy-analysis-json", type=Path,
                        help="Path to policy analysis JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/voi_plots"),
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always plot bundle costs
    print("Plotting bundle costs...")
    plot_bundle_costs(output_dir)

    # Plot strategy comparison if results available
    if args.results_csv and args.results_csv.exists():
        print("Plotting strategy comparison...")
        results_df = pd.read_csv(args.results_csv)
        plot_voi_summary(results_df, output_dir)

    # Plot bundle value if analysis available
    if args.policy_analysis_json and args.policy_analysis_json.exists():
        print("Plotting bundle value-of-information...")
        with open(args.policy_analysis_json) as f:
            analysis = json.load(f)
        plot_bundle_value(analysis["bundle_value_mean"], output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
