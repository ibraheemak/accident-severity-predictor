#!/usr/bin/env python3
"""Plot the acquisition-strategy comparison produced by compare_strategies.py.

Creates:
- strategy_summary.png  accuracy against acquisition cost, one point per
                        strategy, in both steps and seconds
- bundle_costs.png      what each bundle costs to acquire, by kind

For per-bundle VALUE figures see voi/plot_voi.py -- that answers what each
bundle is worth; this one compares whole strategies. (This file was named
visualize_voi.py until September 2026, and also carried a bundle-value chart
that duplicated voi/ with a weaker metric; that chart was removed.)
"""

import argparse
from ast import literal_eval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import BUNDLE_NAMES
from bundle_costs import BUNDLE_COSTS, BUNDLE_TIERS_COSTS

# Validated categorical slots 1-3, matching voi/plot_voi.py.
TIER_COLOR = {"ask_caller": "#2a78d6", "map_lookup": "#eb6834",
              "external_api": "#1baf7a"}
TIER_LABEL = {"ask_caller": "Ask the caller", "map_lookup": "Map lookup",
              "external_api": "External API"}
STRATEGY_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 12, "axes.labelsize": 10,
})


def chrome(ax, axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def acquisition_seconds(cell) -> float:
    """Total acquisition time for one episode's `acquired_bundles` cell.

    The column round-trips through CSV as a list repr ("['aadf', 'osm']"), so it
    must be parsed, not split on commas -- splitting leaves brackets and quotes
    attached, no name ever matches BUNDLE_NAMES, and every strategy silently
    costs 0 seconds. That was the behaviour before September 2026.
    """
    if isinstance(cell, str):
        try:
            cell = literal_eval(cell)
        except (ValueError, SyntaxError):
            return 0.0
    if not isinstance(cell, (list, tuple)):
        return 0.0
    return sum(BUNDLE_COSTS[BUNDLE_NAMES.index(b)] for b in cell
               if b in BUNDLE_NAMES)


def plot_strategy_summary(df: pd.DataFrame, out_dir: Path):
    strategies = list(df["strategy"].unique())
    df = df.assign(seconds=df["acquired_bundles"].map(acquisition_seconds))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, xcol, xlabel in (
        (axes[0], "n_steps", "Mean bundles acquired"),
        (axes[1], "seconds", "Mean acquisition time (seconds)"),
    ):
        for i, s in enumerate(strategies):
            sub = df[df["strategy"] == s]
            x, y = sub[xcol].mean(), sub["is_correct"].mean()
            ax.scatter(x, y, s=130, zorder=4, color=STRATEGY_COLOR[i % 4],
                       edgecolor=SURFACE, linewidth=2,
                       label=s.replace("_", " ") if ax is axes[0] else None)
            ax.annotate(s.replace("_", " "), (x, y), textcoords="offset points",
                        xytext=(0, 13), ha="center", fontsize=9, color=INK_2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        chrome(ax)
        ax.margins(x=0.22, y=0.28)

    axes[0].set_title("Accuracy against acquisition effort", color=INK,
                      fontweight="bold", loc="left", pad=12)
    axes[1].set_title("Accuracy against acquisition time", color=INK,
                      fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    path = out_dir / "strategy_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_bundle_costs(out_dir: Path):
    tier_of = {i: t for t, idxs in BUNDLE_TIERS_COSTS.items() for i in idxs}
    order = sorted(range(10), key=lambda i: BUNDLE_COSTS[i])
    costs = [BUNDLE_COSTS[i] * 1000 for i in order]
    colors = [TIER_COLOR[tier_of[i]] for i in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(range(10), costs, width=0.55, color=colors, zorder=3)
    for x, c in enumerate(costs):
        ax.text(x, c * 1.12, f"{c:.0f} ms" if c < 1000 else f"{c/1000:.1f} s",
                ha="center", va="bottom", fontsize=9, color=INK_2)

    ax.set_yscale("log")
    ax.set_xticks(range(10))
    ax.set_xticklabels([BUNDLE_NAMES[i] for i in order], rotation=35, ha="right")
    ax.set_ylabel("Acquisition cost (ms, log scale)")
    ax.set_title("Acquisition cost per bundle", color=INK, fontweight="bold",
                 loc="left", pad=12)
    ax.set_ylim(5, 12000)
    chrome(ax)
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                          color=TIER_COLOR[t], label=TIER_LABEL[t])
               for t in ("ask_caller", "map_lookup", "external_api")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
              loc="upper left")
    fig.tight_layout()
    path = out_dir / "bundle_costs.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-csv", type=Path,
                    default=Path("docs/figures/strategy_comparison/strategy_comparison.csv"),
                    help="output of compare_strategies.py")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("docs/figures/strategy_comparison"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_bundle_costs(args.output_dir)

    if args.results_csv and args.results_csv.exists():
        plot_strategy_summary(pd.read_csv(args.results_csv), args.output_dir)
    else:
        print(f"(no {args.results_csv} — run compare_strategies.py for the "
              f"strategy figure)")


if __name__ == "__main__":
    main()
