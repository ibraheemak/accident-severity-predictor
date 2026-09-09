#!/usr/bin/env python3
"""Figures for the lambda sweep: what the agent buys, and what it gives up.

Combines the two sweeps -- the linear grid (lambda = 0 .. 0.10, where the
expensive ask-the-caller bundles fall away) and the log-range one
(lambda = 0.2 .. 10, out to where even a 10 ms lookup stops being worth its
time) -- into one continuous picture.

Three figures:

  lambda_operating_curve.png   accuracy against acquisition time. THE figure:
                               it shows the menu a dispatch centre chooses from,
                               rather than one arbitrary operating point.
  lambda_bundle_heatmap.png    how often each bundle is acquired, per lambda.
                               Read a row left-to-right to see the lambda at
                               which the agent gives up on that bundle.
  lambda_vs_metrics.png        accuracy and acquisition time against lambda,
                               as two panels. Deliberately NOT a dual-axis
                               chart -- two scales on one y-axis invite false
                               readings of where the curves "cross".

Usage (from the repo root, after both sweeps):
    python CODE/training/agent/analysis/plot_lambda_curve.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import BUNDLE_NAMES, BUNDLE_TIERS

# Project palette (see voi/plot_voi.py). Blue is the sequential hue; the
# 100->700 steps are the validated ramp for continuous magnitude.
BLUE = "#2a78d6"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
TIER_COLOR = {"ask_caller": "#2a78d6", "map_lookup": "#eb6834",
              "external_api": "#1baf7a"}

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


def load(dirs, fname):
    frames = [pd.read_csv(d / fname) for d in dirs if (d / fname).exists()]
    if not frames:
        raise SystemExit(f"no {fname} found in {[str(d) for d in dirs]}")
    return (pd.concat(frames).drop_duplicates("lam").sort_values("lam")
            .reset_index(drop=True))


# ------------------------------------------------------- 1. operating curve
def plot_operating_curve(c, out):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(c.seconds, c.accuracy, color=BLUE, linewidth=2, zorder=4,
            marker="o", markersize=6, markerfacecolor=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=2)

    # Label selectively -- a number on every point is unreadable.
    for lam in (c.lam.min(), 0.01, 0.02, 0.05, 0.1, 1.0, c.lam.max()):
        row = c.iloc[(c.lam - lam).abs().argmin()]
        ax.annotate(f"λ={row.lam:g}", (row.seconds, row.accuracy),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=9, color=INK_2)

    ax.set_xscale("log")
    ax.set_xlabel("Mean acquisition time per call (seconds, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy against acquisition time",
                 color=INK, fontweight="bold", loc="left", pad=12)
    ax.margins(y=0.16)
    chrome(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", out)


# ------------------------------------------------------------- 2. heatmap
def plot_bundle_heatmap(f, out):
    order = [b for b in BUNDLE_NAMES]
    m = f.set_index("lam")[order].T                    # bundles x lambda
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    im = ax.imshow(m.values, aspect="auto", cmap=CMAP, vmin=0, vmax=1)

    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([f"{v:g}" for v in m.columns], fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    # Colour the bundle names by acquisition kind -- the expensive ones are the
    # ask-the-caller bundles, and they are the ones that vanish first.
    for t, b in zip(ax.get_yticklabels(), order):
        t.set_color(TIER_COLOR[BUNDLE_TIERS[b]])

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(len(m.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_xlabel("λ  (accuracy points per second of acquisition time)")
    ax.set_title("Bundle acquisition frequency by λ",
                 color=INK, fontweight="bold", loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("fraction of calls where acquired", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, color=MUTED, labelcolor=MUTED)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="none", markersize=8,
                              color=TIER_COLOR[t],
                              label=t.replace("_", " ").title())
                       for t in ("ask_caller", "map_lookup", "external_api")],
              frameon=False, fontsize=8.5, labelcolor=INK_2,
              loc="lower left", bbox_to_anchor=(0, -0.30), ncol=3)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", out)


# --------------------------------------------------- 3. metrics vs lambda
def plot_vs_lambda(c, out):
    # Two panels rather than two y-axes: a dual-axis chart makes the crossing
    # point look meaningful when it is an artefact of the chosen scales.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    x = c.lam.replace(0, c.lam[c.lam > 0].min() / 3)   # 0 cannot sit on a log axis

    ax = axes[0]
    ax.plot(x, c.accuracy, color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by λ", color=INK,
                                            fontweight="bold", loc="left", pad=10)

    ax = axes[1]
    ax.plot(x, c.seconds, color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.set_yscale("log")
    ax.set_ylabel("Acquisition time per call (s)")
    ax.set_title("Acquisition time by λ", color=INK, fontweight="bold",
                 loc="left", pad=10)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("λ  (leftmost point is λ = 0)")
        chrome(ax)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+",
                    default=[Path("docs/figures/lambda_curve"),
                             Path("docs/figures/lambda_curve_log")])
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/lambda_curve"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve = load(args.dirs, "lambda_curve.csv")
    freq = load(args.dirs, "bundle_frequency.csv")
    print(f"combined {len(curve)} lambda values: "
          f"{curve.lam.min():g} to {curve.lam.max():g}\n")
    curve.to_csv(args.out_dir / "lambda_curve_combined.csv", index=False)

    plot_operating_curve(curve, args.out_dir / "lambda_operating_curve.png")
    plot_bundle_heatmap(freq, args.out_dir / "lambda_bundle_heatmap.png")
    plot_vs_lambda(curve, args.out_dir / "lambda_vs_metrics.png")

    print("\nsummary:")
    print(curve[["lam", "accuracy", "accuracy_sd", "seconds", "bundles",
                 "distinct_sets"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
