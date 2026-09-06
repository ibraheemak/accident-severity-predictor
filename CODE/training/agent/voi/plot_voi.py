#!/usr/bin/env python3
"""Plot the value-of-information sweep.

Reads the tables written by compute_voi.py and renders three figures:

  voi_marginal.png      what each bundle adds, on average, once you already
                        hold some other arbitrary combination
  voi_saturation.png    best achievable score against number of bundles held --
                        where the action space stops paying
  voi_cost_benefit.png  that value against what the bundle costs to acquire

Usage (from the repo root, after compute_voi.py):
    python CODE/training/agent/voi/plot_voi.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN_DIR = Path("docs/figures/voi_sweep")

# Validated categorical slots 1-3 (all-pairs safe in light mode: worst CVD
# dE 9.2, worst normal-vision dE 24.0). Aqua sits below 3:1 on this surface,
# so every mark carries a direct label -- the documented relief.
TIER_COLOR = {
    "ask_caller":   "#2a78d6",   # slot 1 blue
    "map_lookup":   "#eb6834",   # slot 2 orange
    "external_api": "#1baf7a",   # slot 3 aqua
}
TIER_LABEL = {"ask_caller": "Ask the caller",
              "map_lookup": "Map lookup",
              "external_api": "External API"}

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

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


def chrome(ax, xgrid=False, ygrid=False):
    """Hairline solid grid, no top/right spines, recessive everything."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid,
            color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def legend(ax, tiers, **kw):
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                          color=TIER_COLOR[t], label=TIER_LABEL[t]) for t in tiers]
    lg = ax.legend(handles=handles, frameon=False, fontsize=9,
                   labelcolor=INK_2, **kw)
    return lg


# --------------------------------------------------------------- 1. marginal
def plot_marginal(bv: pd.DataFrame, out: Path):
    d = bv.sort_values("mean_marginal")                  # ascending: best on top
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = [TIER_COLOR[t] for t in d.tier]
    ax.barh(d.bundle, d.mean_marginal, height=0.40, color=colors, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=2)

    span = d.mean_marginal.max() - min(d.mean_marginal.min(), 0)
    for y, v in enumerate(d.mean_marginal):
        off = span * 0.012
        ax.text(v + (off if v >= 0 else -off), y, f"{v:+.4f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK_2)

    ax.set_xlabel("Mean marginal gain in test macro F1\n"
                  "(averaged over all 512 subsets that lack the bundle)")
    ax.set_title("Marginal value of each acquisition bundle", color=INK,
                 fontweight="bold", loc="left", pad=12)
    # Left margin must clear the widest negative value label, not just the bar.
    ax.set_xlim(-span * 0.15, span * 1.20)
    chrome(ax, xgrid=True)
    legend(ax, ["ask_caller", "map_lookup", "external_api"], loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------- 2. saturation
def plot_saturation(sat: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    k = sat.n_bundles
    ax.fill_between(k, sat.worst, sat.best, color="#2a78d6", alpha=0.10,
                    linewidth=0, zorder=2)
    ax.plot(k, sat.best, color="#2a78d6", linewidth=2, zorder=4,
            marker="o", markersize=5, markerfacecolor="#2a78d6",
            markeredgecolor=SURFACE, markeredgewidth=2)

    floor, ceiling = sat.best.iloc[0], sat.best.max()
    # Direct-label only the points that carry the story: floor, the knee, the end.
    knee = int(sat.loc[sat.gain_over_prev.gt(0.005)].n_bundles.max())
    for kk, dx, ha in ((0, 10, "left"), (knee, 0, "center"), (int(k.max()), 0, "center")):
        row = sat.loc[sat.n_bundles == kk].iloc[0]
        ax.annotate(f"{row.best:.4f}", (kk, row.best),
                    textcoords="offset points", xytext=(dx, 10),
                    ha=ha, fontsize=9, color=INK_2)
    # Label the ceiling AT the subset size that achieves it, not floating at the
    # left edge where it describes nothing.
    k_ceiling = int(sat.loc[sat.best.idxmax(), "n_bundles"])
    ax.axhline(ceiling, color=BASELINE, linewidth=0.8, zorder=1)
    ax.annotate(f"ceiling {ceiling:.4f}", (k_ceiling, ceiling),
                textcoords="offset points", xytext=(0, -17),
                ha="center", fontsize=9, color=MUTED)
    ax.annotate(f"past {knee} bundles every further\nacquisition is worth ≤ +0.002",
                (knee + 0.2, floor + (ceiling - floor) * 0.42),
                fontsize=9, color=INK_2)

    # A band needs naming; a legend does that without floating text colliding
    # with the axis, which is what the annotation version did.
    handles = [
        plt.Line2D([], [], color="#2a78d6", linewidth=2, marker="o",
                   markersize=5, markeredgecolor=SURFACE, markeredgewidth=2,
                   label="Best subset of that size"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#2a78d6", alpha=0.10,
                      edgecolor="none", label="Worst-to-best range"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
              loc="lower right")

    ax.set_xlabel("Number of bundles acquired")
    ax.set_ylabel("Best achievable test macro F1")
    ax.set_title("Best achievable performance by acquisition-set size", color=INK,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xticks(range(0, int(k.max()) + 1))
    chrome(ax, ygrid=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------- 3. cost/benefit
def plot_cost_benefit(bv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for tier in ("ask_caller", "map_lookup", "external_api"):
        d = bv[bv.tier == tier]
        ax.scatter(d.seconds, d.mean_marginal, s=110, zorder=4,
                   color=TIER_COLOR[tier], edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=2)

    # Four map-lookup bundles sit within 20ms of each other, so label placement
    # has to be per-point; a single up/down rule collides.
    OFFSETS = {                          # bundle: (dx, dy, ha)
        "osm":                   (0,  12, "center"),
        "aadf":                  (0,  12, "center"),
        "meteo":                 (0, -18, "center"),
        "junction":              (-12, -3, "right"),
        "road_class":            (13, -3, "left"),
        "road_environment":      (0,  12, "center"),
        "crash_history":         (13, -3, "left"),
        "vehicles":              (0,  13, "center"),
        "casualties":            (0,  13, "center"),
        "casualty_demographics": (-13, -13, "right"),
    }
    for _, r in bv.iterrows():
        dx, dy, ha = OFFSETS[r.bundle]
        ax.annotate(r.bundle, (r.seconds, r.mean_marginal),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=9, color=INK_2)

    ax.set_xscale("log")
    ax.set_xlabel("Acquisition cost (seconds, log scale)")
    ax.set_ylabel("Mean marginal gain in test macro F1")
    ax.set_title("Marginal value against acquisition cost",
                 color=INK, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(0.006, 6)
    lo, hi = bv.mean_marginal.min(), bv.mean_marginal.max()
    pad = (hi - lo) * 0.28
    ax.set_ylim(lo - pad, hi + pad)
    chrome(ax, ygrid=True)
    legend(ax, ["ask_caller", "map_lookup", "external_api"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=IN_DIR)
    args = ap.parse_args()

    bv = pd.read_csv(args.in_dir / "bundle_value.csv")
    sat = pd.read_csv(args.in_dir / "saturation.csv")

    plot_marginal(bv, args.in_dir / "voi_marginal.png")
    plot_saturation(sat, args.in_dir / "voi_saturation.png")
    plot_cost_benefit(bv, args.in_dir / "voi_cost_benefit.png")


if __name__ == "__main__":
    main()
