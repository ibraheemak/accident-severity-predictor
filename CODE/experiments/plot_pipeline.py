#!/usr/bin/env python3
"""The project pipeline, end to end, as one figure.

Four stages: raw sources -> feature table -> the 1,024-classifier bank ->
the RL agent that queries it. The point of the figure is the dependency: the
agent cannot exist until every subset has a trained classifier, which is why
the sweep sits between the pipeline and the agent rather than beside them.

Usage (from the repo root):
    python CODE/experiments/plot_pipeline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Same palette as the agent's analysis figures, so the report reads as one set.
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#26241f", "#6b6862", "#898781", "#e8e6e1"
BLUE, ORANGE, AQUA, ANCHOR = "#2a78d6", "#eb6834", "#1baf7a", "#7a5bb5"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9.5, "text.color": INK,
})

OUT = Path("docs/figures/pipeline")


def box(ax, x, y, w, h, title, body=None, color=INK2, fill="#ffffff",
        lw=1.2, title_size=9.5, body_size=8.2):
    """One rounded box. Title in colour, optional muted body beneath it."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09",
        linewidth=lw, edgecolor=color, facecolor=fill, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=title_size, color=color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                fontsize=body_size, color=INK2, zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, color=color, zorder=3, linespacing=1.45)


def band(ax, x, y, w, h, label, color):
    """The faint container behind each stage, with its stage label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0, facecolor=color, alpha=0.055, zorder=0))
    ax.text(x + 0.12, y + h - 0.22, label, ha="left", va="center",
            fontsize=8.6, color=color, fontweight="bold", zorder=1)


def arrow(ax, p0, p1, color=MUTED, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, shrinkA=1, shrinkB=1, zorder=4,
        connectionstyle="arc3,rad=0"))


fig, ax = plt.subplots(figsize=(9.4, 10.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.4)
ax.axis("off")

# ── Stage 1: sources ──────────────────────────────────────────────────────
band(ax, 0.3, 10.25, 9.4, 1.85, "1 · DATA SOURCES", BLUE)
srcs = [
    ("STATS19", "collisions, vehicles,\ncasualties · 2016–2025"),
    ("DfT AADF", "annual traffic\nvolume counts"),
    ("OpenStreetMap", "road infrastructure\n(Geofabrik, Overpass)"),
    ("Open-Meteo", "hourly historical\nweather"),
]
for i, (t, b) in enumerate(srcs):
    box(ax, 0.55 + i * 2.3, 10.4, 2.05, 1.18, t, b, color=BLUE)

for i in range(4):
    arrow(ax, (1.57 + i * 2.3, 10.4), (1.57 + i * 2.3, 9.95), BLUE)

# ── Stage 2: R pipeline ───────────────────────────────────────────────────
band(ax, 0.3, 7.05, 9.4, 2.95, "2 · FEATURE PIPELINE  (R, run_pipeline.R)", AQUA)
box(ax, 0.55, 8.6, 8.9, 1.02,
    "Spatial and temporal joins",
    "nearest AADF count point on the same road · OSM road matched by number,\n"
    "then by distance · weather at the collision's own coordinate and hour",
    color=AQUA)
box(ax, 0.55, 7.78, 4.3, 0.66, "Engineered features",
    "prior-year crash density within 500 m", color=AQUA)
box(ax, 5.15, 7.78, 4.3, 0.66, "Severity-aware sampling",
    "all Fatal kept; Serious / Slight capped", color=AQUA)
box(ax, 0.55, 7.15, 8.9, 0.52,
    "Split 70 / 15 / 15 by year × severity   →   cyclic encoding, log1p, "
    "z-score (train statistics only)", color=AQUA, title_size=8.6)

arrow(ax, (5.0, 7.15), (5.0, 6.8), AQUA)

# ── Stage 3: classifier bank ──────────────────────────────────────────────
band(ax, 0.3, 3.95, 9.4, 3.0, "3 · CLASSIFIER BANK  (bundle_sweep.py)", ORANGE)
box(ax, 0.55, 5.4, 8.9, 1.12,
    "Features grouped into 10 acquisition bundles",
    "ask the caller: vehicles · casualties · casualty demographics\n"
    "map lookup: road class · road environment · junction · crash history\n"
    "external API: AADF · OSM · meteo          free: the call timestamp",
    color=ORANGE)
box(ax, 0.55, 4.25, 8.9, 1.0,
    "One XGBoost classifier trained per subset  →  2¹⁰ = 1,024 models",
    "from timestamp-only up to full acquisition, cached by bitmask\n"
    "hyperparameters tuned with Optuna (TPE, grouped cross-validation)",
    color=ORANGE)

arrow(ax, (5.0, 4.25), (5.0, 3.6), ORANGE)

# ── Stage 4: the agent ────────────────────────────────────────────────────
band(ax, 0.3, 0.35, 9.4, 3.5, "4 · ACTIVE FEATURE ACQUISITION AGENT", ANCHOR)
box(ax, 0.55, 2.45, 8.9, 1.0,
    "MDP environment over the frozen bank",
    "state: bundles acquired (bitmask) + current classifier confidence, binned\n"
    "actions: acquire one of the remaining bundles, or stop and predict",
    color=ANCHOR)
box(ax, 0.55, 1.35, 4.3, 0.85, "Tabular Q-learning",
    "seeded from measured subset scores,\nthen trained for 12,000 episodes",
    color=ANCHOR)
box(ax, 5.15, 1.35, 4.3, 0.85, "Reward",
    "β·accuracy at the end,\nminus λ·seconds for each acquisition",
    color=ANCHOR)
box(ax, 0.55, 0.5, 8.9, 0.72,
    "Evaluated across 19 values of λ, against no acquisition, full acquisition,\n"
    "best fixed subset, static priority and random acquisition",
    color=ANCHOR, title_size=8.8)

# The feedback edge: the agent queries the bank at every step, it does not
# consume it once. Drawn on the right, outside the stage boxes.
ax.add_patch(FancyArrowPatch(
    (9.62, 2.9), (9.62, 4.75), arrowstyle="-|>", mutation_scale=11,
    linewidth=1.3, color=MUTED, zorder=4, linestyle=(0, (4, 2)),
    connectionstyle="arc3,rad=0"))
ax.text(9.75, 3.85, "queries the matching\nclassifier at every step",
        ha="left", va="center", fontsize=7.8, color=MUTED, rotation=90,
        linespacing=1.4)

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "pipeline.png")
