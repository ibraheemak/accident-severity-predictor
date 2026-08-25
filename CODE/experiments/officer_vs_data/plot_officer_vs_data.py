#!/usr/bin/env python3
"""plot_officer_vs_data.py -- figures for the Officer vs. Data Gap experiment.

Regenerates figures from the CSVs weather_agreement.py and weather_predictive_test.py
already wrote, so nothing is retrained here:

    python3 CODE/experiments/officer_vs_data/plot_officer_vs_data.py

Figures (docs/figures/officer_vs_data/):
  fig1_agreement_matrix.png     officer category x Open-Meteo condition, row %
  fig4_model_deltas.png         test scores (3 metrics) vs. no-weather baseline

Colour: two categorical hues only -- blue = the officer's field, orange =
Open-Meteo -- so hue always means "which source", never a magnitude. The
agreement matrix is the one exception and uses a single-hue sequential ramp,
since its cells encode a percentage. Palette validated for colour-vision
deficiency separation (blue/orange adjacent pair, normal dE 33.6, protan 24.7).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUT_DIR = Path("docs/figures/officer_vs_data")

BLUE, ORANGE = "#2a78d6", "#eb6834"       # officer / open-meteo
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#9c9b95"
SOURCE_COLOR = {"officer": BLUE, "open-meteo": ORANGE}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#a8c8ee", "#2a78d6", "#123a69"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

OFFICER_ORDER = [
    "Fine, no high winds", "Fine + high winds", "Raining, no high winds",
    "Raining + high winds", "Snowing, no high winds", "Snowing + high winds",
    "Fog or mist", "Other", "Unknown",
]
BLOCK_LABEL = {"none": "no weather", "officer": "officer field",
               "meteo": "Open-Meteo", "both": "both"}
BLOCK_COLOR = {"none": NEUTRAL, "officer": BLUE, "meteo": ORANGE, "both": "#6b4fa8"}


def _style(ax, axis="x"):
    ax.grid(axis=axis, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_agreement_matrix(out: Path):
    """Two categorical fields against each other -> a matrix. Cells are row
    percentages (what Open-Meteo saw, given what the officer wrote), so each
    row reads as one officer category's objective reality."""
    ct = pd.read_csv(OUT_DIR / "agreement_crosstab.csv", index_col=0)
    ct = ct.reindex([r for r in OFFICER_ORDER if r in ct.index])
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(pct.values, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pct.columns)), pct.columns)
    ax.set_yticks(range(len(pct.index)),
                  [f"{r}  (n={ct.loc[r].sum():,})" for r in pct.index])
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.iat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color=SURFACE if v > 55 else INK)
    ax.set_xlabel("Open-Meteo condition at that coordinate and hour")
    ax.set_ylabel("Officer's STATS19 weather field")
    ax.set_title("What the sky was doing, by what the officer wrote", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.72, label="% of that officer category")
    fig.text(0.5, -0.02, "rows sum to 100%  ·  perfect agreement would put every "
             "row on its matching column", ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_disagreement(out: Path):
    """Two shared claims x two directions of backing -> grouped bars, one
    panel per claim so the two kappas are not implied to be comparable
    magnitudes on one axis."""
    b = pd.read_csv(OUT_DIR / "agreement_binary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    for ax, (_, r) in zip(axes, b.iterrows()):
        labels = ["Officer said so —\nsensor agreed",
                  "Sensor said so —\nofficer agreed"]
        vals = [r.objective_backs_officer, r.officer_backs_objective]
        bars = ax.barh(labels, vals, color=[BLUE, ORANGE], height=0.42)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.0%}",
                    va="center", fontsize=10, color=INK)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.invert_yaxis()
        ax.set_ylim(1.62, -0.62)
        ax.set_title(f"{r.claim}   ·   Cohen's κ = {r.cohen_kappa:.2f}", loc="left", pad=26)
        ax.set_xlabel("share of that source's calls the other source backs")
        ax.text(0, 1.03, f"claimed on {r.officer_rate:.1%} of collisions by the officer, "
                f"{r.objective_rate:.1%} by the sensor",
                transform=ax.transAxes, fontsize=8.5, color=INK_2)
        _style(ax)
    fig.suptitle("Where both sources describe the same thing, they disagree half the time",
                 fontsize=12.5, fontweight="bold", y=1.16)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_model_deltas(out: Path):
    """Paired bars (means over 5 seeds), full dispatch feature set only: values
    on bars, deltas vs. control shown above. Styled like leakage_audit."""
    summary = pd.read_csv(OUT_DIR / "weather_model_summary.csv")
    blocks = ["none", "officer", "meteo", "both"]
    metrics = [("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced Acc."), ("roc_auc", "ROC AUC")]

    d = summary[summary.family == "full"].set_index("weather")

    x = 0
    bar_data = []  # (x_pos, value, color, block)
    metric_xs = []  # (metric_label, start_x, end_x)

    for metric, mlabel in metrics:
        group_x = []
        for block in blocks:
            val = d.loc[block, f"{metric}_mean"]
            bar_data.append((x, val, BLOCK_COLOR[block], block))
            group_x.append(x)
            x += 1
        metric_xs.append((mlabel, min(group_x), max(group_x)))
        x += 1.6

    # Plot
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    xs = [b[0] for b in bar_data]
    vals = [b[1] for b in bar_data]
    colors = [b[2] for b in bar_data]
    bars = ax.bar(xs, vals, color=colors, width=0.68, edgecolor="none")

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
               f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")

    # Metric labels and deltas
    for mlabel, m_start, m_end in metric_xs:
        m_center = (m_start + m_end) / 2
        ax.text(m_center, -0.035, mlabel, ha="center", va="top",
               fontsize=9.5, fontweight="bold", color=INK,
               transform=ax.get_xaxis_transform())

        # Get baseline (officer) and max for this metric
        metric_bars = [b for b in bar_data if m_start <= b[0] <= m_end]
        baseline = metric_bars[0][1]  # first is officer
        max_val = max(b[1] for b in metric_bars)

        for bar_x, val, col, block in metric_bars:
            if block != "none":
                delta = val - baseline
                ax.text(bar_x, max_val + 0.038, f"{delta:+.5f}", ha="center",
                       va="bottom", fontsize=7, color=INK_2, style="italic")

    ax.set_xlim(-1.2, x - 0.5)
    ax.set_ylim(-0.052, max(vals) * 1.22)
    ax.set_xticks([])
    ax.set_ylabel("Test score (mean over 5 seeds)", fontsize=10)


    # Legend
    from matplotlib.patches import Patch
    legend_labels = {"none": "No weather", "officer": "Officer field", "meteo": "Open-Meteo", "both": "Both"}
    legend_elements = [Patch(facecolor=BLOCK_COLOR[b], label=legend_labels[b]) for b in blocks]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)

    _style(ax, axis="y")
    fig.suptitle("Test scores",
                fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_permutation(out: Path):
    """Sorted horizontal bars, no error bars, values labeled. Colour by source."""
    perm = pd.read_csv(OUT_DIR / "weather_permutation.csv").sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    colors = [SOURCE_COLOR[s] for s in perm.source]
    bars = ax.barh(perm.feature, perm.importance_mean, color=colors, height=0.62, edgecolor="none")

    # Value labels
    for bar, v in zip(bars, perm.importance_mean):
        side = 1 if v >= 0 else -1
        ax.text(v + side * 0.00005, bar.get_y() + bar.get_height() / 2, f"{v:+.5f}",
               va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)

    ax.set_xlim(min(perm.importance_mean) * 1.5, max(perm.importance_mean) * 1.4)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("drop in test ROC AUC when shuffled (10 repeats)")
    ax.set_title("Both sources in one model: neither moves the needle")

    handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=c)
               for c in (BLUE, ORANGE)]
    ax.legend(handles, ["officer field", "Open-Meteo"], frameon=False, fontsize=9, loc="lower right")

    ref_feature = perm.reference_feature.iloc[0]
    ref_value = perm.reference_importance.iloc[0]
    ratio = ref_value / perm.importance_mean.max()
    fig.text(0.5, -0.02,
             f"for scale: strongest non-weather feature ({ref_feature}) = {ref_value:.4f} — {ratio:.0f}x "
             f"the largest weather bar",
             ha="center", fontsize=8.5, color=INK_2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig1_agreement_matrix.png", fig_agreement_matrix),
        ("fig4_model_deltas.png", fig_model_deltas),
    ]
    for name, fn in figs:
        fn(OUT_DIR / name)
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
