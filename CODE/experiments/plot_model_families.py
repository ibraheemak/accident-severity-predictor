"""plot_model_families.py -- figures for compare_model_families.py.

Kept separate so the figures can be regenerated from the saved CSVs without
retraining anything:

    python CODE/experiments/plot_model_families.py

Palette: three categorical hues (blue / orange / aqua), validated for
colour-vision deficiency separation; every chart is single-measure so colour
carries at most "this is the incumbent model" and never encodes a number.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

OUT_DIR = Path("docs/figures/model_families")

# Validated categorical slots (light surface) + ink/surface tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#9c9b95"
INCUMBENT = "XGBoost"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def _style(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_macro_f1(summary, out):
    """Magnitude by identity -> horizontal bars, sorted, incumbent highlighted."""
    d = summary.sort_values("macro_f1")
    colors = [ORANGE if m == INCUMBENT else BLUE for m in d.model]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    bars = ax.barh(d.model, d.macro_f1, color=colors, height=0.62,
                   xerr=d.macro_f1_sd.fillna(0), error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
    for bar, v in zip(bars, d.macro_f1):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(d.macro_f1) * 1.16)
    ax.set_xlabel("Test macro F1  (mean over seeds, bars = ±1 sd)")
    ax.set_title("Test macro F1 by model family")
    ax.text(0.99, 0.02, f"orange = deployed classifier ({INCUMBENT})", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK_2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig1_macro_f1.png")
    plt.close(fig)


def fig_roc(curves, y_true, summary, out):
    top = summary.nlargest(5, "auc").model.tolist()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), lw=1.2, color=NEUTRAL, label="Chance (AUC 0.50)")
    for name, color in zip(top, [BLUE, ORANGE, AQUA, INK_2, NEUTRAL]):
        fpr, tpr, _ = roc_curve(y_true, curves[name])
        auc = summary.loc[summary.model == name, "auc"].iloc[0]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  ({auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — five highest-AUC families")
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="AUC", title_fontsize=9)
    _style(ax, xgrid=False)
    ax.grid(axis="x", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out / "fig2_roc.png")
    plt.close(fig)


def fig_pr(curves, y_true, summary, out):
    top = summary.nlargest(5, "pr_auc").model.tolist()
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.axhline(prevalence, ls=(0, (4, 4)), lw=1.2, color=NEUTRAL)
    ax.text(0.02, prevalence + 0.012, f"No-skill baseline (prevalence {prevalence:.2f})",
            fontsize=8.5, color=INK_2)
    for name, color in zip(top, [BLUE, ORANGE, AQUA, INK_2, NEUTRAL]):
        prec, rec, _ = precision_recall_curve(y_true, curves[name])
        ap = summary.loc[summary.model == name, "pr_auc"].iloc[0]
        ax.plot(rec, prec, lw=2, color=color, label=f"{name}  ({ap:.3f})")
    ax.set_xlabel("Recall (Severe)")
    ax.set_ylabel("Precision (Severe)")
    ax.set_ylim(0, 1)
    ax.set_title("Precision–recall curves — Severe class")
    ax.legend(loc="lower left", frameon=False, fontsize=9, title="PR-AUC", title_fontsize=9)
    _style(ax, xgrid=False)
    ax.grid(axis="x", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out / "fig3_pr.png")
    plt.close(fig)


def fig_time_vs_score(summary, out):
    """Cost, with identity on the axis.

    A time-vs-score scatter puts eleven of fifteen families in one unreadable
    blob -- which is itself the finding -- so cost is shown against the score
    ordering instead: rows are sorted by macro F1, the x position is training
    time, and the score rides along as a direct label.
    """
    d = summary.sort_values("macro_f1")
    colors = [ORANGE if m == INCUMBENT else BLUE for m in d.model]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.hlines(d.model, d.fit_seconds.min() * 0.5, d.fit_seconds, color=GRID, lw=1.2, zorder=1)
    ax.scatter(d.fit_seconds, d.model, s=62, c=colors, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    for _, r in d.iterrows():
        ax.text(r.fit_seconds * 1.25, r.model, f"{r.fit_seconds:.2f}s  ·  F1 {r.macro_f1:.3f}",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(d.fit_seconds.min() * 0.5, d.fit_seconds.max() * 9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.set_xlabel("Training time, log scale (rows ordered by macro F1)")
    ax.set_title("Training time by model family")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig4_time_vs_score.png")
    plt.close(fig)


def fig_metric_panels(summary, out):
    """Three measures on different scales -> small multiples, never a dual axis."""
    metrics = [("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"),
               ("auc", "AUC")]
    order = summary.sort_values("macro_f1").model.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for ax, (col, label) in zip(axes, metrics):
        d = summary.set_index("model").loc[order]
        colors = [ORANGE if m == INCUMBENT else BLUE for m in order]
        # Dot plot, not bars: these panels are zoomed to a narrow range, and a bar
        # whose axis does not start at zero misstates the ratio between families.
        ax.hlines(order, d[col].min() - 0.06, d[col], color=GRID, lw=1.2, zorder=1)
        ax.scatter(d[col], order, s=58, c=colors, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.set_xlim(d[col].min() - 0.06, min(1.0, d[col].max() + 0.03))
        ax.set_title(label)
        _style(ax)
    axes[0].set_ylabel("")
    fig.suptitle("Macro F1, balanced accuracy and AUC by model family",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig5_metric_panels.png")
    plt.close(fig)


def fig_confusion(confusions, summary, out):
    top = summary.nlargest(4, "macro_f1").model.tolist()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9))
    labels = ["Slight", "Severe"]
    for ax, name in zip(axes, top):
        cm = confusions[name]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center",
                        fontsize=9, color=INK if cmn[i, j] < 0.55 else SURFACE)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices (row-normalised) — four highest-scoring families",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig6_confusion.png")
    plt.close(fig)


# tuned-table name -> untuned-table name, for the before/after figure
TUNED_TO_DEFAULT = {
    "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
    "HistGradientBoosting": "HistGradientBoosting", "Random forest": "Random forest",
    "Extra trees": "Extra trees", "Neural net (MLP)": "Neural net (MLP)",
    "Logistic regression": "Logistic regression", "Linear SVM": "Linear SVM",
    "k-NN": "k-NN (k=25)", "Decision tree": "Decision tree (depth 5)",
    "Gaussian naive Bayes": "Gaussian naive Bayes",
}


def fig_tuning_gain(tuned, summary, out):
    """Default vs tuned per family -> dumbbell, sorted by tuned score."""
    d = summary.set_index("model").macro_f1
    rows = [(fam, d[dflt], tuned.loc[fam, "test_f1"])
            for fam, dflt in TUNED_TO_DEFAULT.items()
            if dflt in d.index and fam in tuned.index]
    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    before = np.array([r[1] for r in rows])
    after = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.hlines(names, before, after, color=GRID, lw=3, zorder=1)
    ax.scatter(before, names, s=66, color=NEUTRAL, zorder=3, edgecolor=SURFACE,
               linewidth=1.3, label="Library defaults")
    ax.scatter(after, names, s=66, color=BLUE, zorder=3, edgecolor=SURFACE,
               linewidth=1.3, label="Tuned, 120 trials")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(max(a, b) + 0.004, i, f"{a - b:+.4f}", va="center", fontsize=8.5,
                color=INK if a - b > 0.004 else INK_2)
    ax.set_xlim(min(before.min(), after.min()) - 0.02, max(after.max(), before.max()) + 0.035)
    ax.set_xlabel("Test macro F1  (mean of 3 seeds)")
    ax.set_title("Library defaults vs 120-trial tuning, by model family")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig7_tuning_gain.png")
    plt.close(fig)


def make_figures(summary, results, curves, confusions, y_true, out=OUT_DIR):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_macro_f1(summary, out)
    fig_time_vs_score(summary, out)
    fig_metric_panels(summary, out)
    if curves:
        fig_roc(curves, y_true, summary, out)
        fig_pr(curves, y_true, summary, out)
    if confusions:
        fig_confusion(confusions, summary, out)
    tuned_csv = out / "tuned_summary.csv"
    if tuned_csv.exists():
        fig_tuning_gain(pd.read_csv(tuned_csv).set_index("family"), summary, out)


if __name__ == "__main__":
    # Regenerate the score-only figures from saved CSVs (no retraining; the ROC,
    # PR and confusion figures need per-row scores, so they come from a run).
    summary = pd.read_csv(OUT_DIR / "model_families_summary.csv")
    results = pd.read_csv(OUT_DIR / "model_families_results.csv")
    make_figures(summary, results, {}, {}, None)
    print(f"figures -> {OUT_DIR}/")
