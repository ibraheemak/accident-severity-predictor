"""plot_xgboost_performance.py -- headline figures for the deployed XGBoost classifier.

Two figures for the full-information reference model
(models/classifier_aadf_meteo_osm_normalized_binary.json, normalized variant,
binary Slight-vs-Severe):

    fig1_confusion_matrix.png   test confusion matrix (row % headline + counts)
    fig2_headline_metrics.png   macro F1 / balanced accuracy / ROC-AUC on test

Run from the repo root:

    python CODE/experiments/xgboost_report/plot_xgboost_performance.py

Palette: the repo's validated categorical blue for the single bar series, and a
single-hue blue sequential ramp for the confusion-matrix cells -- magnitude never
gets a categorical hue and never a rainbow.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))
from train_classifier import (BINARY_LABELS, MODEL_DIR, feature_columns, load_split,
                              prepare_xy, unify_categorical_dtypes)

OUT_DIR = Path("docs/figures/xgboost_report")
MODEL_PATH = MODEL_DIR / "classifier_aadf_meteo_osm_normalized_binary.json"
SOURCES = ["aadf", "osm", "meteo"]
VARIANT = "normalized"
SPLIT = "test"

BLUE = "#2a78d6"
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
# Sequential ramp, one hue, light -> dark (palette steps 100 -> 650).
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

SHORT = {"Slight": "Slight", "Severe (Fatal+Serious)": "Severe"}


def score_splits():
    """Load the frozen model and score each split. Returns {split: {...}}."""
    cols = feature_columns(SOURCES, VARIANT)
    frames = {s: load_split(s, cols, VARIANT) for s in ["train", "valid", SPLIT]}
    unify_categorical_dtypes(list(frames.values()), cols)

    model = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
    model.load_model(MODEL_PATH)

    X, y = prepare_xy(frames[SPLIT], cols, binary=True)
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "cm": confusion_matrix(y, y_pred),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "n": int(len(y)),
    }


def fig_confusion(scores, out_path):
    """One matrix: cell colour and headline number are the row-normalised share of
    the true class (recall on the diagonal); the raw count sits underneath."""
    labels = [SHORT[BINARY_LABELS[i]] for i in range(len(BINARY_LABELS))]
    cm = scores["cm"]
    row_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(row_pct, cmap=SEQ, vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Ink flips to white only where the cell is dark enough to need it.
            colour = "#ffffff" if row_pct[i, j] > 0.55 else INK
            ax.text(j, i - 0.03, f"{row_pct[i, j]:.1%}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=colour)
            ax.text(j, i + 0.19, f"{cm[i, j]:,} collisions", ha="center", va="center",
                    fontsize=9.5, color=colour)

    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"XGBoost confusion matrix — test set (n = {scores['n']:,})", pad=12)
    # 2px surface gap between adjacent fills.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_metrics(scores, out_path):
    """Three single-measure metrics on the test set. All three share the 0-1 scale
    and the same 0.50 chance floor on this near-balanced binary task, so one axis.
    One series, so no legend -- the title names it."""
    metrics = [("macro_f1", "Macro F1"),
               ("balanced_accuracy", "Balanced accuracy"),
               ("roc_auc", "ROC-AUC")]
    labels = [label for _, label in metrics]
    vals = [scores[k] for k, _ in metrics]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    bars = ax.bar(x, vals, 0.42, color=BLUE, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.014, f"{v:.3f}",
                ha="center", fontsize=11, color=INK, fontweight="bold")

    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(-0.94, 0.5, "chance\n0.50", ha="left", va="center", fontsize=8.5,
            color=MUTED, linespacing=1.4)

    ax.set_xticks(x, labels)
    ax.set_xlim(-1.0, len(metrics) - 0.45)  # left gutter holds the chance-line label
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("XGBoost test performance")
    ax.grid(axis="y", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = score_splits()
    print(f"[{SPLIT}] macro F1 = {scores['macro_f1']:.4f}  "
          f"balanced acc = {scores['balanced_accuracy']:.4f}  ROC-AUC = {scores['roc_auc']:.4f}")
    fig_confusion(scores, OUT_DIR / "fig1_confusion_matrix.png")
    fig_metrics(scores, OUT_DIR / "fig2_headline_metrics.png")

    summary = {SPLIT: {k: v for k, v in scores.items() if k != "cm"}
                      | {"confusion_matrix": scores["cm"].tolist()}}
    (OUT_DIR / "xgboost_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'xgboost_metrics.json'}")


if __name__ == "__main__":
    main()
