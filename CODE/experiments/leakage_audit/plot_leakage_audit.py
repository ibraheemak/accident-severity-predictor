"""plot_leakage_audit.py -- the leakage-audit figures.

Scores the two frozen, identically-tuned models on the same test split and
plots the gap between them:

    dispatch  models/classifier_aadf_meteo_osm_normalized_binary.json
              52 features -- only what a 999 dispatcher knows at call time.
    full      models/classifier_aadf_meteo_osm_normalized_binary_fullstats19.json
              83 features -- the same 52 plus every pre-label STATS19 field
              (officer-recorded / forensic; never a deployment candidate).

Both use the canonical tuned params from best_hyperparams_normalized_binary.json
(a dedicated 80-trial tune for the full feature set lost to them -- see README),
so the gap is an information gap, not a tuning artefact.

    fig1_leakage_gap.png       test metrics side by side, gap annotated
    fig2_feature_budget.png    the 52-vs-83 feature budget
    fig3_added_features_by_group.png   the added features, named by group
    leakage_audit_results.json the numbers behind both figures

Run from the repo root (needs data/splits/full_stats19_v2/, built by
build_full_stats19_features.py):

    python CODE/experiments/leakage_audit/plot_leakage_audit.py

Palette: the repo's validated categorical slots 1 (blue) and 2 (orange) --
blue is always the deployable dispatch model, orange the unusable ceiling.
Adjacent-pair CVD dE 24.7, normal-vision dE 33.6, both modes pass.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import (CATEGORICAL_COLS, MODEL_DIR, TARGET_COL, feature_columns,
                              load_split, prepare_xy, unify_categorical_dtypes)
from build_full_stats19_features import CATEGORICAL_EXTRA
from tune_full_stats19_model import FULL_EXTRA

OUT_DIR = Path("docs/figures/leakage_audit")
FULL_DIR = Path("data/splits/full_stats19_v2")
SOURCES = ["aadf", "osm", "meteo"]
VARIANT = "normalized"
DISPATCH_MODEL = MODEL_DIR / "classifier_aadf_meteo_osm_normalized_binary.json"
FULL_MODEL = MODEL_DIR / "classifier_aadf_meteo_osm_normalized_binary_fullstats19.json"

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

# The +31 fields the full model adds, grouped by where a real investigation gets
# them. Every one is officer-recorded after the fact -- none is knowable at the
# moment the 999 call lands.
ADDED_GROUPS = [
    ("Collision record", ["police_force", "local_authority_district", "road_surface_conditions",
                          "special_conditions_at_site", "carriageway_hazards"]),
    ("Police attendance", ["police_attended"]),
    ("Vehicle / driver", ["any_skidding", "any_hit_object_in", "any_left_carriageway",
                          "any_hit_object_off", "n_front_impact", "n_back_impact",
                          "dominant_manoeuvre", "dominant_junction_location",
                          "any_restricted_lane", "any_work_journey", "youngest_driver_age",
                          "oldest_driver_age", "n_male_drivers", "max_engine_cc",
                          "max_vehicle_age", "mean_driver_imd", "dominant_propulsion",
                          "min_driver_distance_band"]),
    ("Casualty detail", ["any_car_passenger", "any_bus_passenger", "any_road_worker",
                         "mean_casualty_imd", "dominant_casualty_distance_band",
                         "dominant_ped_location", "dominant_ped_movement"]),
]
GROUP_SHADES = ["#f6b699", "#f19a72", "#eb6834", "#c94f20"]

METRICS = [("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced acc."),
           ("roc_auc", "ROC-AUC"), ("severe_recall", "Severe recall")]


def scores(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "severe_recall": float(recall_score(y, pred, pos_label=1)),
        "slight_recall": float(recall_score(y, pred, pos_label=0)),
        "accuracy": float((pred == y).mean()),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n_test": int(len(y)),
    }


def score_dispatch():
    cols = feature_columns(SOURCES, VARIANT)
    frames = {s: load_split(s, cols, VARIANT) for s in ("train", "valid", "test")}
    unify_categorical_dtypes(list(frames.values()), cols)
    model = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
    model.load_model(DISPATCH_MODEL)
    X, y = prepare_xy(frames["test"], cols, binary=True)
    return len(cols), scores(model, X, y)


def score_full():
    feats = feature_columns(SOURCES, VARIANT) + FULL_EXTRA
    cat_cols = [c for c in CATEGORICAL_COLS + CATEGORICAL_EXTRA if c in feats]
    fr = {s: pd.read_csv(FULL_DIR / f"{s}_full_v2.csv") for s in ("train", "valid", "test")}
    Xs = {s: fr[s][feats].copy() for s in fr}
    for col in cat_cols:                       # categories from all splits, as at train time
        cats = sorted(pd.concat([f[col] for f in Xs.values()]).dropna().unique())
        for f in Xs.values():
            f[col] = f[col].astype(pd.CategoricalDtype(categories=cats))
    y = (fr["test"][TARGET_COL].astype(int) != 3).astype(int)
    model = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
    model.load_model(FULL_MODEL)
    return len(feats), scores(model, Xs["test"], y)


def fig_gap(d, f, out_path):
    """Grouped bars, one pair per metric. The gap is the point, so label it."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(METRICS))
    w = 0.34
    gap_px = 0.012                              # 2px surface gap between adjacent fills
    dv = [d[k] for k, _ in METRICS]
    fv = [f[k] for k, _ in METRICS]

    ax.bar(x - w / 2 - gap_px, dv, w, color=BLUE, zorder=3,
           label=f"Dispatch-time only  ({d['n_features']} features)")
    ax.bar(x + w / 2 + gap_px, fv, w, color=ORANGE, zorder=3,
           label=f"Full STATS19  ({f['n_features']} features)")

    for xi, (a, b) in enumerate(zip(dv, fv)):
        ax.text(xi - w / 2 - gap_px, a + .006, f"{a:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.text(xi + w / 2 + gap_px, b + .006, f"{b:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.text(xi, max(a, b) + .045, f"+{b - a:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=INK_2, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylim(0.5, max(fv) + .085)
    ax.set_ylabel("Test score")
    ax.yaxis.grid(True, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9)
    ax.set_title("The cost of deployability: what the dispatch-time constraint gives up",
                 loc="left", pad=34)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_budget(d, f, out_path):
    """One stacked bar per model, shared 0-83 scale. Groups named in fig3."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    gap = 0.35                                  # surface gap between stacked segments

    ax.barh(1, d["n_features"], height=.5, color=BLUE, zorder=3)
    ax.text(d["n_features"] / 2, 1, f"{d['n_features']} dispatch-time features",
            ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    ax.barh(0, d["n_features"] - gap, height=.5, color=BLUE, zorder=3)
    ax.text(d["n_features"] / 2, 0, f"{d['n_features']}", ha="center", va="center",
            color="white", fontsize=9.5, fontweight="bold")
    left = d["n_features"]
    for (name, cols), shade in zip(ADDED_GROUPS, GROUP_SHADES):
        n = len(cols)
        ax.barh(0, n - gap, height=.5, left=left, color=shade, zorder=3)
        # Every segment carries its count. The +1 sliver is ~5px wide, so its
        # label is smaller and sits in ink -- white on the two pale shades
        # would fall under 3:1.
        ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                color=INK if shade in GROUP_SHADES[:2] else "white",
                fontsize=7.5 if n < 3 else 9, fontweight="bold")
        left += n

    handles = [plt.Rectangle((0, 0), 1, 1, color=s) for s in GROUP_SHADES]
    labels = [f"+{len(c)}  {n}" for n, c in ADDED_GROUPS]
    ax.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.42),
              ncol=4, fontsize=8.5, handlelength=1.1, columnspacing=1.2)

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Dispatch-time\n(deployable)", "Full STATS19\n(upper bound)"],
                       fontsize=9.5)
    ax.set_xlim(0, f["n_features"] + 2)
    ax.set_xlabel("Model features")
    ax.xaxis.grid(True, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("What the full-STATS19 model knows that a 999 dispatcher cannot",
                 loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_groups(d, f, out_path):
    """The added features, named. One row per group: heading, bar, the fields.

    Row heights are computed from each group's wrapped field list (18 names take
    far more room than 1), and the figure height follows from their sum -- a
    fixed row pitch makes the long groups collide with the next heading. Same
    shade per group as fig2's stacked bar, so the two read as one story. Field
    names are small and muted: a reference list, not the first thing the eye
    should land on.
    """
    WRAP = 76               # characters per line of the monospace field list
    LINE = 0.62             # vertical units per wrapped line
    HEAD, BAR, PAD = 0.55, 1.15, 0.75   # units to heading, to bar, after the list

    wrapped = [textwrap.fill("  ".join(cols), WRAP).split("\n") for _, cols in ADDED_GROUPS]
    heights = [HEAD + BAR + LINE * len(w) + PAD for w in wrapped]
    total = sum(heights)

    fig, ax = plt.subplots(figsize=(8.6, total * 0.30))
    counts = [len(c) for _, c in ADDED_GROUPS]

    top = total
    for (name, cols), shade, lines, h in zip(ADDED_GROUPS, GROUP_SHADES, wrapped, heights):
        ax.text(0, top - HEAD, name, va="center", ha="left", fontsize=10.5,
                fontweight="bold", color=INK)
        y_bar = top - HEAD - BAR + 0.35
        ax.barh(y_bar, len(cols), height=0.46, color=shade, zorder=3)
        ax.text(len(cols) + 0.25, y_bar, str(len(cols)), va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=INK)
        for j, line in enumerate(lines):
            ax.text(0, y_bar - 0.55 - LINE * j, line, va="top", ha="left",
                    fontsize=7.2, color=MUTED, family="monospace")
        top -= h

    ax.set_ylim(0, total)
    ax.set_xlim(0, max(counts) + 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Features added")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # counts, not 2.5 of a feature
    ax.xaxis.grid(True, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("The features the full-STATS19 model adds, by group\n"
                 "Every one is recorded by an officer after the fact",
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_d, d = score_dispatch()
    n_f, f = score_full()
    d["n_features"], f["n_features"] = n_d, n_f
    added = sum(len(c) for _, c in ADDED_GROUPS)
    assert added == n_f - n_d, f"group table lists {added}, models differ by {n_f - n_d}"

    for k, lab in METRICS:
        print(f"{lab:<16} dispatch {d[k]:.4f}   full {f[k]:.4f}   gap {f[k] - d[k]:+.4f}")

    fig_gap(d, f, OUT_DIR / "fig1_leakage_gap.png")
    fig_budget(d, f, OUT_DIR / "fig2_feature_budget.png")
    fig_groups(d, f, OUT_DIR / "fig3_added_features_by_group.png")
    with open(OUT_DIR / "leakage_audit_results.json", "w") as fh:
        json.dump({"dispatch": d, "full": f,
                   "gap": {k: f[k] - d[k] for k, _ in METRICS},
                   "features_added": {n: len(c) for n, c in ADDED_GROUPS},
                   "n_features_added": added}, fh, indent=2)
    print(f"wrote {OUT_DIR / 'leakage_audit_results.json'}")


if __name__ == "__main__":
    main()
