#!/usr/bin/env python3
"""weather_predictive_test.py -- which weather source predicts severity better?

Part 2 of the "Officer vs. Data Gap" experiment. Trains matched XGBoost models
that differ ONLY in the weather block, added to the full dispatch feature set.

Protocol: normalized variant, binary Slight-vs-Severe, balanced sample weights,
Optuna-tuned hyperparameters, early stopping on valid, scored on test, repeated
over seeds. Paired per-seed deltas against the no-weather control.

Output for fig4 (model_deltas.png):
  weather_model_summary.csv    per-configuration means, SDs, paired deltas

Usage:
  python CODE/experiments/officer_vs_data/weather_predictive_test.py
  python CODE/experiments/officer_vs_data/weather_predictive_test.py --seeds 42 43 44 45 46
  python CODE/experiments/officer_vs_data/weather_predictive_test.py --quick   # 1 seed
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

from train_classifier import (BASE_COLS, CATEGORICAL_COLS, MODEL_DIR, SOURCE_COLS,
                              TARGET_COL, free_columns, load_split, prepare_xy,
                              unify_categorical_dtypes)

VARIANT = "normalized"
OUT_DIR = Path("docs/figures/officer_vs_data")

OFFICER_COLS = ["weather_conditions"]
METEO_COLS = SOURCE_COLS["meteo"]
WEATHER_BLOCKS = {
    "none": [],
    "officer": OFFICER_COLS,
    "meteo": METEO_COLS,
    "both": OFFICER_COLS + METEO_COLS,
}
BLOCK_ORDER = list(WEATHER_BLOCKS)
FAMILY_BASE = {
    "full": lambda: free_columns(VARIANT) + list(BASE_COLS)
                    + SOURCE_COLS["aadf"] + SOURCE_COLS["osm"],
    "weather-only": lambda: free_columns(VARIANT),
}


def tuned_params() -> dict:
    path = MODEL_DIR / f"best_hyperparams_{VARIANT}_binary.json"
    if not path.exists():
        path = MODEL_DIR / "best_hyperparams_binary.json"
    if not path.exists():
        print("No tuned hyperparameters found -- using max_depth=6, lr=0.05")
        return {"max_depth": 6, "learning_rate": 0.05}
    with open(path) as f:
        return json.load(f)["params"]


def load_all(cols: list[str]):
    """Load train/valid/test for one column set, with categorical dtypes unified."""
    frames = {s: load_split(s, cols, VARIANT) for s in ("train", "valid", "test")}
    Xy = {s: prepare_xy(frames[s], cols, binary=True) for s in frames}
    unify_categorical_dtypes([Xy[s][0] for s in Xy], cols)
    return Xy


def fit_one(Xy, params: dict, seed: int):
    X_tr, y_tr = Xy["train"]
    X_va, y_va = Xy["valid"]
    model = xgb.XGBClassifier(
        n_estimators=1000, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=30, random_state=seed, n_jobs=-1,
        objective="binary:logistic", eval_metric="logloss", **params,
    )
    model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr),
              eval_set=[(X_va, y_va)], verbose=False)
    return model


def score(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "severe_recall": recall_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc_severe": average_precision_score(y, proba),
    }


METRICS = ["macro_f1", "balanced_accuracy", "severe_recall", "roc_auc", "pr_auc_severe"]


def run(seeds: list[int]) -> pd.DataFrame:
    params = tuned_params()
    rows = []
    for family, base_fn in FAMILY_BASE.items():
        base = base_fn()
        for block, weather in WEATHER_BLOCKS.items():
            cols = base + weather
            Xy = load_all(cols)
            for seed in seeds:
                t0 = time.perf_counter()
                model = fit_one(Xy, params, seed)
                metrics = score(model, *Xy["test"])
                rows.append({"family": family, "weather": block, "seed": seed,
                             "n_features": len(cols),
                             "fit_seconds": time.perf_counter() - t0, **metrics})
                print(f"{family:12s} {block:8s} seed {seed}  "
                      f"macro F1 {metrics['macro_f1']:.4f}  AUC {metrics['roc_auc']:.4f}")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    agg = (results.groupby(["family", "weather"])[METRICS]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["family", "weather"] + [f"{m}_{s}" for m, s in agg.columns[2:]]

    control = results[results.weather == "none"].set_index(["family", "seed"])
    deltas = []
    for (family, block), grp in results.groupby(["family", "weather"]):
        paired = grp.set_index(["family", "seed"])
        for metric in METRICS:
            d = paired[metric] - control[metric]
            deltas.append({"family": family, "weather": block, "metric": metric,
                           "delta_mean": d.mean(), "delta_std": d.std()})
    d = pd.DataFrame(deltas).pivot(index=["family", "weather"], columns="metric",
                                   values=["delta_mean", "delta_std"])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    return agg.merge(d.reset_index(), on=["family", "weather"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--quick", action="store_true", help="one seed, smoke test")
    args = ap.parse_args()
    seeds = [42] if args.quick else args.seeds

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run(seeds)
    results.to_csv(OUT_DIR / "weather_model_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(OUT_DIR / "weather_model_summary.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n-- Test macro F1 / AUC by feature set (mean +/- sd over seeds) --")
    for family in FAMILY_BASE:
        sub = summary[summary.family == family].set_index("weather").loc[BLOCK_ORDER]
        print(f"\n[{family}]")
        for block, r in sub.iterrows():
            print(f"  {block:8s} macro F1 {r.macro_f1_mean:.4f} +/- {r.macro_f1_std:.4f}"
                  f"   AUC {r.roc_auc_mean:.4f} +/- {r.roc_auc_std:.4f}"
                  f"   delta F1 vs none {r.delta_mean_macro_f1:+.4f}")

    print(f"\nWrote CSVs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
