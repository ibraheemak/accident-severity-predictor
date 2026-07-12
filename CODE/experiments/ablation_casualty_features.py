#!/usr/bin/env python3
"""ablation_casualty_features.py

Clean ablation: do the casualty features actually help the binary (Slight-vs-
Severe) classifier on the normalized (unmasked) variant? Everything is held
fixed -- tuned hyperparameters, splits, threshold-calibration procedure --
and only the feature set is toggled:

  full            all 59 features (current model)
  drop new-5      remove the 5 recently-added casualty features
  drop all-9      remove every casualty feature (original 4 + new 5)

Each config is trained over several seeds (XGBoost is slightly stochastic via
column/row subsampling) and the mean test metric is reported, so a ~0.003
difference isn't mistaken for signal. Threshold is chosen per-run on valid to
maximize macro F1, then applied to test.

Usage:
    python ablation_casualty_features.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import (
    MODEL_DIR, feature_columns, load_split, prepare_xy, unify_categorical_dtypes,
)

SOURCES = ["aadf", "osm", "meteo"]

ORIGINAL_4 = ["n_casualties", "n_pedestrian_casualties",
              "youngest_casualty_age", "oldest_casualty_age"]
NEW_5 = ["n_child_casualties", "n_elderly_casualties", "casualty_age_span",
         "n_cyclist_casualties", "n_motorcyclist_casualties"]


def best_threshold(y, p):
    ts = np.linspace(0.1, 0.9, 81)
    return float(ts[int(np.argmax([f1_score(y, (p >= t).astype(int), average="macro") for t in ts]))])


def run_config(name, drop, train, valid, test, base_cols, xgb_params, seeds):
    cols = [c for c in base_cols if c not in drop]
    Xtr, ytr = prepare_xy(train, cols, binary=True)
    Xva, yva = prepare_xy(valid, cols, binary=True)
    Xte, yte = prepare_xy(test, cols, binary=True)
    unify_categorical_dtypes([Xtr, Xva, Xte], cols)
    sw = compute_sample_weight("balanced", ytr)

    mf1s, accs, bals = [], [], []
    for seed in seeds:
        m = xgb.XGBClassifier(
            n_estimators=1000, tree_method="hist", enable_categorical=True,
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=30, random_state=seed, n_jobs=-1, **xgb_params,
        )
        m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        p_va = m.predict_proba(Xva)[:, 1]
        p_te = m.predict_proba(Xte)[:, 1]
        t = best_threshold(yva, p_va)
        pred = (p_te >= t).astype(int)
        mf1s.append(f1_score(yte, pred, average="macro"))
        accs.append(accuracy_score(yte, pred))
        bals.append(balanced_accuracy_score(yte, pred))
    print(f"  {name:<16}({len(cols)} feats)  "
          f"macro F1 {np.mean(mf1s):.4f} ±{np.std(mf1s):.4f}   "
          f"acc {np.mean(accs):.4f}   bal-acc {np.mean(bals):.4f}")
    return np.mean(mf1s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-variant", choices=["normalized", "masked", "raw"], default="normalized")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--skip-all9", action="store_true",
                         help="Skip the drop-all-9 config (the expensive masked runs "
                              "only need full vs drop-new-5 to decide on the new features).")
    args = parser.parse_args()
    variant = args.data_variant

    base_cols = feature_columns(SOURCES, variant)
    train = load_split("train", base_cols, variant)
    valid = load_split("valid", base_cols, variant)
    test = load_split("test", base_cols, variant)
    variant_suffix = "" if variant == "masked" else f"_{variant}"
    with open(MODEL_DIR / f"best_hyperparams{variant_suffix}_binary.json") as f:
        xgb_params = json.load(f)["params"]

    print(f"Binary / {variant} ablation, mean over seeds {args.seeds} "
          f"(test set, {len(train)} train rows):\n")
    full = run_config("full", [], train, valid, test, base_cols, xgb_params, args.seeds)
    d5 = run_config("drop new-5", NEW_5, train, valid, test, base_cols, xgb_params, args.seeds)
    print(f"\n  new-5 casualty features worth:  {full - d5:+.4f} macro F1")
    if not args.skip_all9:
        d9 = run_config("drop all-9", ORIGINAL_4 + NEW_5, train, valid, test, base_cols, xgb_params, args.seeds)
        print(f"  all-9 casualty features worth:  {full - d9:+.4f} macro F1")


if __name__ == "__main__":
    main()
