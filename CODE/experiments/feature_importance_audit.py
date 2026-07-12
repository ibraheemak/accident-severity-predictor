#!/usr/bin/env python3
"""feature_importance_audit.py

Diagnostic for an already-trained classifier (see train_classifier.py) --
no retraining. Answers: are any of the current features actively hurting
the model, or just dead weight?

Two signals, printed side by side:
1. XGBoost's built-in gain-based importance -- how often/effectively a
   feature is used for splits. Quick, but can overstate high-cardinality
   or frequently-split features that don't actually help accuracy.
2. Permutation importance on the valid split -- shuffle each feature's
   values (breaking its relationship with the target, preserving its
   marginal distribution and NaN pattern), re-predict, measure the macro F1
   drop vs. the unshuffled baseline. Averaged over several shuffles per
   feature to reduce noise. This is the more reliable signal: it directly
   measures each feature's actual contribution to the eval metric, not
   just how often it's used.

Usage:
    python feature_importance_audit.py --sources aadf osm meteo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import (
    MODEL_DIR, feature_columns, load_split, prepare_xy, subset_name,
    unify_categorical_dtypes,
)

N_REPEATS = 5
SEED = 42


def permutation_importance(model, X: pd.DataFrame, y, baseline_f1: float, rng: np.random.RandomState) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(N_REPEATS):
            X_shuf = X.copy()
            X_shuf[col] = X_shuf[col].to_numpy()[rng.permutation(len(X_shuf))]
            y_pred = model.predict(X_shuf)
            shuf_f1 = f1_score(y, y_pred, average="macro")
            drops.append(baseline_f1 - shuf_f1)
        rows.append({"feature": col, "mean_f1_drop": np.mean(drops), "std_f1_drop": np.std(drops)})
    return pd.DataFrame(rows).sort_values("mean_f1_drop", ascending=False).reset_index(drop=True)


def missingness_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rates = df[cols].isna().mean().sort_values(ascending=False)
    return rates[rates > 0].rename("missing_rate").reset_index().rename(columns={"index": "feature"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", choices=["aadf", "osm", "meteo"],
                         default=["aadf", "osm", "meteo"])
    parser.add_argument("--data-variant", choices=["masked", "normalized", "raw"], default="masked")
    args = parser.parse_args()

    name = subset_name(args.sources)
    file_stem = name if args.data_variant == "masked" else f"{name}_{args.data_variant}"
    cols = feature_columns(args.sources, args.data_variant)

    model_path = MODEL_DIR / f"classifier_{file_stem}.json"
    print(f"Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    valid_df = load_split("valid", cols, args.data_variant)
    X_valid, y_valid = prepare_xy(valid_df, cols)
    unify_categorical_dtypes([X_valid], cols)

    baseline_f1 = f1_score(y_valid, model.predict(X_valid), average="macro")
    print(f"\nBaseline valid macro F1 (unshuffled): {baseline_f1:.4f}")

    print("\n=== Gain-based importance (XGBoost built-in) ===")
    gain = model.get_booster().get_score(importance_type="gain")
    gain_df = pd.DataFrame(sorted(gain.items(), key=lambda kv: -kv[1]), columns=["feature", "gain"])
    # Features that were never split on don't appear in get_score() at all.
    unused = sorted(set(cols) - set(gain_df["feature"]))
    print(gain_df.to_string(index=False))
    if unused:
        print(f"\nNever used in any split ({len(unused)}): {unused}")

    print(f"\n=== Permutation importance on valid ({N_REPEATS} shuffles/feature) ===")
    rng = np.random.RandomState(SEED)
    perm_df = permutation_importance(model, X_valid, y_valid, baseline_f1, rng)
    print(perm_df.to_string(index=False))

    near_zero = perm_df[perm_df["mean_f1_drop"] <= 0.0005]
    negative = perm_df[perm_df["mean_f1_drop"] < 0]
    print(f"\nNear-zero-or-negative importance ({len(near_zero)} features, "
          f"drop <= 0.0005): {near_zero['feature'].tolist()}")
    if len(negative) > 0:
        print(f"Negative importance ({len(negative)} features -- shuffling *helped* "
              f"macro F1, candidates for removal): {negative['feature'].tolist()}")

    print("\n=== Missingness rates for flagged low-importance features ===")
    flagged = near_zero["feature"].tolist()
    if flagged:
        print(missingness_report(valid_df, flagged).to_string(index=False))
    else:
        print("(none flagged)")


if __name__ == "__main__":
    main()
