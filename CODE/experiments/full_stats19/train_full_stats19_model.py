#!/usr/bin/env python3
"""train_full_stats19_model.py

Trains and SAVES the full-STATS19 upper-bound model (the 84-feature set from
build_full_stats19_features.py) to models/, so the artifact sits alongside the
dispatch-legit deployment classifiers. This is a REFERENCE / upper-bound model,
NOT a deployment candidate: it uses post-collision / forensic STATS19 fields no
999 dispatcher has at call time, and police_attended is leakage-adjacent
(attendance is itself driven by apparent severity). The saved metrics JSON
carries an explicit "experiment" warning field to that effect.

Uses the dispatch-tuned hyperparameters (models/best_hyperparams_normalized_binary.json)
-- a dedicated 80-trial tune for this feature set lost the head-to-head, see
tune_full_stats19_model.py. Reads data/splits/full_stats19_v2/{train,valid,
test}_full_v2.csv (build them first with build_full_stats19_features.py).

Note: models/ is gitignored repo-wide, so the saved artifact is local-only,
like every other classifier here.

Usage:
    python CODE/experiments/full_stats19/build_full_stats19_features.py --out-dir data/splits/full_stats19_v2  # (see its docstring for the v2 field set)
    python CODE/experiments/full_stats19/train_full_stats19_model.py
"""

import argparse
import json
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "training" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from train_classifier import CATEGORICAL_COLS, MODEL_DIR, TARGET_COL, feature_columns
from build_full_stats19_features import CATEGORICAL_EXTRA
from tune_full_stats19_model import FULL_EXTRA

FILE_STEM = "classifier_aadf_meteo_osm_normalized_binary_fullstats19"
EXPERIMENT_NOTE = ("full-STATS19 upper bound (NOT a deployment model -- uses "
                   "post-collision/forensic fields no 999 dispatcher has; "
                   "police_attended is leakage-adjacent). See "
                   "CODE/experiments/full_stats19/full_stats19_upper_bound.py.")


def load_frames(full_dir: Path, feats, cat_cols):
    fr = {s: pd.read_csv(full_dir / f"{s}_full_v2.csv") for s in ("train", "valid", "test")}
    X, y, Xl = {}, {}, []
    for s in ("train", "valid", "test"):
        X[s] = fr[s][feats].copy()
        y[s] = (fr[s][TARGET_COL].astype(int) != 3).astype(int)
        Xl.append(X[s])
    for col in cat_cols:
        cats = sorted(pd.concat([f[col] for f in Xl]).dropna().unique())
        for f in Xl:
            f[col] = f[col].astype(pd.CategoricalDtype(categories=cats))
    return X, y


def eval_split(model, X, y):
    pred = model.predict(X)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "report": classification_report(y, pred, target_names=["Slight", "Severe"],
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-dir", default="data/splits/full_stats19_v2")
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feats = feature_columns(["aadf", "osm", "meteo"], "normalized") + FULL_EXTRA
    cat_cols = [c for c in CATEGORICAL_COLS + CATEGORICAL_EXTRA if c in feats]
    X, y = load_frames(Path(args.full_dir), feats, cat_cols)
    print(f"{len(feats)} features  rows: train={len(X['train'])} valid={len(X['valid'])} test={len(X['test'])}")

    with open(MODEL_DIR / "best_hyperparams_normalized_binary.json") as f:
        params = json.load(f)["params"]

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators, tree_method="hist", enable_categorical=True,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=args.early_stopping_rounds, random_state=args.seed,
        n_jobs=-1, **params,
    )
    model.fit(X["train"], y["train"], sample_weight=compute_sample_weight("balanced", y["train"]),
              eval_set=[(X["valid"], y["valid"])], verbose=False)
    print(f"Best iteration: {model.best_iteration}")

    valid_metrics = eval_split(model, X["valid"], y["valid"])
    test_metrics = eval_split(model, X["test"], y["test"])
    print(f"valid: macro F1={valid_metrics['macro_f1']:.4f}  acc={valid_metrics['accuracy']:.4f}")
    print(f"test : macro F1={test_metrics['macro_f1']:.4f}  acc={test_metrics['accuracy']:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / f"{FILE_STEM}.json")
    meta = {
        "experiment": EXPERIMENT_NOTE,
        "data_variant": "normalized", "binary": True,
        "n_features": len(feats), "feature_columns": feats,
        "categorical_columns": cat_cols, "best_iteration": int(model.best_iteration),
        "params": {"n_estimators": args.n_estimators, "seed": args.seed, **params},
        "valid": {k: valid_metrics[k] for k in ("macro_f1", "accuracy", "balanced_accuracy")},
        "test": {k: test_metrics[k] for k in ("macro_f1", "accuracy", "balanced_accuracy")},
    }
    with open(MODEL_DIR / f"{FILE_STEM}_metrics.json", "w") as f:
        json.dump({**meta, "valid_full": valid_metrics, "test_full": test_metrics}, f, indent=2)

    print(f"\nModel saved to   models/{FILE_STEM}.json")
    print(f"Metrics saved to models/{FILE_STEM}_metrics.json")


if __name__ == "__main__":
    main()
