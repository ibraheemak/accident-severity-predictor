#!/usr/bin/env python3
"""tune_classifier.py

Small random search over XGBoost hyperparameters for one source-subset
classifier (see train_classifier.py). Selection metric is macro F1 on the
valid split (matches the README's stated evaluation metric), not mlogloss --
early stopping still uses mlogloss internally since that's what XGBoost's
native early stopping supports, but the *winning config* is chosen by macro
F1 on valid.

Usage:
    python tune_classifier.py --sources aadf osm meteo --n-iter 10
"""

import argparse
import json
import random
import time
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from train_classifier import (
    MODEL_DIR, feature_columns, load_split, prepare_xy, subset_name, evaluate,
)

PARAM_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1, 5, 10],
}


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_SPACE.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", choices=["aadf", "osm", "meteo"],
                         default=["aadf", "osm", "meteo"])
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    name = subset_name(args.sources)
    cols = feature_columns(args.sources)
    print(f"Subset: {name}  ({len(cols)} feature columns)")

    train_df = load_split("train", cols)
    valid_df = load_split("valid", cols)
    test_df = load_split("test", cols)

    X_train, y_train = prepare_xy(train_df, cols)
    X_valid, y_valid = prepare_xy(valid_df, cols)
    X_test, y_test = prepare_xy(test_df, cols)
    sample_weight = compute_sample_weight("balanced", y_train)

    rng = random.Random(args.seed)
    results = []
    best = None

    for i in range(args.n_iter):
        params = sample_params(rng)
        t0 = time.time()
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=args.n_estimators,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.seed,
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        y_pred_valid = model.predict(X_valid)
        macro_f1 = f1_score(y_valid, y_pred_valid, average="macro")
        elapsed = time.time() - t0
        print(f"[{i+1}/{args.n_iter}] {params} best_iter={model.best_iteration} "
              f"valid_macro_f1={macro_f1:.4f} ({elapsed:.0f}s)")
        results.append({"params": params, "best_iteration": int(model.best_iteration),
                         "valid_macro_f1": macro_f1})
        if best is None or macro_f1 > best["macro_f1"]:
            best = {"params": params, "model": model, "macro_f1": macro_f1}

    print(f"\nBest config: {best['params']}  valid_macro_f1={best['macro_f1']:.4f}")

    valid_metrics = evaluate(best["model"], X_valid, y_valid, "valid")
    test_metrics = evaluate(best["model"], X_test, y_test, "test")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"classifier_{name}.json"
    best["model"].save_model(model_path)

    meta = {
        "subset": args.sources,
        "feature_columns": cols,
        "best_iteration": int(best["model"].best_iteration),
        "params": best["params"],
        "valid": {"macro_f1": valid_metrics["macro_f1"]},
        "test": {"macro_f1": test_metrics["macro_f1"]},
        "search_results": results,
    }
    meta_path = MODEL_DIR / f"classifier_{name}_metrics.json"
    with open(meta_path, "w") as f:
        json.dump({**meta, "valid_full": valid_metrics, "test_full": test_metrics}, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {meta_path}")


if __name__ == "__main__":
    main()
