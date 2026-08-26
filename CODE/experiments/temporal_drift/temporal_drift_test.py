#!/usr/bin/env python3
"""temporal_drift_test.py -- does accident severity predictability drift over 7 years?

Trains on 2016-2017 STATS19 data and tests on 2024 to examine whether accident
severity patterns are stable over time. Compares performance against a control
model trained on mixed years.

Protocol: normalized variant, binary Slight-vs-Severe, balanced sample weights,
Optuna-tuned hyperparameters, early stopping on validation, scored on test.

Outputs:
  temporal_drift_results.csv     test metrics for 2016-2017→2024 vs. full-data control
  temporal_drift_by_year.csv     test metrics broken down by year-of-incident

Usage:
  python CODE/experiments/temporal_drift/temporal_drift_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))

import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from train_classifier import (
    BASE_COLS,
    CATEGORICAL_COLS,
    DATA_VARIANT_DIRS,
    DATA_VARIANT_FILE,
    MODEL_DIR,
    free_columns,
    prepare_xy,
    unify_categorical_dtypes,
)

VARIANT = "normalized"
OUT_DIR = Path("docs/figures/temporal_drift")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load hyperparameters (use best tuned params or defaults)
BEST_PARAMS_PATH = MODEL_DIR / "best_hyperparams_normalized_binary.json"
if BEST_PARAMS_PATH.exists():
    with open(BEST_PARAMS_PATH) as f:
        BEST_PARAMS = json.load(f)
else:
    BEST_PARAMS = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_child_weight": 1,
    }


def evaluate_model(model, X, y, prefix=""):
    """Evaluate model and return a dict of metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = {
        f"{prefix}Macro_F1": f1_score(y, y_pred, average="macro", zero_division=0),
        f"{prefix}Balanced_Acc": balanced_accuracy_score(y, y_pred),
        f"{prefix}ROC_AUC": roc_auc_score(y, y_pred_proba),
        f"{prefix}Recall_Severe": recall_score(y, y_pred, pos_label=1, zero_division=0),
        f"{prefix}Recall_Slight": recall_score(y, y_pred, pos_label=0, zero_division=0),
    }
    return metrics


def main():
    print("Loading data...")

    # Get feature columns
    feature_cols = list(BASE_COLS) + free_columns(VARIANT)

    # Load splits with year information
    splits = {}
    for split_name in ("train", "valid", "test"):
        path = DATA_VARIANT_DIRS[VARIANT] / DATA_VARIANT_FILE[VARIANT].format(name=split_name)
        df = pd.read_csv(path)
        df["split"] = split_name
        splits[split_name] = df

    # Unify categorical dtypes across all splits
    unify_categorical_dtypes([splits[s][feature_cols] for s in splits], feature_cols)

    all_data = pd.concat(list(splits.values()), ignore_index=True)

    print(f"Total records: {len(all_data)}")
    print(f"Year distribution:\n{all_data['collision_year'].value_counts().sort_index()}")

    # === Control: Train on all years (mixed) ===
    print("\n" + "=" * 70)
    print("CONTROL: Training on all years (2016-2024)...")
    print("=" * 70)

    train_all = all_data[all_data["split"] == "train"].copy()
    valid_all = all_data[all_data["split"] == "valid"].copy()
    test_all = all_data[all_data["split"] == "test"].copy()

    X_train_all, y_train_all = prepare_xy(train_all, feature_cols, binary=True)
    X_valid_all, y_valid_all = prepare_xy(valid_all, feature_cols, binary=True)
    X_test_all, y_test_all = prepare_xy(test_all, feature_cols, binary=True)

    sample_weights = compute_sample_weight("balanced", y_train_all)

    model_control = xgb.XGBClassifier(
        **BEST_PARAMS,
        early_stopping_rounds=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model_control.fit(
        X_train_all,
        y_train_all,
        eval_set=[(X_valid_all, y_valid_all)],
        sample_weight=sample_weights,
        verbose=False,
    )

    metrics_control = evaluate_model(model_control, X_test_all, y_test_all, prefix="Control_")
    print("Control model (all years) test metrics:")
    for k, v in metrics_control.items():
        print(f"  {k}: {v:.4f}")

    # === Temporal drift: Train on 2016-2017 only, test on 2024 ===
    print("\n" + "=" * 70)
    print("TEMPORAL DRIFT: Training on 2016-2017, testing on 2024...")
    print("=" * 70)

    train_early = all_data[(all_data["collision_year"] >= 2016) &
                           (all_data["collision_year"] <= 2017) &
                           (all_data["split"] == "train")].copy()
    valid_early = all_data[(all_data["collision_year"] >= 2016) &
                           (all_data["collision_year"] <= 2017) &
                           (all_data["split"] == "valid")].copy()
    test_late = all_data[(all_data["collision_year"] == 2024) &
                         (all_data["split"] == "test")].copy()

    print(f"Early training set (2016-2017): {len(train_early)} records")
    print(f"  Year breakdown:\n{train_early['collision_year'].value_counts().sort_index()}")
    print(f"Early validation set (2016-2017): {len(valid_early)} records")
    print(f"Late test set (2024): {len(test_late)} records")

    X_train_early, y_train_early = prepare_xy(train_early, feature_cols, binary=True)
    X_valid_early, y_valid_early = prepare_xy(valid_early, feature_cols, binary=True)
    X_test_late, y_test_late = prepare_xy(test_late, feature_cols, binary=True)

    sample_weights_early = compute_sample_weight("balanced", y_train_early)

    model_drift = xgb.XGBClassifier(
        **BEST_PARAMS,
        early_stopping_rounds=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model_drift.fit(
        X_train_early,
        y_train_early,
        eval_set=[(X_valid_early, y_valid_early)],
        sample_weight=sample_weights_early,
        verbose=False,
    )

    metrics_drift = evaluate_model(model_drift, X_test_late, y_test_late, prefix="Drift_")
    print("Drift model (2016-2017 → 2024) test metrics:")
    for k, v in metrics_drift.items():
        print(f"  {k}: {v:.4f}")

    # === Compare deltas ===
    print("\n" + "=" * 70)
    print("DELTA (Drift - Control)")
    print("=" * 70)

    # For fair comparison, also evaluate control model on 2024 test
    metrics_control_2024 = evaluate_model(model_control, X_test_late, y_test_late, prefix="Control_2024_")

    for key in ["Macro_F1", "Balanced_Acc", "ROC_AUC", "Recall_Severe", "Recall_Slight"]:
        drift_val = metrics_drift.get(f"Drift_{key}", None)
        control_val = metrics_control_2024.get(f"Control_2024_{key}", None)
        if drift_val is not None and control_val is not None:
            delta = drift_val - control_val
            print(f"  {key}: Drift={drift_val:.4f}, Control(2024)={control_val:.4f}, Delta={delta:+.4f}")

    # === Save results ===
    results = {
        **metrics_control,
        **metrics_drift,
        **metrics_control_2024,
    }

    results_df = pd.DataFrame([results])
    results_csv = OUT_DIR / "temporal_drift_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")

    # === Analyze performance by year ===
    print("\n" + "=" * 70)
    print("PERFORMANCE BY YEAR (Control model on test split)")
    print("=" * 70)

    by_year_rows = []
    for year in sorted(test_all["collision_year"].unique()):
        test_year = test_all[test_all["collision_year"] == year].copy()
        X_year, y_year = prepare_xy(test_year, feature_cols, binary=True)

        year_metrics = evaluate_model(model_control, X_year, y_year, prefix="")
        year_metrics["year"] = year
        year_metrics["n_samples"] = len(test_year)

        by_year_rows.append(year_metrics)
        print(f"  {year} (n={len(test_year):4d}): Macro F1={year_metrics['Macro_F1']:.4f}, "
              f"Balanced Acc={year_metrics['Balanced_Acc']:.4f}")

    by_year_df = pd.DataFrame(by_year_rows)
    by_year_csv = OUT_DIR / "temporal_drift_by_year.csv"
    by_year_df.to_csv(by_year_csv, index=False)
    print(f"\nPer-year results saved to {by_year_csv}")


if __name__ == "__main__":
    main()
