#!/usr/bin/env python3
"""plot_temporal_drift.py -- visualize temporal drift results

Generates plots from the temporal drift experiment:
  - Performance degradation from 2016-2017 to 2024
  - Trend line showing how model performance changes across years

Usage:
  python CODE/experiments/temporal_drift/plot_temporal_drift.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("docs/figures/temporal_drift")


def plot_by_year_trend():
    """Plot drift model vs control model performance across years."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))

    from train_classifier import (
        BASE_COLS, DATA_VARIANT_DIRS, DATA_VARIANT_FILE,
        free_columns, prepare_xy, unify_categorical_dtypes,
    )
    import xgboost as xgb
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.utils.class_weight import compute_sample_weight
    import json

    VARIANT = "normalized"
    MODEL_DIR = Path("models")

    # Load hyperparameters
    BEST_PARAMS_PATH = MODEL_DIR / "best_hyperparams_normalized_binary.json"
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            BEST_PARAMS = json.load(f)
    else:
        BEST_PARAMS = {
            "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        }

    # Load data
    feature_cols = list(BASE_COLS) + free_columns(VARIANT)
    splits = {}
    for split_name in ("train", "valid", "test"):
        path = DATA_VARIANT_DIRS[VARIANT] / DATA_VARIANT_FILE[VARIANT].format(name=split_name)
        df = pd.read_csv(path)
        df["split"] = split_name
        splits[split_name] = df

    unify_categorical_dtypes([splits[s][feature_cols] for s in splits], feature_cols)
    all_data = pd.concat(list(splits.values()), ignore_index=True)

    # Train drift model (2016-2017)
    train_early = all_data[(all_data["collision_year"] >= 2016) & (all_data["collision_year"] <= 2017) & (all_data["split"] == "train")].copy()
    valid_early = all_data[(all_data["collision_year"] >= 2016) & (all_data["collision_year"] <= 2017) & (all_data["split"] == "valid")].copy()

    X_train_early, y_train_early = prepare_xy(train_early, feature_cols, binary=True)
    X_valid_early, y_valid_early = prepare_xy(valid_early, feature_cols, binary=True)

    sample_weights = compute_sample_weight("balanced", y_train_early)

    model_drift = xgb.XGBClassifier(**BEST_PARAMS, early_stopping_rounds=10, eval_metric="logloss", random_state=42, n_jobs=-1)
    model_drift.fit(X_train_early, y_train_early, eval_set=[(X_valid_early, y_valid_early)], sample_weight=sample_weights, verbose=False)

    # Train control model (all years)
    train_all = all_data[all_data["split"] == "train"].copy()
    valid_all = all_data[all_data["split"] == "valid"].copy()

    X_train_all, y_train_all = prepare_xy(train_all, feature_cols, binary=True)
    X_valid_all, y_valid_all = prepare_xy(valid_all, feature_cols, binary=True)

    sample_weights_all = compute_sample_weight("balanced", y_train_all)

    model_control = xgb.XGBClassifier(**BEST_PARAMS, early_stopping_rounds=10, eval_metric="logloss", random_state=42, n_jobs=-1)
    model_control.fit(X_train_all, y_train_all, eval_set=[(X_valid_all, y_valid_all)], sample_weight=sample_weights_all, verbose=False)

    # Evaluate both models on test data by year
    test_all = all_data[all_data["split"] == "test"].copy()

    results = []
    for year in sorted(test_all["collision_year"].unique()):
        test_year = test_all[test_all["collision_year"] == year].copy()
        X_year, y_year = prepare_xy(test_year, feature_cols, binary=True)

        # Drift model predictions
        drift_pred = model_drift.predict(X_year)
        drift_proba = model_drift.predict_proba(X_year)[:, 1]

        # Control model predictions
        control_pred = model_control.predict(X_year)
        control_proba = model_control.predict_proba(X_year)[:, 1]

        results.append({
            "year": year,
            "n_samples": len(test_year),
            "drift_f1": f1_score(y_year, drift_pred, average="macro", zero_division=0),
            "drift_balanced_acc": balanced_accuracy_score(y_year, drift_pred),
            "drift_auc": roc_auc_score(y_year, drift_proba),
            "control_f1": f1_score(y_year, control_pred, average="macro", zero_division=0),
            "control_balanced_acc": balanced_accuracy_score(y_year, control_pred),
            "control_auc": roc_auc_score(y_year, control_proba),
        })

    results_df = pd.DataFrame(results)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [("Macro F1", "drift_f1", "control_f1"),
               ("Balanced Accuracy", "drift_balanced_acc", "control_balanced_acc"),
               ("ROC AUC", "drift_auc", "control_auc")]

    for idx, (metric_name, drift_col, control_col) in enumerate(metrics):
        ax = axes[idx]

        ax.plot(results_df["year"], results_df[drift_col], marker="o", linewidth=2.5, markersize=9,
                label="Drift Model (2016-17)", color="#d62728", alpha=0.8)
        ax.plot(results_df["year"], results_df[control_col], marker="s", linewidth=2.5, markersize=9,
                label="Control Model (all years)", color="#1f77b4", alpha=0.8)

        ax.set_xlabel("Test Year", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(results_df["year"].unique())

    fig.suptitle("Temporal Drift: 2016-2017 Model vs. Mixed-Year Control",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_by_year_trend.png", dpi=150, bbox_inches="tight")
    print("Saved fig2_by_year_trend.png (updated: drift vs control comparison)")
    plt.close()


if __name__ == "__main__":
    print("Generating temporal drift visualization...")
    plot_by_year_trend()
    print("Done. Output in docs/figures/temporal_drift/")
