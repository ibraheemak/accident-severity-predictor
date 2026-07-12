#!/usr/bin/env python3
"""compare_algorithms.py

Compares Random Forest, KNN, and AdaBoost against the XGBoost baseline on
the Full-Acquisition subset. Runs on the "normalized" data variant (single
row per collision, no deliberate masking, full information) rather than
"masked": Random Forest/KNN/AdaBoost don't natively handle the deliberately
-masked NaNs the way XGBoost does, and KNN prediction cost scales with
training-set size, so the much smaller unmasked variant keeps this
tractable and keeps the comparison apples-to-apples (same imputation
applied uniformly to every model here).

class_weight="balanced" is used where the model supports it (Random
Forest); KNN and AdaBoost have no direct equivalent, so their results
partly reflect that structural difference, not just raw algorithm quality.

Usage:
    python compare_algorithms.py --sources aadf osm meteo
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import (
    SEVERITY_LABELS, evaluate, feature_columns, load_split, prepare_xy,
    subset_name, unify_categorical_dtypes,
)

VARIANT = "normalized"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", choices=["aadf", "osm", "meteo"],
                         default=["aadf", "osm", "meteo"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    name = subset_name(args.sources)
    cols = feature_columns(args.sources, VARIANT)
    print(f"Subset: {name}  data_variant={VARIANT}  ({len(cols)} feature columns)")

    train_df = load_split("train", cols, VARIANT)
    valid_df = load_split("valid", cols, VARIANT)
    test_df = load_split("test", cols, VARIANT)
    print(f"Rows: train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    X_train, y_train = prepare_xy(train_df, cols)
    X_valid, y_valid = prepare_xy(valid_df, cols)
    X_test, y_test = prepare_xy(test_df, cols)
    unify_categorical_dtypes([X_train, X_valid, X_test], cols)

    # None of these models take pandas `category` dtype directly -- convert
    # to plain numeric codes (same integer values, just not a special dtype).
    for df in (X_train, X_valid, X_test):
        for col in df.columns:
            if str(df[col].dtype) == "category":
                df[col] = df[col].cat.codes.replace(-1, np.nan).astype(float)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=args.seed,
        ),
        "KNN": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        ),
        "AdaBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=200, random_state=args.seed,
            ),
        ),
    }

    results = {}
    for model_name, model in models.items():
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        print(f"Fit time: {fit_time:.1f}s")

        valid_metrics = evaluate(model, X_valid, y_valid, "valid")
        test_metrics = evaluate(model, X_test, y_test, "test")
        results[model_name] = {
            "valid_macro_f1": valid_metrics["macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "fit_time_s": fit_time,
        }

    print(f"\n{'=' * 60}\nSummary (data_variant={VARIANT})\n{'=' * 60}")
    print("XGBoost (recorded earlier, same variant): test macro F1 = 0.4635")
    for model_name, r in results.items():
        print(f"{model_name:15s} valid={r['valid_macro_f1']:.4f}  "
              f"test={r['test_macro_f1']:.4f}  fit={r['fit_time_s']:.1f}s")


if __name__ == "__main__":
    main()
