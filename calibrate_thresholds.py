#!/usr/bin/env python3
"""calibrate_thresholds.py

Post-hoc decision calibration for an already-trained classifier (see
train_classifier.py) -- no retraining. XGBoost's argmax(predict_proba) is
optimal for 0-1 loss, but macro F1 weights all classes equally regardless
of frequency, so scaling each class's predicted probability by a per-class
weight before taking argmax (argmax(w_i * p_i)) can recover real macro F1
that plain argmax leaves on the table -- the weights are chosen on the
*validation* set (never on test) to avoid overfitting the calibration
itself. For --binary models this reduces to moving the decision threshold
on P(Severe) instead of a fixed 0.5.

Usage:
    python calibrate_thresholds.py --sources aadf osm meteo
    python calibrate_thresholds.py --sources aadf osm meteo --binary
"""

import argparse
import json

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from train_classifier import (
    BINARY_LABELS, MODEL_DIR, SEVERITY_LABELS, evaluate, feature_columns,
    load_split, prepare_xy, subset_name, unify_categorical_dtypes,
)


def weighted_predict(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(proba * weights, axis=1)


def evaluate_weighted(proba: np.ndarray, y, weights: np.ndarray, split: str, labels: dict) -> dict:
    y_pred = weighted_predict(proba, weights)
    macro_f1 = f1_score(y, y_pred, average="macro")
    target_names = [labels[i] for i in range(len(labels))]
    report = classification_report(
        y, y_pred, target_names=target_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred).tolist()
    print(f"\n[{split}, weighted] macro F1 = {macro_f1:.4f}")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
    print(f"[{split}, weighted] confusion matrix (rows=true, cols=pred, order={'/'.join(target_names)}):\n{np.array(cm)}")
    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", choices=["aadf", "osm", "meteo"],
                         default=["aadf", "osm", "meteo"])
    parser.add_argument("--data-variant", choices=["masked", "normalized", "raw"], default="masked")
    parser.add_argument("--binary", action="store_true",
                         help="Calibrate the Slight-vs-Severe binary model instead of the 3-class one.")
    args = parser.parse_args()

    labels = BINARY_LABELS if args.binary else SEVERITY_LABELS
    n_classes = len(labels)

    name = subset_name(args.sources)
    file_stem = name if args.data_variant == "masked" else f"{name}_{args.data_variant}"
    if args.binary:
        file_stem += "_binary"
    cols = feature_columns(args.sources, args.data_variant)

    model_path = MODEL_DIR / f"classifier_{file_stem}.json"
    print(f"Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    valid_df = load_split("valid", cols, args.data_variant)
    test_df = load_split("test", cols, args.data_variant)
    X_valid, y_valid = prepare_xy(valid_df, cols, args.binary)
    X_test, y_test = prepare_xy(test_df, cols, args.binary)
    unify_categorical_dtypes([X_valid, X_test], cols)

    print("Baseline (argmax, no calibration):")
    evaluate(model, X_valid, y_valid, "valid", labels)
    evaluate(model, X_test, y_test, "test", labels)

    proba_valid = model.predict_proba(X_valid)
    proba_test = model.predict_proba(X_test)

    def objective(w):
        w = np.abs(w)  # weights must be positive; sign is not meaningful
        return -f1_score(y_valid, weighted_predict(proba_valid, w), average="macro")

    result = minimize(objective, x0=np.ones(n_classes), method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 500})
    weights = np.abs(result.x)
    weights = weights / weights.sum() * n_classes  # normalize for readability (mean weight = 1)
    print(f"\nBest weights ({', '.join(labels[i] for i in range(n_classes))}): {weights}")
    print(f"Valid macro F1 at these weights: {-objective(weights):.4f}")

    valid_metrics = evaluate_weighted(proba_valid, y_valid, weights, "valid", labels)
    test_metrics = evaluate_weighted(proba_test, y_test, weights, "test", labels)

    out_path = MODEL_DIR / f"classifier_{file_stem}_thresholds.json"
    with open(out_path, "w") as f:
        json.dump({
            "subset": args.sources,
            "data_variant": args.data_variant,
            "binary": args.binary,
            "weights": {labels[i]: float(weights[i]) for i in range(n_classes)},
            "valid_macro_f1": valid_metrics["macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
        }, f, indent=2)
    print(f"\nWeights saved to {out_path}")


if __name__ == "__main__":
    main()
