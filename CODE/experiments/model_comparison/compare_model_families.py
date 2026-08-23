"""compare_model_families.py -- how much of the score is XGBoost, and how much is the data?

Trains a spread of model families on the SAME split, features and protocol as
the headline classifier (normalized, binary Slight-vs-Severe, full acquisition
= aadf+osm+meteo) and reports every family on the same metrics, over several
seeds. Includes deliberately simple baselines (majority-class, linear
regression, logistic regression, naive Bayes, a single decision tree, kNN) so
the gap between "trivial model" and "tuned gradient booster" is measurable
rather than assumed.

Feature handling per family, so nothing is handicapped by another family's
requirements:
  * XGBoost / LightGBM / CatBoost / HistGradientBoosting -- native categorical
    handling and native NaN handling, no imputation.
  * everything else -- median-impute + z-score numerics, most-frequent-impute +
    one-hot categoricals (a scikit-learn ColumnTransformer).
Class balance is handled the way the headline model does it: balanced sample
weights / class_weight where the family supports it.

Outputs (all under docs/figures/model_families/):
  model_families_results.csv   one row per (model, seed)
  model_families_summary.csv   per-model means and standard deviations
  fig1_macro_f1.png            macro F1 by family, seed error bars
  fig2_roc.png                 ROC curves
  fig3_pr.png                  precision-recall curves
  fig4_time_vs_score.png       training cost vs macro F1
  fig5_metric_panels.png       macro F1 / balanced accuracy / AUC, same order
  fig6_confusion.png           confusion matrices for the four best families

Usage:
  python CODE/experiments/model_comparison/compare_model_families.py [--seeds 42 43 44] [--quick]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from train_classifier import (CATEGORICAL_COLS, MODEL_DIR, feature_columns, load_split,
                              prepare_xy, unify_categorical_dtypes)

import json

VARIANT = "normalized"
SOURCES = ["aadf", "osm", "meteo"]
OUT_DIR = Path("docs/figures/model_families")
POS_LABEL = "Severe"   # y == 1


# ── data ──────────────────────────────────────────────────────────────────────
def load_data():
    cols = feature_columns(SOURCES, VARIANT)
    frames = {s: load_split(s, cols, VARIANT) for s in ("train", "valid", "test")}
    Xy = {s: prepare_xy(frames[s], cols, binary=True) for s in frames}
    cat_cols = [c for c in CATEGORICAL_COLS if c in cols]
    num_cols = [c for c in cols if c not in cat_cols]
    return Xy, cols, cat_cols, num_cols


def sklearn_preprocessor(num_cols, cat_cols):
    """Impute + scale numerics, impute + one-hot categoricals.

    Needed by every family that cannot eat NaNs or raw category codes; the
    gradient boosters below skip this entirely and use their own handling.
    """
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]),
         cat_cols),
    ])


class LinearRegressionClassifier:
    """Ordinary least squares on the 0/1 target, thresholded at 0.5.

    Not a sensible classifier -- it is here as the floor of the model ladder,
    to show what treating a classification problem as regression costs.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def decision_function(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return (self.model.predict(X) >= 0.5).astype(int)


# ── model zoo ─────────────────────────────────────────────────────────────────
def build_models(seed, num_cols, cat_cols, xgb_params, quick):
    """(name, family, estimator, needs_preprocessing, supports_sample_weight)."""
    pre = lambda est: Pipeline([("pre", sklearn_preprocessor(num_cols, cat_cols)), ("est", est)])
    models = [
        ("Majority baseline", "baseline",
         pre(DummyClassifier(strategy="most_frequent")), True, False),
        ("Random (stratified)", "baseline",
         pre(DummyClassifier(strategy="stratified", random_state=seed)), True, False),
        # Holte's OneR: the single best feature-threshold rule, chosen on train.
        # The standard "how far does one rule get you" reference point.
        ("One-rule (stump)", "baseline",
         pre(DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=seed)),
         True, False),
        ("Linear regression", "linear", pre(LinearRegressionClassifier()), True, True),
        ("Logistic regression", "linear",
         pre(LogisticRegression(max_iter=2000, class_weight="balanced")), True, False),
        ("Linear SVM", "linear",
         pre(LinearSVC(class_weight="balanced", dual="auto", max_iter=5000)), True, False),
        ("Gaussian naive Bayes", "simple", pre(GaussianNB()), True, True),
        ("Decision tree (depth 5)", "simple",
         pre(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed)),
         True, False),
        ("Decision tree (full)", "simple",
         pre(DecisionTreeClassifier(class_weight="balanced", random_state=seed)), True, False),
        ("k-NN (k=25)", "simple", pre(KNeighborsClassifier(n_neighbors=25, n_jobs=-1)), True, False),
        ("Random forest", "ensemble",
         pre(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    n_jobs=-1, random_state=seed)), True, False),
        ("Extra trees", "ensemble",
         pre(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)), True, False),
        ("Neural net (MLP)", "ensemble",
         pre(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                           early_stopping=True, random_state=seed)), True, False),
        ("HistGradientBoosting", "boosting",
         pre(HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
         True, False),
        ("LightGBM", "boosting",
         lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, class_weight="balanced",
                            random_state=seed, n_jobs=-1, verbose=-1), False, False),
        ("CatBoost", "boosting", "catboost", False, True),
        ("XGBoost", "boosting",
         xgb.XGBClassifier(n_estimators=1000, tree_method="hist", enable_categorical=True,
                           random_state=seed, n_jobs=-1, objective="binary:logistic",
                           eval_metric="logloss", **xgb_params), False, True),
    ]
    if quick:
        models = [m for m in models if m[0] in
                  ("Majority baseline", "One-rule (stump)", "Logistic regression",
                   "XGBoost")]
    return models


def scores_from(est, X):
    """Positive-class score, whatever the estimator exposes."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def run_one(name, est, needs_pre, takes_weight, Xy, cat_cols, seed):
    (Xtr, ytr), (Xte, yte) = Xy["train"], Xy["test"]
    w = compute_sample_weight("balanced", ytr)

    if est != "catboost" and not needs_pre:
        # LightGBM/XGBoost take pandas categoricals natively -- but only if the
        # columns actually carry a shared CategoricalDtype. Without this they
        # arrive as plain numbers and both models score ~0.002 lower than the
        # project's own metrics for the identical configuration.
        Xtr, Xte = Xtr.copy(), Xte.copy()
        unify_categorical_dtypes([Xtr, Xte], list(Xtr.columns))

    if est == "catboost":
        # CatBoost wants categoricals as strings with no NaN.
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype(str).fillna("missing")
            Xte[c] = Xte[c].astype(str).fillna("missing")
        # allow_writing_files=False: CatBoost otherwise dumps training logs into a
        # tracked catboost_info/ directory in the repo root on every fit.
        est = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                                 random_seed=seed, verbose=0, cat_features=cat_cols,
                                 allow_writing_files=False)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if takes_weight:
            fit_kw = {"est__sample_weight": w} if needs_pre else {"sample_weight": w}
            est.fit(Xtr, ytr, **fit_kw)
        else:
            est.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        y_pred = est.predict(Xte)
        y_score = scores_from(est, Xte)

    return {
        "model": name, "seed": seed, "fit_seconds": fit_s,
        "macro_f1": f1_score(yte, y_pred, average="macro"),
        "accuracy": accuracy_score(yte, y_pred),
        "balanced_accuracy": balanced_accuracy_score(yte, y_pred),
        "severe_recall": recall_score(yte, y_pred, pos_label=1),
        "auc": roc_auc_score(yte, y_score),
        "pr_auc": average_precision_score(yte, y_score),
    }, y_score, y_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true", help="4 models only, smoke test")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xy, cols, cat_cols, num_cols = load_data()
    print(f"features={len(cols)} ({len(cat_cols)} categorical)  "
          f"train={len(Xy['train'][0])} test={len(Xy['test'][0])}  "
          f"test Severe rate={Xy['test'][1].mean():.3f}")

    xgb_params = json.load(open(MODEL_DIR / "best_hyperparams_normalized_binary.json"))["params"]

    rows, curves, confusions = [], {}, {}
    for seed in args.seeds:
        for name, family, est, needs_pre, takes_w in build_models(
                seed, num_cols, cat_cols, xgb_params, args.quick):
            rec, y_score, y_pred = run_one(name, est, needs_pre, takes_w, Xy, cat_cols, seed)
            rec["family"] = family
            rows.append(rec)
            if seed == args.seeds[0]:               # curves from the first seed only
                curves[name] = y_score
                confusions[name] = confusion_matrix(Xy["test"][1], y_pred)
            print(f"  [seed {seed}] {name:24s} macroF1={rec['macro_f1']:.4f} "
                  f"auc={rec['auc']:.4f} {rec['fit_seconds']:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "model_families_results.csv", index=False)
    summary = (df.groupby(["model", "family"])
                 .agg(macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                      accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
                      severe_recall=("severe_recall", "mean"), auc=("auc", "mean"),
                      auc_sd=("auc", "std"), pr_auc=("pr_auc", "mean"),
                      fit_seconds=("fit_seconds", "mean"))
                 .reset_index().sort_values("macro_f1", ascending=False))
    summary.to_csv(OUT_DIR / "model_families_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    from plot_model_families import make_figures      # noqa: E402  (same directory)
    make_figures(summary, df, curves, confusions, Xy["test"][1], OUT_DIR)
    print(f"\nfigures + tables -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
