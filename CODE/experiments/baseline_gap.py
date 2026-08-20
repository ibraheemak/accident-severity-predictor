"""Does the linear-vs-booster gap stay small on harder benchmarks?

compare_model_families.py measures every family on one benchmark (normalized,
binary) and finds only ~0.02 macro F1 between ordinary least squares and a tuned
booster. This script asks whether that holds when the problem gets harder: the
same three models -- OLS-thresholded linear regression, logistic regression,
tuned XGBoost -- across four benchmarks crossing task (binary / 3-class) with
observability (normalized = fully observed, masked = 0-31 fields blanked).

Result (3 seeds, 2026-08-19): the gap roughly doubles under partial
observability -- +0.0199 on normalized/binary against +0.0297 on masked/binary,
with the AUC gap going +0.020 -> +0.037. The linear pipeline has to impute what
the booster routes natively, and masking is the regime a real dispatcher is in.

Usage:
  python CODE/experiments/baseline_gap.py        # writes baseline_gap_results.csv
"""
import sys, json, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "train"))
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from train_classifier import (CATEGORICAL_COLS, MODEL_DIR, feature_columns, load_split,
                              prepare_xy, unify_categorical_dtypes)

SOURCES, SEEDS = ["aadf", "osm", "meteo"], [42, 43, 44]
BENCHMARKS = [("normalized", True), ("normalized", False), ("masked", True), ("masked", False)]
params = json.load(open(MODEL_DIR / "best_hyperparams_normalized_binary.json"))["params"]
rows = []

for variant, binary in BENCHMARKS:
    cols = feature_columns(SOURCES, variant)
    frames = {s: load_split(s, cols, variant) for s in ("train", "valid", "test")}
    Xy = {s: prepare_xy(frames[s], cols, binary) for s in frames}
    cat = [c for c in CATEGORICAL_COLS if c in cols]
    num = [c for c in cols if c not in cat]
    (Xtr, ytr), (Xte, yte) = Xy["train"], Xy["test"]
    w = compute_sample_weight("balanced", ytr)
    nclass = len(np.unique(ytr))
    tag = f"{variant}/{'binary' if binary else '3-class'}"
    print(f"\n=== {tag}  train={len(Xtr)} test={len(Xte)} classes={nclass}", flush=True)

    pre = lambda est: Pipeline([("pre", ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
    ])), ("est", est)])

    for name in ("Linear regression", "Logistic regression", "XGBoost (tuned)"):
        if name == "Linear regression" and not binary:
            continue                      # OLS thresholding is binary-only
        for seed in SEEDS:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "XGBoost (tuned)":
                    Xtr2 = Xtr.copy(); Xte2 = Xte.copy()
                    unify_categorical_dtypes([Xtr2, Xte2], cols)
                    obj = ({"objective": "binary:logistic", "eval_metric": "logloss"} if binary
                           else {"objective": "multi:softprob", "num_class": nclass,
                                 "eval_metric": "mlogloss"})
                    m = xgb.XGBClassifier(n_estimators=1000, tree_method="hist",
                                          enable_categorical=True, random_state=seed,
                                          n_jobs=-1, **obj, **params)
                    m.fit(Xtr2, ytr, sample_weight=w)
                    pred, proba = m.predict(Xte2), m.predict_proba(Xte2)
                elif name == "Linear regression":
                    m = pre(LinearRegression()).fit(Xtr, ytr, est__sample_weight=w)
                    raw = m.predict(Xte)
                    pred, proba = (raw >= 0.5).astype(int), raw
                else:
                    m = pre(LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)
                    pred, proba = m.predict(Xte), m.predict_proba(Xte)
            auc = (roc_auc_score(yte, proba if proba.ndim == 1 else proba[:, 1]) if binary
                   else roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
            rec = dict(benchmark=tag, model=name, seed=seed,
                       macro_f1=f1_score(yte, pred, average="macro"), auc=auc,
                       secs=time.perf_counter() - t0)
            rows.append(rec)
            print(f"  {name:20s} seed={seed} macroF1={rec['macro_f1']:.4f} "
                  f"auc={auc:.4f} {rec['secs']:.0f}s", flush=True)

df = pd.DataFrame(rows)
out = Path("docs/figures/model_families/baseline_gap_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
g = df.groupby(["benchmark", "model"]).agg(macro_f1=("macro_f1", "mean"),
                                           sd=("macro_f1", "std"), auc=("auc", "mean")).reset_index()
print("\n=== means ===")
print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n=== gap: XGBoost minus each baseline ===")
for bench in g.benchmark.unique():
    sub = g[g.benchmark == bench].set_index("model")
    x = sub.loc["XGBoost (tuned)", "macro_f1"]
    for base in ("Linear regression", "Logistic regression"):
        if base in sub.index:
            print(f"  {bench:22s} vs {base:20s} +{x - sub.loc[base,'macro_f1']:.4f} macro F1")
