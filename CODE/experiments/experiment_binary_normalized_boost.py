#!/usr/bin/env python3
"""experiment_binary_normalized_boost.py

Best-effort accuracy push for the binary (Slight-vs-Severe) classifier on the
normalized (unmasked, full-information) variant. Combines every remaining
legitimate lever in one place and reports the honest delta vs the current
tuned+calibrated baseline (test macro F1 0.6984):

  1. Extra features
       - max_engine_cc         (per-collision max engine size; weak but real
                                 signal -- severe accidents ~17% bigger engines)
       - a few continuous x continuous interaction terms XGBoost can only
         approximate through successive splits
  2. Model diversity + stacking
       - XGBoost (tuned normalized-binary params)
       - sklearn HistGradientBoosting (LightGBM-class, native NaN handling)
       - ExtraTrees (decorrelated bagging)
       - LogisticRegression (linear, very different bias)
       a soft-vote average AND a logistic stacker fit on validation
  3. Threshold calibration: pick the decision threshold maximizing macro F1
     on validation for every model, apply unchanged to test.

Everything is computed in-process against the existing normalized splits --
no pipeline re-run. Nothing is written to models/ (pure experiment).

Usage:
    python experiment_binary_normalized_boost.py

Result (2026-07-11): the ceiling holds. Best config was a 4-model soft-vote
at test macro F1 0.7006 -- only +0.0022 over the tuned+calibrated single-XGB
baseline (0.6984). The engine feature + interaction terms slightly *hurt*
XGBoost on their own (0.6984 -> 0.6978); the marginal gain came entirely from
model diversity, not new signal. All four model families (XGBoost, HistGB,
ExtraTrees, LogisticRegression) converge to ~0.774 test AUC -- a hard
information ceiling for this pre-outcome feature set, not a modelling
limitation. A +0.05 macro-F1 jump is not attainable from these features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import (
    CATEGORICAL_COLS, MODEL_DIR, feature_columns, load_split, prepare_xy,
    unify_categorical_dtypes,
)
import json

SOURCES = ["aadf", "osm", "meteo"]
VARIANT = "normalized"
VEHICLE_CSV = "data/raw/stats19/stats19_20260706_193034/codes/stats19_20260706_193034_vehicles.csv"

INTERACTIONS = [
    ("speed_limit", "number_of_vehicles"),
    ("speed_limit", "aadf_traffic_volume"),
    ("hist_acc_rate_500m", "speed_limit"),
    ("n_casualties", "speed_limit"),
]


def engine_feature() -> pd.DataFrame:
    v = pd.read_csv(VEHICLE_CSV, usecols=["collision_index", "engine_capacity_cc"])
    v.loc[v["engine_capacity_cc"] < 0, "engine_capacity_cc"] = np.nan
    return (v.groupby("collision_index")["engine_capacity_cc"].max()
            .rename("max_engine_cc").reset_index())


def add_extra_features(df: pd.DataFrame, cols: list[str], eng: pd.DataFrame,
                       eng_mean: float, eng_std: float) -> list[str]:
    """Add max_engine_cc (standardized) + interaction terms, in place. Returns
    the new column names."""
    df = df  # in place
    merged = df.merge(eng, on="collision_index", how="left")
    df["max_engine_cc"] = ((merged["max_engine_cc"].to_numpy() - eng_mean) / eng_std)
    new_cols = ["max_engine_cc"]
    for a, b in INTERACTIONS:
        name = f"{a}__x__{b}"
        df[name] = df[a].to_numpy() * df[b].to_numpy()
        new_cols.append(name)
    return new_cols


def best_threshold(y_valid, p_valid) -> float:
    ts = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_valid, (p_valid >= t).astype(int), average="macro") for t in ts]
    return float(ts[int(np.argmax(f1s))])


def report(name, y_valid, p_valid, y_test, p_test) -> float:
    t = best_threshold(y_valid, p_valid)
    mf1 = f1_score(y_test, (p_test >= t).astype(int), average="macro")
    auc = roc_auc_score(y_test, p_test)
    print(f"  {name:<28} thr={t:.3f}  test macro F1={mf1:.4f}  AUC={auc:.4f}")
    return mf1


def main():
    cols = feature_columns(SOURCES, VARIANT)
    # collision_index needed for the engine join; load it alongside features.
    train = load_split("train", cols + ["collision_index"], VARIANT)
    valid = load_split("valid", cols + ["collision_index"], VARIANT)
    test = load_split("test", cols + ["collision_index"], VARIANT)

    eng = engine_feature()
    eng_train = train.merge(eng, on="collision_index", how="left")["max_engine_cc"]
    eng_mean, eng_std = float(eng_train.mean()), float(eng_train.std())

    new_cols = []
    for df in (train, valid, test):
        new_cols = add_extra_features(df, cols, eng, eng_mean, eng_std)
    all_cols = cols + new_cols
    print(f"Features: {len(cols)} base + {len(new_cols)} new ({new_cols}) = {len(all_cols)}")

    y_train = (train["collision_severity"].astype(int) != 3).astype(int).to_numpy()
    y_valid = (valid["collision_severity"].astype(int) != 3).astype(int).to_numpy()
    y_test = (test["collision_severity"].astype(int) != 3).astype(int).to_numpy()

    # --- XGBoost (native categorical), tuned normalized-binary params ---
    Xx_train = train[all_cols].copy()
    Xx_valid = valid[all_cols].copy()
    Xx_test = test[all_cols].copy()
    cat_present = unify_categorical_dtypes([Xx_train, Xx_valid, Xx_test], all_cols)
    with open(MODEL_DIR / "best_hyperparams_normalized_binary.json") as f:
        xgb_params = json.load(f)["params"]
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_train)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, tree_method="hist", enable_categorical=True,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=30, random_state=42, n_jobs=-1, **xgb_params,
    )
    xgb_model.fit(Xx_train, y_train, sample_weight=sw,
                  eval_set=[(Xx_valid, y_valid)], verbose=False)
    p_xgb_valid = xgb_model.predict_proba(Xx_valid)[:, 1]
    p_xgb_test = xgb_model.predict_proba(Xx_test)[:, 1]

    # --- Numeric matrix for the sklearn models (categorical codes as numeric,
    #     NaN -> handled by HistGB natively; imputed to 0 for ExtraTrees/Logit) ---
    def numeric(df):
        M = df[all_cols].apply(pd.to_numeric, errors="coerce")
        return M
    Xn_train, Xn_valid, Xn_test = numeric(train), numeric(valid), numeric(test)
    Xi_train = Xn_train.fillna(0.0)
    Xi_valid = Xn_valid.fillna(0.0)
    Xi_test = Xn_test.fillna(0.0)

    cat_idx = [all_cols.index(c) for c in CATEGORICAL_COLS if c in all_cols]

    hgb = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=63,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.1,
    )
    hgb.fit(Xn_train, y_train)
    p_hgb_valid = hgb.predict_proba(Xn_valid)[:, 1]
    p_hgb_test = hgb.predict_proba(Xn_test)[:, 1]

    et = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, class_weight="balanced",
                              random_state=42)
    et.fit(Xi_train, y_train)
    p_et_valid = et.predict_proba(Xi_valid)[:, 1]
    p_et_test = et.predict_proba(Xi_test)[:, 1]

    logit = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    logit.fit(Xi_train, y_train)
    p_lr_valid = logit.predict_proba(Xi_valid)[:, 1]
    p_lr_test = logit.predict_proba(Xi_test)[:, 1]

    print("\n=== Individual models (threshold-calibrated on valid) ===")
    print(f"  {'BASELINE (tuned+calib XGB)':<28} (reference)          test macro F1=0.6984")
    report("XGBoost (+new feats)", y_valid, p_xgb_valid, y_test, p_xgb_test)
    report("HistGradientBoosting", y_valid, p_hgb_valid, y_test, p_hgb_test)
    report("ExtraTrees", y_valid, p_et_valid, y_test, p_et_test)
    report("LogisticRegression", y_valid, p_lr_valid, y_test, p_lr_test)

    # --- Ensembles ---
    print("\n=== Ensembles ===")
    # soft-vote average of the three tree/boosting models (drop weak logit)
    p_avg_valid = np.mean([p_xgb_valid, p_hgb_valid, p_et_valid], axis=0)
    p_avg_test = np.mean([p_xgb_test, p_hgb_test, p_et_test], axis=0)
    report("Soft-vote (XGB+HGB+ET)", y_valid, p_avg_valid, y_test, p_avg_test)

    p_avg4_valid = np.mean([p_xgb_valid, p_hgb_valid, p_et_valid, p_lr_valid], axis=0)
    p_avg4_test = np.mean([p_xgb_test, p_hgb_test, p_et_test, p_lr_test], axis=0)
    report("Soft-vote (all 4)", y_valid, p_avg4_valid, y_test, p_avg4_test)

    # logistic stacker fit on validation out-of-fold-ish predictions
    S_valid = np.column_stack([p_xgb_valid, p_hgb_valid, p_et_valid, p_lr_valid])
    S_test = np.column_stack([p_xgb_test, p_hgb_test, p_et_test, p_lr_test])
    stacker = LogisticRegression(max_iter=2000)
    stacker.fit(S_valid, y_valid)
    p_stk_valid = stacker.predict_proba(S_valid)[:, 1]
    p_stk_test = stacker.predict_proba(S_test)[:, 1]
    report("Logistic stacker", y_valid, p_stk_valid, y_test, p_stk_test)
    print(f"  stacker weights (xgb,hgb,et,lr): {stacker.coef_[0].round(3)}")


if __name__ == "__main__":
    main()
