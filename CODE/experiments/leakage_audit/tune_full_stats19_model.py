#!/usr/bin/env python3
"""tune_full_stats19_model.py

Hyperparameter search dedicated to the full-STATS19 upper-bound model (the
84-feature set built by build_full_stats19_features.py), which otherwise
borrows the dispatch model's hyperparameters. 80-trial Optuna with 4-fold CV
on pooled train+valid, followed by a multi-seed head-to-head on the held-out
test set against the incumbent (dispatch-tuned) params.

Result (2026-07-20, 1400/2800-cap config): the dedicated tune LOST -- CV
0.7589, and on test the challenger scored 0.7610 vs the dispatch-tuned
incumbent's 0.7625 (3-seed means). This was the 5th independent search on
this project (across dispatch + full feature sets, 3 dataset configs) and the
dispatch-era params (max_depth 8, lr 0.0133, gamma 2.72) won every one --
the hyperparameter surface is flat; retuning is not a lever here. Kept as a
reproducible record, not because it improves anything.

Requires data/splits/full_stats19_v2/{train,valid,test}_full_v2.csv (build
them first with build_full_stats19_features.py --out-dir, or point --full-dir
elsewhere). Run from repo root.

Usage:
    python CODE/experiments/leakage_audit/full_stats19_upper_bound.py   # (builds nothing; see its docstring)
    python CODE/experiments/leakage_audit/tune_full_stats19_model.py --n-trials 80
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "training" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import CATEGORICAL_COLS, MODEL_DIR, TARGET_COL, feature_columns
from build_full_stats19_features import CATEGORICAL_EXTRA

optuna.logging.set_verbosity(optuna.logging.WARNING)

# The 84-feature "best combo" set: dispatch cols + full-STATS19 extras minus
# the two ablation-rejected high-cardinality/redundant fields
# (generic_make_model, local_authority_highway -- see build script).
FULL_EXTRA = [
    "police_force", "local_authority_district", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards", "police_attended",
    "any_skidding", "any_hit_object_in", "any_left_carriageway", "any_hit_object_off",
    "n_front_impact", "n_back_impact", "dominant_manoeuvre", "dominant_junction_location",
    "any_restricted_lane", "any_work_journey", "youngest_driver_age", "oldest_driver_age",
    "n_male_drivers", "max_engine_cc", "max_vehicle_age", "mean_driver_imd",
    "dominant_propulsion", "min_driver_distance_band", "any_car_passenger",
    "any_bus_passenger", "any_road_worker", "mean_casualty_imd",
    "dominant_casualty_distance_band", "dominant_ped_location", "dominant_ped_movement",
]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-dir", default="data/splits/full_stats19_v2")
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feats = feature_columns(["aadf", "osm", "meteo"], "normalized") + FULL_EXTRA
    cat_cols = [c for c in CATEGORICAL_COLS + CATEGORICAL_EXTRA if c in feats]
    X, y = load_frames(Path(args.full_dir), feats, cat_cols)

    Xpool = pd.concat([X["train"], X["valid"]], ignore_index=True)
    ypool = pd.concat([y["train"], y["valid"]], ignore_index=True)
    folds = list(KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed).split(Xpool))
    print(f"{len(feats)} features, pool={len(Xpool)} rows, {args.n_splits}-fold CV, {args.n_trials} trials")

    def objective(trial):
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
             "max_depth": trial.suggest_int("max_depth", 4, 14),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 40, log=True),
             "gamma": trial.suggest_float("gamma", 0.0, 8.0),
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True)}
        sc = []
        for i, (tr, va) in enumerate(folds):
            Xt, Xv = Xpool.iloc[tr], Xpool.iloc[va]
            yt, yv = ypool.iloc[tr], ypool.iloc[va]
            m = xgb.XGBClassifier(n_estimators=3000, tree_method="hist", enable_categorical=True,
                                  objective="binary:logistic", eval_metric="logloss",
                                  early_stopping_rounds=60, random_state=args.seed, n_jobs=-1, **p)
            m.fit(Xt, yt, sample_weight=compute_sample_weight("balanced", yt),
                  eval_set=[(Xv, yv)], verbose=False)
            sc.append(f1_score(yv, m.predict(Xv), average="macro"))
            trial.report(float(np.mean(sc)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(sc))

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=args.n_trials,
                   callbacks=[lambda st, tr: tr.number % 10 == 0 and
                              print(f"[{tr.number + 1}/{args.n_trials}] best={st.best_value:.4f} "
                                    f"({time.time() - t0:.0f}s)", flush=True)])
    print(f"\nBest CV macro F1: {study.best_value:.4f}\nBest params: {study.best_params}")

    # Head-to-head vs the incumbent dispatch-tuned params, multi-seed.
    with open(MODEL_DIR / "best_hyperparams_normalized_binary.json") as f:
        incumbent = json.load(f)["params"]
    sw = compute_sample_weight("balanced", y["train"])
    for tag, params in [("incumbent (dispatch-tuned)", incumbent),
                        ("challenger (full-tuned)", study.best_params)]:
        f1s, accs = [], []
        for seed in (42, 7, 2026):
            m = xgb.XGBClassifier(n_estimators=3000, tree_method="hist", enable_categorical=True,
                                  objective="binary:logistic", eval_metric="logloss",
                                  early_stopping_rounds=60, random_state=seed, n_jobs=-1, **params)
            m.fit(X["train"], y["train"], sample_weight=sw,
                  eval_set=[(X["valid"], y["valid"])], verbose=False)
            pred = m.predict(X["test"])
            f1s.append(f1_score(y["test"], pred, average="macro"))
            accs.append(accuracy_score(y["test"], pred))
        print(f"{tag:<28} test macro F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  acc={np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
