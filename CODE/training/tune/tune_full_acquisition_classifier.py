#!/usr/bin/env python3
"""tune_full_acquisition_classifier.py

One-time hyperparameter search for the Full-Acquisition (aadf+osm+meteo)
classifier, using Optuna + GroupKFold cross-validation. The winning
hyperparameters are saved to models/best_hyperparams.json and are meant to
be reused, unchanged, for every subset classifier trained by
train_classifier.py (--use-best-params) -- we tune once on the richest
feature set rather than re-running a search per subset -- which matters far
more now that there are 1024 bundle subsets (bundle_sweep.py) than it did
when there were 8. A per-subset tune would also make any comparison
between subsets a comparison of tuning budgets, not of information.

GroupKFold is grouped by collision_index: generate_masked_splits.R expands
each original accident into 11 near-duplicate copies (1 unmasked + 10
randomly masked). Those copies must never be split across a fold's train and
validation sides, or the "validation" score would partly reward
memorization of copies the model has effectively already seen.

Train + valid splits are pooled into one cross-validation set (test stays a
fully untouched holdout, evaluated only once at the end with the winning
config). To keep 60 trials tractable, each trial's CV runs on a random
subsample of *collision groups* (not individual rows) from that pool --
subsampling by group preserves the property that all 11 copies of a sampled
accident travel together.

Usage:
    python tune_full_acquisition_classifier.py --n-trials 60
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_classifier import (
    DATA_VARIANT_DIRS, DATA_VARIANT_FILE, MODEL_DIR, TARGET_COL,
    feature_columns, prepare_xy,
)

SOURCES = ["aadf", "osm", "meteo"]  # Full-Acquisition subset only

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_pool(cols: list[str], variant: str) -> pd.DataFrame:
    frames = []
    for split in ["train", "valid"]:
        path = DATA_VARIANT_DIRS[variant] / DATA_VARIANT_FILE[variant].format(name=split)
        df = pd.read_csv(path, usecols=lambda c: c in cols + [TARGET_COL, "collision_index"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def subsample_by_group(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    groups = df["collision_index"].unique()
    rng = np.random.RandomState(seed)
    keep = rng.choice(groups, size=int(len(groups) * frac), replace=False)
    return df[df["collision_index"].isin(keep)].reset_index(drop=True)


def make_objective(df: pd.DataFrame, cols: list[str], n_splits: int, n_estimators: int,
                    early_stopping_rounds: int, seed: int, binary: bool):
    X_all, y_all = prepare_xy(df, cols, binary)
    groups = df["collision_index"].values
    fold_indices = list(GroupKFold(n_splits=n_splits).split(X_all, y_all, groups))

    if binary:
        objective_kwargs = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        objective_kwargs = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
            sample_weight = compute_sample_weight("balanced", y_tr)

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                tree_method="hist",
                early_stopping_rounds=early_stopping_rounds,
                random_state=seed,
                n_jobs=-1,
                **objective_kwargs,
                **params,
            )
            model.fit(X_tr, y_tr, sample_weight=sample_weight,
                      eval_set=[(X_va, y_va)], verbose=False)
            score = f1_score(y_va, model.predict(X_va), average="macro")
            fold_scores.append(score)

            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--sample-frac", type=float, default=0.3,
                         help="Fraction of collision groups (not rows) sampled from the "
                              "pooled train+valid data for cross-validation.")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary", action="store_true",
                         help="Tune for the Slight-vs-Severe binary objective instead of the "
                              "3-class one; saves to best_hyperparams[_variant]_binary.json.")
    parser.add_argument("--data-variant", choices=list(DATA_VARIANT_DIRS), default="masked",
                         help="Which split to tune on. 'masked' (default) uses the 691k-row "
                              "partial-observability splits; 'normalized' uses the clean 62k-row "
                              "one-row-per-collision splits -- the optimal hyperparameters differ "
                              "(the masked set's heavy regularization can be wrong for clean data). "
                              "GroupKFold by collision_index is still valid for 'normalized' since "
                              "each collision appears exactly once (equivalent to plain KFold).")
    args = parser.parse_args()

    variant = args.data_variant
    cols = feature_columns(SOURCES, variant)
    print(f"Full-Acquisition subset: {len(cols)} feature columns  variant={variant}  binary={args.binary}")

    pool = load_pool(cols, variant)
    print(f"Pooled train+valid: {pool['collision_index'].nunique()} collision groups, {len(pool)} rows")

    cv_df = subsample_by_group(pool, args.sample_frac, args.seed)
    print(f"CV subsample: {cv_df['collision_index'].nunique()} groups, {len(cv_df)} rows "
          f"({args.n_splits}-fold GroupKFold)")

    objective = make_objective(cv_df, cols, args.n_splits, args.n_estimators,
                                args.early_stopping_rounds, args.seed, args.binary)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    t0 = time.time()

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        state = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else "done"
        print(f"[trial {trial.number + 1}/{args.n_trials}] {state} "
              f"value={trial.value if trial.value is not None else float('nan'):.4f} "
              f"best={study.best_value:.4f} ({time.time() - t0:.0f}s elapsed)")

    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_trial])

    print(f"\nSearch done in {time.time() - t0:.0f}s over {len(study.trials)} trials")
    print(f"Best CV macro F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    MODEL_DIR.mkdir(exist_ok=True)
    # Filename: best_hyperparams[_variant][_binary].json. The 'masked' variant
    # keeps the original unsuffixed names for backward compatibility.
    variant_suffix = "" if variant == "masked" else f"_{variant}"
    binary_suffix = "_binary" if args.binary else ""
    out_path = MODEL_DIR / f"best_hyperparams{variant_suffix}{binary_suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "source_subset": SOURCES,
            "data_variant": variant,
            "binary": args.binary,
            "cv_macro_f1": study.best_value,
            "n_trials": args.n_trials,
            "n_splits": args.n_splits,
            "sample_frac": args.sample_frac,
            "params": study.best_params,
        }, f, indent=2)
    print(f"Best hyperparameters saved to {out_path}")
    binary_flag = " --binary" if args.binary else ""
    variant_flag = "" if variant == "masked" else f" --data-variant {variant}"
    print(f"\nNext: python train_classifier.py --sources aadf osm meteo{variant_flag}{binary_flag} --use-best-params")
    if not args.binary:
        print("      (and reuse --use-best-params for the other 7 subsets too)")


if __name__ == "__main__":
    main()
