#!/usr/bin/env python3
"""bundle_sweep.py

Trains one classifier per acquisition subset over the 9-bundle action space
defined in train_classifier.py (BUNDLE_COLS) -- 2^9 = 512 models -- and records
each one's valid/test macro F1.

This is the offline pre-fill of the model cache the RL Active Feature
Acquisition agent will sit on: at runtime the agent picks a subset of bundles,
and the environment loads models/subsets/<stem>/<mask>.json rather than fitting
anything on the fly. The resulting CSV is also a result in its own right -- it
is the value-of-information curve over the action space (what each bundle is
worth, and how much of that value is redundant with the bundles already held).

Resumable: a mask whose model file already exists is skipped, so an interrupted
sweep continues where it stopped. Use --force to refit regardless.

Usage (from the repo root):
    python CODE/training/train/bundle_sweep.py                 # all 1024, binary/normalized
    python CODE/training/train/bundle_sweep.py --max-bundles 1 # free + each single bundle
    python CODE/training/train/bundle_sweep.py --masks 0 1023
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from train_classifier import (
    BINARY_LABELS, BUNDLE_COLS, BUNDLE_NAMES, BUNDLE_TIERS, CATEGORICAL_COLS,
    MODEL_DIR, N_BUNDLES, SEVERITY_LABELS, bundle_feature_columns,
    bundle_subset_name, load_split, mask_bundles, prepare_xy,
    unify_categorical_dtypes,
)


def all_columns(variant: str) -> list[str]:
    """Union of every column any subset can need, so the splits are read from
    disk once instead of 512 times."""
    cols = bundle_feature_columns(BUNDLE_NAMES, variant)
    cols += [c for c in bundle_feature_columns([], variant) if c not in cols]
    return cols


def fit_subset(mask, frames, ys, cols, xgb_params, args):
    X_train, X_valid, X_test = (f[cols] for f in frames)
    y_train, y_valid, y_test = ys

    if args.binary:
        objective_kwargs = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        objective_kwargs = {"objective": "multi:softprob", "num_class": 3,
                            "eval_metric": "mlogloss"}

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=args.early_stopping_rounds, random_state=args.seed,
        n_jobs=-1, **objective_kwargs, **xgb_params,
    )
    model.fit(X_train, y_train,
              sample_weight=compute_sample_weight("balanced", y_train),
              eval_set=[(X_valid, y_valid)], verbose=False)

    out = {}
    for split, X, y in (("valid", X_valid, y_valid), ("test", X_test, y_test)):
        pred = model.predict(X)
        out[f"{split}_macro_f1"] = f1_score(y, pred, average="macro")
        out[f"{split}_accuracy"] = accuracy_score(y, pred)
        out[f"{split}_balanced_accuracy"] = balanced_accuracy_score(y, pred)
    return model, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-variant", default="normalized",
                    choices=["normalized", "raw", "masked"])
    ap.add_argument("--binary", action="store_true", default=True)
    ap.add_argument("--three-class", dest="binary", action="store_false",
                    help="3-class Fatal/Serious/Slight instead of the default binary.")
    ap.add_argument("--max-bundles", type=int, default=N_BUNDLES,
                    help="Only fit subsets of at most this many bundles (default: all 9). "
                         "--max-bundles 1 is a 10-model smoke test of the whole sweep.")
    ap.add_argument("--masks", type=int, nargs="*", default=None,
                    help="Fit only these specific masks instead of a full sweep.")
    ap.add_argument("--force", action="store_true", help="Refit masks already cached.")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    variant = args.data_variant
    stem = f"{variant}_{'binary' if args.binary else '3class'}"
    cache_dir = MODEL_DIR / "subsets" / stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path("docs/figures/bundle_sweep") / f"bundle_sweep_{stem}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical params: the tune that has won five independent head-to-heads on
    # this project (see models/best_hyperparams_normalized_binary.json note).
    # One shared config across all 512 subsets -- per-subset tuning would make
    # the value-of-information curve a comparison of tuning budgets, not of
    # information.
    hp_path = MODEL_DIR / f"best_hyperparams_{variant}_binary.json"
    if not hp_path.exists():
        hp_path = MODEL_DIR / "best_hyperparams.json"
    xgb_params = json.load(open(hp_path))["params"]
    print(f"Hyperparameters from {hp_path}: {xgb_params}")

    cols_all = all_columns(variant)
    frames_raw = [load_split(n, cols_all, variant) for n in ("train", "valid", "test")]
    print(f"Rows: train={len(frames_raw[0])} valid={len(frames_raw[1])} "
          f"test={len(frames_raw[2])}  ({len(cols_all)} columns loaded once)")

    prepared = [prepare_xy(df, cols_all, args.binary) for df in frames_raw]
    frames = [X for X, _ in prepared]
    ys = [y for _, y in prepared]
    unify_categorical_dtypes(frames, cols_all)

    if args.masks is not None:
        masks = list(args.masks)
    else:
        masks = [m for m in range(1 << N_BUNDLES)
                 if bin(m).count("1") <= args.max_bundles]
    print(f"Sweeping {len(masks)} subsets -> {cache_dir}")

    rows, t0 = [], time.time()
    for i, mask in enumerate(masks, 1):
        bundles = mask_bundles(mask)
        model_path = cache_dir / f"{mask:03d}.json"
        meta_path = cache_dir / f"{mask:03d}_metrics.json"
        if model_path.exists() and meta_path.exists() and not args.force:
            rows.append(json.load(open(meta_path)))
            continue

        cols = bundle_feature_columns(bundles, variant)
        t = time.time()
        model, metrics = fit_subset(mask, frames, ys, cols, xgb_params, args)
        model.save_model(model_path)

        row = {
            "mask": mask,
            "n_bundles": len(bundles),
            "subset": bundle_subset_name(bundles),
            "n_features": len(cols),
            "n_ask_caller": sum(BUNDLE_TIERS[b] == "ask_caller" for b in bundles),
            "n_map_lookup": sum(BUNDLE_TIERS[b] == "map_lookup" for b in bundles),
            "n_external_api": sum(BUNDLE_TIERS[b] == "external_api" for b in bundles),
            **{b: int(b in bundles) for b in BUNDLE_NAMES},
            **metrics,
            "best_iteration": int(model.best_iteration),
            "fit_seconds": round(time.time() - t, 2),
        }
        json.dump(row, open(meta_path, "w"), indent=2)
        rows.append(row)

        elapsed = time.time() - t0
        print(f"[{i}/{len(masks)}] mask={mask:3d} n={len(bundles)} "
              f"valid={metrics['valid_macro_f1']:.4f} test={metrics['test_macro_f1']:.4f} "
              f"({row['fit_seconds']:.1f}s, elapsed {elapsed/60:.1f}m) "
              f"{row['subset']}")

    # Rebuild the results CSV from *every* cached subset, not just the masks
    # this invocation touched -- otherwise a partial run (--masks/--max-bundles)
    # would clobber a full sweep's results with its own handful of rows.
    cached = [json.load(open(f)) for f in sorted(cache_dir.glob("*_metrics.json"))]
    df = pd.DataFrame(cached).sort_values("mask")
    df.to_csv(results_path, index=False)
    print(f"\n{len(rows)} subsets this run; {len(df)} cached in total -> {results_path}")

    if len(df) > 1:
        print("\nTop 10 by valid macro F1:")
        cols_show = ["mask", "n_bundles", "valid_macro_f1", "test_macro_f1", "subset"]
        print(df.nlargest(10, "valid_macro_f1")[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
