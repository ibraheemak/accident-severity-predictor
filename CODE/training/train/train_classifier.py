#!/usr/bin/env python3
"""train_classifier.py

Trains one of the 8 frozen supervised classifiers described in the README's
"Active Feature Acquisition" design -- one classifier per possible subset of
the three external data sources (AADF traffic, OSM road infrastructure,
Open-Meteo weather). The RL agent later chooses, per accident, which sources
to acquire before invoking the matching frozen classifier.

Reads the masked/normalized splits produced by normalize_splits.R +
generate_masked_splits.R (data/splits/masked/{train,valid,test}_masked.csv) by
default. --data-variant switches to the normalized-but-unmasked splits or the
raw (untransformed) splits instead, to isolate whether masking or
normalization is responsible for a given result -- e.g. if macro F1 looks
the same on "raw" as on "masked", the masking/normalization pipeline isn't
the bottleneck.

Usage:
    python train_classifier.py --sources aadf osm meteo   # full-acquisition subset
    python train_classifier.py --sources                  # no external sources (base only)
    python train_classifier.py --sources aadf
    python train_classifier.py --data-variant normalized   # normalized, not masked
    python train_classifier.py --data-variant raw          # neither normalized nor masked
    python train_classifier.py --binary                    # Slight vs. Severe (Fatal+Serious merged)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# Where each --data-variant reads its {train,valid,test} split from, and the
# filename pattern within that directory.
DATA_VARIANT_DIRS = {
    "masked":     Path("data/splits/masked"),
    "normalized": Path("data/splits/normalized"),
    "raw":        Path("data/splits"),
}
DATA_VARIANT_FILE = {
    "masked":     "{name}_masked.csv",
    "normalized": "{name}_norm.csv",
    "raw":        "{name}.csv",
}
# masked/normalized splits already have day_of_week/time replaced by their
# cyclic sin/cos encoding (normalize_splits.R step 1); the raw split still
# has the original day_of_week (1-7) and time ("HH:MM:SS") columns, so we
# derive a plain (non-cyclic) "hour" column from it on load instead.
DATA_VARIANT_TIME_COLS = {
    "masked":     ["day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"],
    "normalized": ["day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"],
    "raw":        ["day_of_week", "hour"],
}

MODEL_DIR = Path("models")

SEVERITY_LABELS = {0: "Fatal", 1: "Serious", 2: "Slight"}  # 0-based, from raw 1/2/3
BINARY_LABELS = {0: "Slight", 1: "Severe (Fatal+Serious)"}  # --binary: merges Fatal+Serious

# Columns that are never features: join keys, traceability, masking bookkeeping.
METADATA_COLS = [
    "collision_index", "collision_year", "longitude", "latitude", "date",
    "data_year", "copy_id", "mask_rate", "aadf_match_type"
]
TARGET_COL = "collision_severity"

# STATS19-derived fields always available as features, regardless of which
# external sources are acquired. weather_conditions is deliberately excluded
# here: it's the officer's subjective assessment and is superseded by the
# Open-Meteo columns when the "meteo" source is acquired (see BASE_COLS below
# for the non-meteo case). light_conditions has no external replacement yet,
# so it always stays.
BASE_COLS = [
    "number_of_vehicles", "first_road_class", "first_road_number", "road_type",
    "speed_limit", "junction_detail", "junction_control", "second_road_class",
    "second_road_number", "pedestrian_crossing", "urban_or_rural_area",
    "trunk_road_flag", "light_conditions",
    "has_motorcycle", "has_pedal_cycle", "has_van", "has_car", "has_hgv",
    "has_bus_coach", "has_towing",
    # Spatial risk-density features (add_risk_density_features.R): derived
    # purely from lat/lon against prior-year accident history, so they're
    # free at 999-call time -- base features, not an acquirable source.
    "hist_acc_rate_500m", "hist_severe_frac_500m", "hist_fatal_rate_500m",
    "dist_to_nearest_hist_acc_m",
    # Dispatch-time casualty facts (build_realtime_casualty_features.R):
    # what a 999 caller reports -- how many hurt, pedestrian involvement,
    # rough ages, and who the hurt are (driver/passenger/child/elderly/
    # motorcyclist/cyclist/female counts, re-added 2026-07-19: worth +0.006
    # test macro F1 on the 1400/2800-cap config). Recorded injury severity
    # is deliberately excluded.
    "n_casualties", "n_pedestrian_casualties", "youngest_casualty_age",
    "oldest_casualty_age",
    "n_driver_cas", "n_passenger_cas", "n_child_cas", "n_elderly_cas",
    "n_motorcyclist_cas", "n_cyclist_cas", "n_female_cas",
]

# Nominal/unordered category codes (mirrors NOMINAL_COLS in
# normalize_splits.R, which leaves these as raw integer codes rather than
# z-scoring them). Cast to pandas `category` dtype + XGBoost's
# enable_categorical=True so the tree can use native categorical splits
# instead of treating unordered codes as if they had numeric order.
CATEGORICAL_COLS = [
    "first_road_class", "road_type", "junction_detail", "junction_control",
    "second_road_class", "pedestrian_crossing", "urban_or_rural_area",
    "light_conditions", "weather_conditions", "weather_code",
]

# Per-source feature columns, added only when that source is in --sources.
# dist_to_speed_camera_m was dropped from "osm" (2026-07-21): 90.4% NaN (most
# collisions aren't near a camera), near-zero feature importance, and an
# external-source ablation on the 1400/2800-cap binary model found dropping it
# a slight net positive (aadf+osm 0.7396 -> aadf+osm_dense 0.7398). The R
# pipeline (add_osm_features.R) still computes the column; it is simply no
# longer fed to the model. (has_speed_camera, the binary presence flag, is kept
# -- it's the part of the camera signal that carries value.)
SOURCE_COLS = {
    "aadf": ["aadf_traffic_volume", "aadf_distance_m",],
    "osm": [
        "has_speed_camera", "has_traffic_calming",
        "has_traffic_signals", "lane_count", "has_median",
    ],
    "meteo": [
        "temperature_2m_c", "precipitation_mm", "wind_speed_10m_kmh",
        "weather_code", "snowfall_cm", "snow_depth_m",
    ],
}

# feature_importance_audit.py flagged 11 features with negative permutation
# importance on the Full-Acquisition model (mostly the sparsest/highest-
# missingness fields -- junction_control 70.8% missing, dist_to_speed_camera_m
# 90.2%, second_road_number 69.8% -- plus OSM current-day-snapshot / zero-
# inflated weather fields). Tested dropping them and retraining: net
# *worse* (test macro F1 0.4674->0.4631 uncalibrated, 0.4768->0.4747
# calibrated) -- permutation importance on one trained model doesn't
# reliably predict what a retrain-from-scratch does without those features.
# Reverted; kept all columns below.


def subset_name(sources: list[str]) -> str:
    return "base" if not sources else "_".join(sorted(sources))


def feature_columns(sources: list[str], variant: str = "masked") -> list[str]:
    cols = list(DATA_VARIANT_TIME_COLS[variant]) + list(BASE_COLS)
    if "meteo" not in sources:
        cols.append("weather_conditions")  # fallback when no objective weather acquired
    for src in sources:
        cols += SOURCE_COLS[src]
    return cols


def load_split(name: str, cols: list[str], variant: str = "masked") -> pd.DataFrame:
    path = DATA_VARIANT_DIRS[variant] / DATA_VARIANT_FILE[variant].format(name=name)
    needs_hour = variant == "raw" and "hour" in cols
    wanted = [c for c in cols if c != "hour"] + [TARGET_COL] + (["time"] if needs_hour else [])
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    if needs_hour:
        df["hour"] = df["time"].str.split(":").str[0].astype(int)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    return df


def prepare_xy(df: pd.DataFrame, cols: list[str], binary: bool = False):
    X = df[cols].copy()
    y = df[TARGET_COL].astype(int) - 1  # 1/2/3 (Fatal/Serious/Slight) -> 0/1/2
    if binary:
        y = (y != 2).astype(int)  # 1 = Severe (was Fatal or Serious), 0 = Slight
    return X, y


def unify_categorical_dtypes(frames: list[pd.DataFrame], cols: list[str]) -> list[str]:
    """Cast nominal columns to a shared pandas CategoricalDtype (union of
    categories across all frames), in place, so train/valid/test agree on
    category codes for XGBoost's native categorical splits. Returns the
    categorical columns actually present."""
    present = [c for c in CATEGORICAL_COLS if c in cols]
    for col in present:
        categories = sorted(pd.concat([f[col] for f in frames]).dropna().unique())
        dtype = pd.CategoricalDtype(categories=categories)
        for f in frames:
            f[col] = f[col].astype(dtype)
    return present


def evaluate(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, split: str,
             labels: dict = SEVERITY_LABELS, verbose: bool = True) -> dict:
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    target_names = [labels[i] for i in range(len(labels))]
    report = classification_report(
        y, y_pred, target_names=target_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred).tolist()
    if verbose:
        print(f"\n[{split}] macro F1 = {macro_f1:.4f}  accuracy = {accuracy:.4f}  "
              f"balanced accuracy = {balanced_accuracy:.4f}")
        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
        print(f"[{split}] confusion matrix (rows=true, cols=pred, order={'/'.join(target_names)}):\n{np.array(cm)}")
    return {"macro_f1": macro_f1, "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy, "report": report,
            "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", choices=list(SOURCE_COLS), default=list(SOURCE_COLS),
                         help="External sources acquired for this classifier (default: all three).")
    parser.add_argument("--data-variant", choices=list(DATA_VARIANT_DIRS), default="masked",
                         help="'masked' (default): normalized + partial-observability masking. "
                              "'normalized': normalized but not masked (one row per collision, "
                              "no deliberate NaNs) -- isolates the effect of masking. "
                              "'raw': neither normalized nor masked -- isolates the effect of "
                              "normalization too.")
    parser.add_argument("--eval-variant", choices=list(DATA_VARIANT_DIRS), default=None,
                         help="Evaluate valid/test on this variant instead of --data-variant, "
                              "while still training on --data-variant. E.g. "
                              "--data-variant masked --eval-variant normalized trains on the "
                              "masked (partial-observability) splits but evaluates on the clean, "
                              "unmasked, one-row-per-collision splits -- tests whether masking "
                              "during training helps or hurts when the model is scored at "
                              "full-information inference time, rather than being diluted across "
                              "the masked test set's mix of missingness levels. Only variants "
                              "with matching feature schemas can be combined (masked <-> "
                              "normalized both use cyclic day/hour encoding; 'raw' uses a "
                              "different time encoding and can't be mixed with the other two).")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-best-params", action="store_true",
                         help="Load models/best_hyperparams.json (from "
                              "tune_full_acquisition_classifier.py) and use those XGBoost "
                              "hyperparameters instead of --max-depth/--learning-rate, so all "
                              "8 subset classifiers share the config tuned once on the "
                              "Full-Acquisition subset.")
    parser.add_argument("--binary", action="store_true",
                         help="Merge Fatal+Serious into one 'Severe' class and train a binary "
                              "Slight-vs-Severe classifier instead of the 3-class one.")
    parser.add_argument("--train-frac", type=float, default=1.0,
                         help="Use only this fraction of the training split (default 1.0 = all "
                              "of it), sampled by collision group with --seed. valid/test are "
                              "never subsampled, so this is a controlled way to trace out a "
                              "learning curve or speed up iteration -- NOT a way to reproduce "
                              "old pre-2026-07-17 scores: those were measured on a smaller AND "
                              "differently-distributed benchmark (Severe was 36% Fatal, easier; "
                              "now 94% Serious, harder). A 62,828-row --train-frac subsample of "
                              "the current 290k train set scores ~0.663 test macro F1 on the "
                              "current test set, not the ~0.697 the old model got on the old one.")
    args = parser.parse_args()

    if not 0 < args.train_frac <= 1.0:
        raise ValueError(f"--train-frac must be in (0, 1], got {args.train_frac}")

    eval_variant = args.eval_variant or args.data_variant

    name = subset_name(args.sources)
    cols = feature_columns(args.sources, args.data_variant)
    print(f"Subset: {name}  data_variant={args.data_variant}  eval_variant={eval_variant}  "
          f"({len(cols)} feature columns)")

    train_load_cols = cols if args.train_frac == 1.0 else cols + ["collision_index"]
    train_df = load_split("train", train_load_cols, args.data_variant)
    if args.train_frac < 1.0:
        groups = train_df["collision_index"].unique()
        rng = np.random.RandomState(args.seed)
        n_keep = max(1, round(len(groups) * args.train_frac))
        keep = rng.choice(groups, size=n_keep, replace=False)
        n_rows_before = len(train_df)
        train_df = train_df[train_df["collision_index"].isin(keep)].reset_index(drop=True)
        print(f"--train-frac {args.train_frac}: {len(keep)}/{len(groups)} collision groups "
              f"-> {len(train_df)}/{n_rows_before} rows")
    valid_df = load_split("valid", cols, eval_variant)
    test_df = load_split("test", cols, eval_variant)
    print(f"Rows: train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    X_train, y_train = prepare_xy(train_df, cols, args.binary)
    X_valid, y_valid = prepare_xy(valid_df, cols, args.binary)
    X_test, y_test = prepare_xy(test_df, cols, args.binary)

    categorical_cols = unify_categorical_dtypes([X_train, X_valid, X_test], cols)
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    sample_weight = compute_sample_weight("balanced", y_train)

    if args.use_best_params:
        # Prefer a variant-specific tune (best_hyperparams_<variant>[_binary].json)
        # when one exists; otherwise fall back to the masked-variant tune, which
        # is shared across all 8 subset classifiers.
        variant_suffix = "" if args.data_variant == "masked" else f"_{args.data_variant}"
        binary_suffix = "_binary" if args.binary else ""
        best_path = MODEL_DIR / f"best_hyperparams{variant_suffix}{binary_suffix}.json"
        if not best_path.exists() and variant_suffix:
            best_path = MODEL_DIR / f"best_hyperparams{binary_suffix}.json"
        with open(best_path) as f:
            xgb_params = json.load(f)["params"]
        print(f"Using tuned hyperparameters from {best_path}: {xgb_params}")
    else:
        xgb_params = {"max_depth": args.max_depth, "learning_rate": args.learning_rate}

    if args.binary:
        objective_kwargs = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        objective_kwargs = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"}

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=-1,
        **objective_kwargs,
        **xgb_params,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")

    labels = BINARY_LABELS if args.binary else SEVERITY_LABELS
    valid_metrics = evaluate(model, X_valid, y_valid, "valid", labels)
    test_metrics = evaluate(model, X_test, y_test, "test", labels)

    # Non-default variants/binary get a filename suffix so they don't clobber
    # the main "masked" 3-class model/metrics used everywhere else.
    file_stem = name if args.data_variant == "masked" else f"{name}_{args.data_variant}"
    if eval_variant != args.data_variant:
        file_stem += f"_evalon_{eval_variant}"
    if args.binary:
        file_stem += "_binary"
    if args.train_frac < 1.0:
        file_stem += f"_trainfrac{args.train_frac:g}"

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"classifier_{file_stem}.json"
    model.save_model(model_path)

    meta = {
        "subset": args.sources,
        "data_variant": args.data_variant,
        "eval_variant": eval_variant,
        "binary": args.binary,
        "train_frac": args.train_frac,
        "n_train_rows": len(train_df),
        "feature_columns": cols,
        "categorical_columns": categorical_cols,
        "best_iteration": int(model.best_iteration),
        "params": {
            "n_estimators": args.n_estimators,
            "seed": args.seed,
            **xgb_params,
        },
        "valid": {"macro_f1": valid_metrics["macro_f1"],
                  "accuracy": valid_metrics["accuracy"],
                  "balanced_accuracy": valid_metrics["balanced_accuracy"]},
        "test": {"macro_f1": test_metrics["macro_f1"],
                 "accuracy": test_metrics["accuracy"],
                 "balanced_accuracy": test_metrics["balanced_accuracy"]},
    }
    meta_path = MODEL_DIR / f"classifier_{file_stem}_metrics.json"
    with open(meta_path, "w") as f:
        json.dump({**meta, "valid_full": valid_metrics, "test_full": test_metrics}, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {meta_path}")


if __name__ == "__main__":
    main()
