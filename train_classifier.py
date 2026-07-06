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
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
    "has_motorcycle", "has_car", "has_hgv", "has_pedal_cycle", "has_minibus",
    "has_bus_coach", "has_ridden_horse", "has_agricultural", "has_tram",
    "has_van", "has_mobility_scooter", "has_other_vehicle",
    "has_unknown_vehicle", "has_towing", "has_escooter",
]

# Per-source feature columns, added only when that source is in --sources.
SOURCE_COLS = {
    "aadf": ["aadf_traffic_volume", "aadf_distance_m",],
    "osm": [
        "has_speed_camera", "dist_to_speed_camera_m", "has_traffic_calming",
        "has_traffic_signals", "lane_count", "has_median",
    ],
    "meteo": [
        "temperature_2m_c", "precipitation_mm", "wind_speed_10m_kmh",
        "weather_code", "snowfall_cm", "snow_depth_m",
    ],
}


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


def prepare_xy(df: pd.DataFrame, cols: list[str]):
    X = df[cols].copy()
    y = df[TARGET_COL].astype(int) - 1  # 1/2/3 (Fatal/Serious/Slight) -> 0/1/2
    return X, y


def evaluate(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    report = classification_report(
        y, y_pred, target_names=[SEVERITY_LABELS[i] for i in range(3)],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred).tolist()
    print(f"\n[{split}] macro F1 = {macro_f1:.4f}")
    print(classification_report(
        y, y_pred, target_names=[SEVERITY_LABELS[i] for i in range(3)], zero_division=0,
    ))
    print(f"[{split}] confusion matrix (rows=true, cols=pred, order=Fatal/Serious/Slight):\n{np.array(cm)}")
    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm}


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
    args = parser.parse_args()

    name = subset_name(args.sources)
    cols = feature_columns(args.sources, args.data_variant)
    print(f"Subset: {name}  data_variant={args.data_variant}  ({len(cols)} feature columns)")

    train_df = load_split("train", cols, args.data_variant)
    valid_df = load_split("valid", cols, args.data_variant)
    test_df = load_split("test", cols, args.data_variant)
    print(f"Rows: train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    X_train, y_train = prepare_xy(train_df, cols)
    X_valid, y_valid = prepare_xy(valid_df, cols)
    X_test, y_test = prepare_xy(test_df, cols)

    sample_weight = compute_sample_weight("balanced", y_train)

    if args.use_best_params:
        best_path = MODEL_DIR / "best_hyperparams.json"
        with open(best_path) as f:
            xgb_params = json.load(f)["params"]
        print(f"Using tuned hyperparameters from {best_path}: {xgb_params}")
    else:
        xgb_params = {"max_depth": args.max_depth, "learning_rate": args.learning_rate}

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=-1,
        **xgb_params,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")

    valid_metrics = evaluate(model, X_valid, y_valid, "valid")
    test_metrics = evaluate(model, X_test, y_test, "test")

    # Non-default variants get a filename suffix so they don't clobber the
    # main "masked" model/metrics used everywhere else.
    file_stem = name if args.data_variant == "masked" else f"{name}_{args.data_variant}"

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"classifier_{file_stem}.json"
    model.save_model(model_path)

    meta = {
        "subset": args.sources,
        "data_variant": args.data_variant,
        "feature_columns": cols,
        "best_iteration": int(model.best_iteration),
        "params": {
            "n_estimators": args.n_estimators,
            "seed": args.seed,
            **xgb_params,
        },
        "valid": {"macro_f1": valid_metrics["macro_f1"]},
        "test": {"macro_f1": test_metrics["macro_f1"]},
    }
    meta_path = MODEL_DIR / f"classifier_{file_stem}_metrics.json"
    with open(meta_path, "w") as f:
        json.dump({**meta, "valid_full": valid_metrics, "test_full": test_metrics}, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {meta_path}")


if __name__ == "__main__":
    main()
