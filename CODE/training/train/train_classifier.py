#!/usr/bin/env python3
"""train_classifier.py

Trains a single supervised severity classifier, and doubles as the shared
library every other Python script in the project imports from (feature
definitions, split loading, evaluation).

NOTE ON SCOPE (August 2026): the --sources CLI here trains one of the *8*
classifiers from the original "Active Feature Acquisition" design -- one per
subset of the three external data sources (AADF / OSM / Open-Meteo). That
action space was superseded: the external sources carry little of the
available signal, so the STATS19 fields became acquirable too, grouped into
the 10 bundles defined in BUNDLE_COLS below (2^10 = 1024 classifiers,
pre-trained by CODE/training/train/bundle_sweep.py). The --sources path is kept
because the historical experiments in CODE/experiments/ are written against
it and their results are still cited; new acquisition work should use
bundle_feature_columns() rather than feature_columns().

Reads the normalized splits produced by normalize_splits.R
(data/splits/normalized/{train,valid,test}_norm.csv) by default -- one row per
collision, no deliberate NaNs. --data-variant switches to the raw
(untransformed) splits, or to the masked splits from generate_masked_splits.R.
The masked variant was the default until August 2026 and is now deprecated:
masking a single model's inputs lost to training a dedicated model per
acquisition subset at every level of missingness, and it made every headline
metric a pooled average across missingness levels rather than a number about
any real deployment scenario. It is kept only for reproducing historical
results.

Usage (current, 10-bundle action space -- pick bundles by name or by bitmask):
    python train_classifier.py --bundles vehicles casualties   # two bundles held
    python train_classifier.py --bundles                       # free info only (timestamp)
    python train_classifier.py --mask 7        # same as --bundles, by acquisition bitmask
    python train_classifier.py --mask 1023     # full acquisition (all 10 bundles)

Usage (legacy 3-external-source action space):
    python train_classifier.py --sources aadf osm meteo   # full-acquisition subset
    python train_classifier.py --sources                  # no external sources (base only)
    python train_classifier.py --sources aadf

Combines with, in either action space:
    python train_classifier.py --binary                    # Slight vs. Severe (Fatal+Serious merged)
    python train_classifier.py --use-best-params           # Optuna-tuned hyperparameters
    python train_classifier.py --data-variant raw          # neither normalized nor masked
    python train_classifier.py --data-variant masked       # deprecated; historical results only

To train every bundle subset at once (2^10 = 1024 models, ~30 min, cached to
models/subsets/ as the RL environment's model bank), use bundle_sweep.py
instead of looping over this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
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


# ---------------------------------------------------------------------------
# Bundle action space (Active Feature Acquisition)
#
# The RL agent does not acquire individual columns -- it performs *acquisition
# acts*: one question to the 999 caller, one map lookup against the collision
# location, or one external API call. BUNDLE_COLS groups every acquirable
# feature into one of those acts, which is what makes 2^n tractable: 9 bundles
# = 512 subsets, versus 2^31 if each column were its own action.
#
# The free set is just the call timestamp: at the moment the 999 call connects,
# it is the only thing known without asking a question or looking anything up.
# Everything else -- including the spatial risk-density features, which need a
# nearest-neighbour + 500m-radius query against the historical collision
# database (add_risk_density_features.R) -- is an acquisition act with a cost,
# and therefore lives in a bundle. free_columns() below is the accessor; the
# timestamp column names are variant-dependent (cyclic sin/cos on
# normalized/masked, plain day_of_week/hour on raw), which is why they come
# from DATA_VARIANT_TIME_COLS rather than a flat constant.
#
# Bundle order below IS the bit order of the acquisition mask (bit 0 =
# vehicles); do not reorder without invalidating every cached subset model.

BUNDLE_COLS = {
    # -- ask the caller: costs seconds of call time --
    "vehicles": [
        "number_of_vehicles", "has_motorcycle", "has_pedal_cycle", "has_van",
        "has_car", "has_hgv", "has_bus_coach", "has_towing",
    ],
    "casualties": [
        "n_casualties", "n_pedestrian_casualties", "n_driver_cas",
        "n_passenger_cas", "n_motorcyclist_cas", "n_cyclist_cas",
    ],
    "casualty_demographics": [
        "youngest_casualty_age", "oldest_casualty_age", "n_child_cas",
        "n_elderly_cas", "n_female_cas",
    ],
    # -- map lookup on the collision coordinate: costs GIS query latency --
    "road_class": [
        "first_road_class", "first_road_number", "second_road_class",
        "second_road_number", "road_type", "trunk_road_flag",
    ],
    # Conditions at the scene. weather_conditions is the officer's subjective
    # weather judgement; it sits here rather than being swapped out when
    # "meteo" is acquired, so that every bundle is strictly additive -- an
    # acquisition can only ever add information, never remove it.
    "road_environment": [
        "speed_limit", "urban_or_rural_area", "light_conditions",
        "weather_conditions", "pedestrian_crossing",
    ],
    "junction": ["junction_detail", "junction_control"],
    "crash_history": [
        "hist_acc_rate_500m", "hist_severe_frac_500m", "hist_fatal_rate_500m",
        "dist_to_nearest_hist_acc_m",
    ],
    # -- external API: costs a network call that can be slow or fail --
    "aadf": SOURCE_COLS["aadf"],
    "osm": SOURCE_COLS["osm"],
    "meteo": SOURCE_COLS["meteo"],
}
BUNDLE_NAMES = list(BUNDLE_COLS)
N_BUNDLES = len(BUNDLE_NAMES)

# Which *kind* of acquisition act each bundle represents. This is a purely
# descriptive label, used for reporting and analysis ("how much of the value
# comes from questions vs. lookups?"). It does NOT group bundles into a single
# action and it does NOT set cost -- every bundle is an independent action with
# its own cost -- see the note below.
BUNDLE_TIERS = {
    "vehicles": "ask_caller", "casualties": "ask_caller",
    "casualty_demographics": "ask_caller",
    "road_class": "map_lookup", "road_environment": "map_lookup",
    "junction": "map_lookup", "crash_history": "map_lookup",
    "aadf": "external_api", "osm": "external_api", "meteo": "external_api",
}

# Acquisition cost (the C_i term in the README's reward function) is NOT
# defined here yet -- it belongs to the RL work and will be set then. When it
# is added, it must be **one cost per bundle, set independently**: every bundle
# is its own action, and bundles sharing a BUNDLE_TIERS label are not required
# to cost the same (asking a caller for casualty ages is slower and likelier to
# stall than asking what vehicles are involved). Do not derive cost from tier.

# The bundles cover exactly the acquirable half of the legacy feature set:
# every bundle == BASE_COLS + every SOURCE_COLS entry + weather_conditions
# (which is not in BASE_COLS because feature_columns() adds it conditionally).
# Asserted so a column added to BASE_COLS without a home in a bundle fails
# loudly here rather than silently dropping out of the acquisition action space.
assert {c for cols in BUNDLE_COLS.values() for c in cols} == (
    set(BASE_COLS) | {c for cols in SOURCE_COLS.values() for c in cols}
    | {"weather_conditions"}
), "BUNDLE_COLS no longer covers BASE_COLS + SOURCE_COLS + weather_conditions"


def free_columns(variant: str = "normalized") -> list[str]:
    """Columns available at zero acquisition cost -- the call timestamp, and
    nothing else. Every other feature belongs to a bundle."""
    return list(DATA_VARIANT_TIME_COLS[variant])


def bundle_mask(bundles) -> int:
    """Bundle names -> integer bitmask (bit i = BUNDLE_NAMES[i] acquired)."""
    unknown = set(bundles) - set(BUNDLE_NAMES)
    if unknown:
        raise ValueError(f"unknown bundle(s): {sorted(unknown)}")
    return sum(1 << i for i, b in enumerate(BUNDLE_NAMES) if b in set(bundles))


def mask_bundles(mask: int) -> list[str]:
    """Integer bitmask -> the bundle names it acquires, in BUNDLE_NAMES order."""
    return [b for i, b in enumerate(BUNDLE_NAMES) if mask >> i & 1]


def bundle_subset_name(bundles) -> str:
    return "none" if not bundles else "+".join(
        b for b in BUNDLE_NAMES if b in set(bundles))


def bundle_feature_columns(bundles, variant: str = "normalized") -> list[str]:
    """Feature columns for one acquisition subset: always-free columns plus
    every column in each acquired bundle."""
    acquired = set(bundles)
    cols = free_columns(variant)
    for b in BUNDLE_NAMES:
        if b in acquired:
            cols += BUNDLE_COLS[b]
    return cols


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
    pr_auc = _pr_auc(model, X, y, target_names)
    if verbose:
        print(f"\n[{split}] macro F1 = {macro_f1:.4f}  accuracy = {accuracy:.4f}  "
              f"balanced accuracy = {balanced_accuracy:.4f}")
        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
        print(f"[{split}] confusion matrix (rows=true, cols=pred, order={'/'.join(target_names)}):\n{np.array(cm)}")
        print(f"[{split}] PR-AUC (average precision, one-vs-rest):")
        for name in target_names:
            print(f"    {name:24s} {pr_auc['per_class'][name]:.4f}  "
                  f"(baseline = prevalence {pr_auc['baseline'][name]:.4f})")
        print(f"    {'macro':24s} {pr_auc['macro']:.4f}  "
              f"(baseline {pr_auc['macro_baseline']:.4f})")
    return {"macro_f1": macro_f1, "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy, "report": report,
            "confusion_matrix": cm, "pr_auc": pr_auc}


def _pr_auc(model, X: pd.DataFrame, y: pd.Series, target_names: list) -> dict:
    """One-vs-rest average precision per class, plus the no-skill baseline.

    The baseline for a PR curve is the positive-class prevalence, so it is
    reported alongside every number -- on these severity-oversampled splits a
    bare PR-AUC is not interpretable without it (see CLAUDE.md on class balance).
    """
    proba = model.predict_proba(X)
    y = np.asarray(y)
    per_class, baseline = {}, {}
    for idx, name in enumerate(target_names):
        y_bin = (y == idx).astype(int)
        per_class[name] = float(average_precision_score(y_bin, proba[:, idx]))
        baseline[name] = float(y_bin.mean())
    return {
        "per_class": per_class,
        "baseline": baseline,
        "macro": float(np.mean(list(per_class.values()))),
        "macro_baseline": float(np.mean(list(baseline.values()))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action_space = parser.add_mutually_exclusive_group()
    action_space.add_argument("--bundles", nargs="*", choices=BUNDLE_NAMES, default=None,
                         metavar="BUNDLE",
                         help="Acquisition bundles held by this classifier, from the current "
                              f"10-bundle action space: {', '.join(BUNDLE_NAMES)}. "
                              "Empty (--bundles with no values) trains the free-information "
                              "model -- call timestamp only. This is the current action space; "
                              "to train all 2^10 subsets at once use bundle_sweep.py instead.")
    action_space.add_argument("--mask", type=int, default=None,
                         help="Same as --bundles, but as the integer acquisition bitmask "
                              "(0..1023; bit i = BUNDLE_NAMES[i]). This is the RL state "
                              "encoding and the subset-model cache key, so it is the "
                              "convenient form when reproducing one subset from a sweep.")
    action_space.add_argument("--sources", nargs="*", choices=list(SOURCE_COLS), default=None,
                         help="LEGACY 3-source action space: external sources acquired for this "
                              "classifier, any subset of aadf/osm/meteo. Superseded by --bundles "
                              "(August 2026), but kept because the experiments in "
                              "CODE/experiments/ are written against it. Defaults to all three "
                              "when neither --bundles nor --mask is given.")
    parser.add_argument("--data-variant", choices=list(DATA_VARIANT_DIRS), default="normalized",
                         help="'normalized' (default): normalized, one row per collision, no "
                              "deliberate NaNs. 'masked': adds partial-observability masking -- "
                              "DEPRECATED, superseded by training a dedicated model per "
                              "acquisition subset; kept for reproducing historical results. "
                              "'raw': neither normalized nor masked -- isolates the effect of "
                              "normalization.")
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
                              "hyperparameters instead of --max-depth/--learning-rate, so every "
                              "subset classifier shares the config tuned once on the "
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
                              "differently-distributed benchmark (Severe was 36%% Fatal, easier; "
                              "now 94%% Serious, harder). A 62,828-row --train-frac subsample of "
                              "the current 290k train set scores ~0.663 test macro F1 on the "
                              "current test set, not the ~0.697 the old model got on the old one.")
    args = parser.parse_args()

    if not 0 < args.train_frac <= 1.0:
        raise ValueError(f"--train-frac must be in (0, 1], got {args.train_frac}")

    eval_variant = args.eval_variant or args.data_variant

    # Two action spaces coexist: --bundles/--mask (current) and --sources
    # (legacy). They produce different feature sets and different filenames, so
    # resolve exactly one here and let the rest of main() stay agnostic.
    if args.mask is not None:
        if not 0 <= args.mask < (1 << N_BUNDLES):
            raise ValueError(f"--mask must be in [0, {(1 << N_BUNDLES) - 1}], got {args.mask}")
        bundles = mask_bundles(args.mask)
    else:
        bundles = args.bundles
    using_bundles = bundles is not None

    if using_bundles:
        mask = bundle_mask(bundles)
        name = bundle_subset_name(bundles)
        cols = bundle_feature_columns(bundles, args.data_variant)
        subset_meta = {"action_space": "bundles", "bundles": bundles, "mask": mask}
        stem = f"bundles{mask:04d}"
        print(f"Subset: mask={mask} ({name})  data_variant={args.data_variant}  "
              f"eval_variant={eval_variant}  ({len(cols)} feature columns)")
    else:
        sources = list(SOURCE_COLS) if args.sources is None else args.sources
        name = subset_name(sources)
        cols = feature_columns(sources, args.data_variant)
        subset_meta = {"action_space": "sources", "subset": sources}
        stem = name
        print(f"Subset: {name}  data_variant={args.data_variant}  eval_variant={eval_variant}  "
              f"({len(cols)} feature columns)  [legacy --sources action space]")

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
        # is shared across every subset classifier.
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
    # the main "masked" 3-class model/metrics used everywhere else. (The
    # "masked" stem carries no variant tag for backwards compatibility with
    # every filename written before the default moved to "normalized".)
    file_stem = stem if args.data_variant == "masked" else f"{stem}_{args.data_variant}"
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
        **subset_meta,
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
