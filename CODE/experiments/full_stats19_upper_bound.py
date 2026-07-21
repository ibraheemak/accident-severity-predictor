#!/usr/bin/env python3
"""full_stats19_upper_bound.py

How much does the dispatch-time-only constraint cost, and does that cost
change with dataset size? Trains two models side by side at each of several
(SERIOUS_CAP, SLIGHT_CAP) dataset-construction sizes:

  dispatch   the real deployment feature set (BASE_COLS + aadf/osm/meteo,
             53 cols as of 2026-07-19) -- what a 999 dispatcher could know.
  full       + every pre-label STATS19 field (build_full_stats19_features.py's
             28 "clean" fields + the leakage-adjacent police_attended flag +
             2 fields an ablation found net-positive) -- an upper bound using
             information no dispatcher has, NEVER a deployment candidate.

Both draw from the archived 415,550-collision scale-up dataset
(data/archive_2026-07-17_scaleup_415k/), subsampled per-year the same way
stats19_download.R's severity_sample() does -- no fresh download, no re-join
of AADF/OSM/meteo (those don't depend on which rows are sampled, so the
archive's already-joined values are reused directly).

Usage:
    python full_stats19_upper_bound.py
    python full_stats19_upper_bound.py --sizes 1400:2800 2800:5600 4200:8400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import CATEGORICAL_COLS, TARGET_COL, feature_columns, unify_categorical_dtypes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_full_stats19_features import CATEGORICAL_EXTRA, load_extras

ARCH_NORM = "data/archive_2026-07-17_scaleup_415k/splits/normalized"
ARCH_RAW = Path("data/archive_2026-07-17_scaleup_415k/stats19_20260716_175811_raw/codes")
SPLIT_FRAC = {"train": 0.70, "valid": 0.15, "test": 0.15}
YEARS = list(range(2016, 2025))
SEEDS = [42, 7, 2026]

DISPATCH_COLS = feature_columns(["aadf", "osm", "meteo"], "normalized")
FULL_EXTRA_COLS = [
    "police_force", "local_authority_district", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards", "police_attended",
    "any_skidding", "any_hit_object_in", "any_left_carriageway", "any_hit_object_off",
    "n_front_impact", "n_back_impact", "dominant_manoeuvre", "dominant_junction_location",
    "any_restricted_lane", "any_work_journey",
    "youngest_driver_age", "oldest_driver_age", "n_male_drivers", "max_engine_cc",
    "max_vehicle_age", "mean_driver_imd", "dominant_propulsion", "min_driver_distance_band",
    "any_car_passenger", "any_bus_passenger", "any_road_worker", "mean_casualty_imd",
    "dominant_casualty_distance_band", "dominant_ped_location", "dominant_ped_movement",
]
FULL_CAT = CATEGORICAL_COLS + CATEGORICAL_EXTRA


def subsample(df: pd.DataFrame, scap: int, lcap: int, frac: float, seed: int) -> pd.DataFrame:
    parts = []
    rng = np.random.RandomState(seed)
    for year, g in df.groupby("data_year"):
        fatal = g[g[TARGET_COL] == 1]
        serious = g[g[TARGET_COL] == 2]
        slight = g[g[TARGET_COL] == 3]
        ns = min(len(serious), int(round(scap * frac)))
        nl = min(len(slight), int(round(lcap * frac)))
        parts += [fatal,
                  serious.sample(n=ns, random_state=rng) if ns < len(serious) else serious,
                  slight.sample(n=nl, random_state=rng) if nl < len(slight) else slight]
    return pd.concat(parts, ignore_index=True)


def fit_eval(feats, cat_cols, frames, y, seed, params):
    Xl = [frames[s][feats].copy() for s in ("train", "valid", "test")]
    unify_categorical_dtypes(Xl, feats) if cat_cols is None else None
    if cat_cols is not None:
        present = [c for c in cat_cols if c in feats]
        for col in present:
            cats = sorted(pd.concat([f[col] for f in Xl]).dropna().unique())
            dtype = pd.CategoricalDtype(categories=cats)
            for f in Xl:
                f[col] = f[col].astype(dtype)
    Xtr, Xva, Xte = Xl
    m = xgb.XGBClassifier(n_estimators=2000, tree_method="hist", enable_categorical=True,
                          objective="binary:logistic", eval_metric="logloss",
                          early_stopping_rounds=50, random_state=seed, n_jobs=-1, **params)
    m.fit(Xtr, y["train"], sample_weight=compute_sample_weight("balanced", y["train"]),
          eval_set=[(Xva, y["valid"])], verbose=False)
    te = f1_score(y["test"], m.predict(Xte), average="macro")
    auc = roc_auc_score(y["test"], m.predict_proba(Xte)[:, 1])
    return te, auc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["1400:2800", "2800:5600", "4200:8400", "11200:22400"],
                        help="SERIOUS_CAP:SLIGHT_CAP pairs, per-year (Fatal always all kept).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    dispatch_need = DISPATCH_COLS + [TARGET_COL, "data_year", "collision_index"]
    splits = {s: pd.read_csv(f"{ARCH_NORM}/{s}_norm.csv", usecols=lambda c: c in dispatch_need)
              for s in ("train", "valid", "test")}

    print("Loading full-STATS19 extra columns from the raw archive ...")
    coll, veh_agg, cas_agg, ped_agg = load_extras(ARCH_RAW)

    import json
    with open("models/best_hyperparams_normalized_binary.json") as f:
        params = json.load(f)["params"]

    print(f"\n{'size':<14} {'n_total':>8} {'fatalOfSev':>10}  "
          f"{'dispatch_F1':>12} {'dispatch_AUC':>13}  {'full_F1':>9} {'full_AUC':>9}  {'gap':>6}")
    for size_spec in args.sizes:
        scap, lcap = (int(x) for x in size_spec.split(":"))
        d_tes, d_aucs, f_tes, f_aucs, n_total = [], [], [], [], 0
        for seed in args.seeds:
            sub = {s: subsample(splits[s], scap, lcap, SPLIT_FRAC[s], seed) for s in splits}
            n_total = sum(len(s) for s in sub.values())

            y = {s: (sub[s][TARGET_COL].astype(int) != 3).astype(int) for s in sub}
            d_te, d_auc = fit_eval(DISPATCH_COLS, None, sub, y, seed, params)
            d_tes.append(d_te); d_aucs.append(d_auc)

            full_frames = {}
            for s in sub:
                fdf = sub[s].copy()
                for extra in (coll, veh_agg, cas_agg, ped_agg):
                    fdf = fdf.merge(extra, on="collision_index", how="left")
                full_frames[s] = fdf
            f_te, f_auc = fit_eval(DISPATCH_COLS + FULL_EXTRA_COLS, FULL_CAT, full_frames, y, seed, params)
            f_tes.append(f_te); f_aucs.append(f_auc)

        all_y = pd.concat([subsample(splits[s], scap, lcap, SPLIT_FRAC[s], args.seeds[0])[TARGET_COL]
                           for s in splits])
        fos = (all_y == 1).sum() / (all_y <= 2).sum()
        gap = np.mean(f_tes) - np.mean(d_tes)
        print(f"{size_spec:<14} {n_total:>8} {fos:>10.1%}  "
              f"{np.mean(d_tes):.4f}±{np.std(d_tes):.4f}  {np.mean(d_aucs):>8.4f}  "
              f"{np.mean(f_tes):.4f}  {np.mean(f_aucs):>8.4f}  {gap:>+.4f}")


if __name__ == "__main__":
    main()
