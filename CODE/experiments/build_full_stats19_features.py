#!/usr/bin/env python3
"""build_full_stats19_features.py

Builds the "full-STATS19 upper bound" feature set used by
full_stats19_upper_bound.py: every pre-label STATS19 field not already in
the dispatch-legit BASE_COLS/SOURCE_COLS (train_classifier.py), joined onto
a given set of {train,valid,test}_norm.csv splits by collision_index. This
is a REFERENCE experiment, never a deployment candidate -- several of these
fields (most importantly police_attended) are things no 999 dispatcher
would know at call time, and one (police_attended) is leakage-adjacent
(attendance is itself driven by how severe the collision looks).

Column groups added (28 "clean" pre-label fields + 1 flagged leakage-
adjacent field + 2 fields found net-positive in a follow-up ablation; 2
fields -- generic_make_model, local_authority_highway -- were tested and
found net-NEGATIVE, high-cardinality noise, and are deliberately excluded):

  collision  police_force, local_authority_district, road_surface_conditions,
             special_conditions_at_site, carriageway_hazards
  attend     did_police_officer_attend_scene_of_accident (LEAKAGE-ADJACENT --
             see docstring above; kept separate so callers can drop it)
  vehicle    forensic detail (skidding, impact points, manoeuvre, junction
             location, restricted lane, work journey) + driver demographics
             (age, sex, engine cc, vehicle age, IMD decile) + propulsion type
             + driver distance-from-home banding
  casualty   passenger type, road-worker flag, IMD decile, distance-from-home
             banding, dominant pedestrian location/movement

EXCLUDED everywhere (direct label family -- would be leakage): collision/
casualty enhanced_severity, injury_based, adjusted_severity_serious/slight.

Raw source: the archived 415k download's codes/ tables (no re-download).
Update RAW_DIR below if working from a different raw download.

Usage:
    python build_full_stats19_features.py \\
        --norm-dir data/splits/normalized \\
        --out-dir data/splits/full_stats19
    python build_full_stats19_features.py \\
        --norm-dir data/archive_2026-07-17_90k_default/splits/normalized \\
        --out-dir /tmp/full_stats19_90k
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = ("data/archive_2026-07-17_scaleup_415k/"
           "stats19_20260716_175811_raw/codes")
RAW_FILE = "stats19_20260716_175811_{table}.csv"

# net-negative in ablation (high-cardinality noise / redundant) -- excluded
EXCLUDE_NOISY = {"generic_make_model", "local_authority_highway"}

CATEGORICAL_EXTRA = [
    "police_force", "local_authority_district", "road_surface_conditions",
    "dominant_manoeuvre", "dominant_junction_location", "dominant_propulsion",
    "min_driver_distance_band", "dominant_casualty_distance_band",
    "dominant_ped_location", "dominant_ped_movement",
]


def mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan


def load_extras(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coll = pd.read_csv(raw_dir / RAW_FILE.format(table="collisions"),
                       usecols=["collision_index", "police_force",
                                "local_authority_district", "road_surface_conditions",
                                "special_conditions_at_site", "carriageway_hazards",
                                "did_police_officer_attend_scene_of_accident"])
    coll = coll.rename(columns={"did_police_officer_attend_scene_of_accident": "police_attended"})

    veh_cols = ["collision_index", "vehicle_manoeuvre", "junction_location",
                "vehicle_location_restricted_lane", "skidding_and_overturning",
                "hit_object_in_carriageway", "vehicle_leaving_carriageway",
                "hit_object_off_carriageway", "first_point_of_impact",
                "journey_purpose_of_driver", "sex_of_driver", "age_of_driver",
                "engine_capacity_cc", "age_of_vehicle", "driver_imd_decile",
                "propulsion_code", "driver_distance_banding"]
    veh = pd.read_csv(raw_dir / RAW_FILE.format(table="vehicles"), usecols=veh_cols,
                      low_memory=False)
    for c in ("age_of_driver", "engine_capacity_cc", "age_of_vehicle", "driver_imd_decile"):
        veh.loc[veh[c] <= 0, c] = np.nan
    g = veh.groupby("collision_index")
    veh_agg = pd.DataFrame({
        "any_skidding": g["skidding_and_overturning"].apply(lambda s: int((s > 0).any())),
        "any_hit_object_in": g["hit_object_in_carriageway"].apply(lambda s: int((s > 0).any())),
        "any_left_carriageway": g["vehicle_leaving_carriageway"].apply(lambda s: int((s > 0).any())),
        "any_hit_object_off": g["hit_object_off_carriageway"].apply(lambda s: int((s > 0).any())),
        "n_front_impact": g["first_point_of_impact"].apply(lambda s: int((s == 1).sum())),
        "n_back_impact": g["first_point_of_impact"].apply(lambda s: int((s == 2).sum())),
        "dominant_manoeuvre": g["vehicle_manoeuvre"].apply(mode_or_nan),
        "dominant_junction_location": g["junction_location"].apply(mode_or_nan),
        "any_restricted_lane": g["vehicle_location_restricted_lane"].apply(lambda s: int((s > 0).any())),
        "any_work_journey": g["journey_purpose_of_driver"].apply(lambda s: int(s.isin([1, 2]).any())),
        "youngest_driver_age": g["age_of_driver"].min(),
        "oldest_driver_age": g["age_of_driver"].max(),
        "n_male_drivers": g["sex_of_driver"].apply(lambda s: int((s == 1).sum())),
        "max_engine_cc": g["engine_capacity_cc"].max(),
        "max_vehicle_age": g["age_of_vehicle"].max(),
        "mean_driver_imd": g["driver_imd_decile"].mean(),
        "dominant_propulsion": g["propulsion_code"].apply(mode_or_nan),
        "min_driver_distance_band": g["driver_distance_banding"].apply(mode_or_nan),
    }).reset_index()

    cas_cols = ["collision_index", "car_passenger", "bus_or_coach_passenger",
                "pedestrian_road_maintenance_worker", "pedestrian_location",
                "pedestrian_movement", "casualty_imd_decile", "casualty_class",
                "casualty_distance_banding"]
    cas = pd.read_csv(raw_dir / RAW_FILE.format(table="casualties"), usecols=cas_cols)
    cas.loc[cas.casualty_imd_decile <= 0, "casualty_imd_decile"] = np.nan
    gc = cas.groupby("collision_index")
    cas_agg = pd.DataFrame({
        "any_car_passenger": gc["car_passenger"].apply(lambda s: int((s > 0).any())),
        "any_bus_passenger": gc["bus_or_coach_passenger"].apply(lambda s: int((s > 0).any())),
        "any_road_worker": gc["pedestrian_road_maintenance_worker"].apply(lambda s: int((s == 1).any())),
        "mean_casualty_imd": gc["casualty_imd_decile"].mean(),
        "dominant_casualty_distance_band": gc["casualty_distance_banding"].apply(mode_or_nan),
    }).reset_index()
    ped_agg = (cas[cas.casualty_class == 3].groupby("collision_index").agg(
        dominant_ped_location=("pedestrian_location", mode_or_nan),
        dominant_ped_movement=("pedestrian_movement", mode_or_nan),
    ).reset_index())

    return coll, veh_agg, cas_agg, ped_agg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--norm-dir", default="data/splits/normalized",
                        help="Directory with {train,valid,test}_norm.csv to extend.")
    parser.add_argument("--out-dir", default="data/splits/full_stats19",
                        help="Output directory (created if missing). Never overwrites "
                             "the input files -- writes {split}_full.csv alongside.")
    parser.add_argument("--raw-dir", default=RAW_DIR,
                        help="Directory with the raw STATS19 codes/ CSVs.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coll, veh_agg, cas_agg, ped_agg = load_extras(raw_dir)

    for split in ("train", "valid", "test"):
        in_path = Path(args.norm_dir) / f"{split}_norm.csv"
        df = pd.read_csv(in_path)
        n0 = df.shape[1]
        for extra in (coll, veh_agg, cas_agg, ped_agg):
            df = df.merge(extra, on="collision_index", how="left")
        out_path = out_dir / f"{split}_full.csv"
        df.to_csv(out_path, index=False)
        print(f"{split}: {len(df)} rows, {n0} -> {df.shape[1]} cols "
              f"({in_path} -> {out_path})")


if __name__ == "__main__":
    main()
