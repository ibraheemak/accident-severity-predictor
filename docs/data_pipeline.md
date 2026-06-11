# Data Pipeline — Summary

This document summarizes the data acquisition and feature-extraction pipeline
built for the accident severity predictor project.

---

## 1. Download & Sampling — `stats19_download.R`

Downloads UK STATS19 road safety data via the official `stats19` R package.

### Config
```r
YEARS       <- 2020:2025
SERIOUS_CAP <- 5000   # per year
SLIGHT_CAP  <- 10000  # per year
RANDOM_SEED <- 42
```

### What it does
1. For each year, downloads three tables: **collisions**, **casualties**, **vehicles**
   - Note: DfT renamed "accident" → "collision" (post-2016 package); type strings
     are `"collision"`, `"casualty"`, `"vehicle"`; join key is `collision_index`.
2. **Severity-aware sampling** (applied to collisions only, then propagated):
   - Fatal: keep **all** rows
   - Serious: keep up to `SERIOUS_CAP` (5,000), all if fewer available
   - Slight: keep up to `SLIGHT_CAP` (10,000), all if fewer available
3. Casualties and vehicles are filtered to the same sampled `collision_index`
   values, so all three tables stay row-aligned across the full pipeline.
4. **Two parallel versions of every table are downloaded**:
   - `format=TRUE` → human-readable string labels (for EDA / reports)
   - `format=FALSE` → raw numeric codes (for ML pipeline)
   - Both versions use the *same* sampled `collision_index` set, and rows are
     ordered identically (collisions: Fatal→Serious→Slight; casualties/vehicles:
     sorted by `collision_index`) so row N in labels == row N in codes.
5. 404s / empty downloads (e.g. unpublished future years) are skipped with a
   warning, not a crash.

### Output structure
```
data/raw/stats19_<TIMESTAMP>/
├── stats19_<TIMESTAMP>_meta.txt
├── labels/
│   ├── stats19_<TIMESTAMP>_collisions.csv
│   ├── stats19_<TIMESTAMP>_casualties.csv
│   └── stats19_<TIMESTAMP>_vehicles.csv
└── codes/
    ├── stats19_<TIMESTAMP>_collisions.csv
    ├── stats19_<TIMESTAMP>_casualties.csv
    └── stats19_<TIMESTAMP>_vehicles.csv
```

### Metadata file contains
- Run timestamp, YEARS, SERIOUS_CAP, SLIGHT_CAP, RANDOM_SEED
- Per-year breakdown: Fatal/Serious/Slight counts (sampled vs. total available),
  casualty/vehicle row counts
- Final severity class distribution
- Skipped years and reasons

---

## 2. Column Cleanup — `_historic` columns (codes version only)

DfT's 2024 spec change introduced `*_historic` columns alongside renamed/recoded
primary columns. `merge_historic_cols()` in `stats19_download.R` reconciles these
so the codes CSV has a single clean column per concept. Full details in
[`docs/column_merges.md`](column_merges.md). Summary:

- **`junction_detail_historic` → `junction_detail`**: codes are compatible across
  specs, so historic values backfill `NA`s in the primary column, then the
  historic column is dropped.
- **`pedestrian_crossing_human_control_historic` +
  `pedestrian_crossing_physical_facilities_historic`**: DfT already pre-populates
  the unified `pedestrian_crossing` column (new spec codes) for all years
  2020–2024. The two `_historic` columns add no recoverable info (verified: the
  ~0.6% of rows where `pedestrian_crossing` is NA also have blank `_historic`
  values) — both are simply dropped.

---

## 3. Realtime Collision Features — `build_realtime_features.R`

Reads the **codes** collisions CSV and selects only columns that will be
available in real time at inference (i.e. excludes columns that require
post-collision investigation).

### Output: `data/features/collision_realtime_features.csv`
One row per sampled collision. Columns are grouped into:

| Group | Columns | Purpose |
|---|---|---|
| **Infrastructure** (6) | `collision_index`, `collision_year`, `longitude`, `latitude`, `date`, `data_year` | Join keys / traceability — never fed to the model |
| **Target** (1) | `collision_severity` | Label (1=Fatal, 2=Serious, 3=Slight) |
| **ML features** (14) | `number_of_vehicles`, `day_of_week`, `time`, `first_road_class`, `first_road_number`, `road_type`, `speed_limit`, `junction_detail`, `junction_control`, `second_road_class`, `second_road_number`, `pedestrian_crossing`, `urban_or_rural_area`, `trunk_road_flag` | Fed into the model |
| **To-replace** (2) | `light_conditions`, `weather_conditions` | Kept for comparison only — will be swapped for live external weather/lighting API values at inference time |

The script also prints an NA summary per column. Notable NA rates (2020–2025
sample, ~82.5k rows): `junction_control` 44.6%, `second_road_number` 40.3%,
`trunk_road_flag` 8.2% (all expected — junction-only fields).

---

## 4. Realtime Vehicle Features — `build_vehicle_features.R`

Reads the **codes** vehicles CSV, filters to realtime columns, then aggregates
from one-row-per-vehicle to **one row per `collision_index`** using multi-hot
binary flags (since a collision can involve multiple vehicles).

### Step 1 — keep only:
`collision_index`, `vehicle_type`, `towing_and_articulation`, `escooter_flag`

### Step 2 — aggregate per collision into binary flags:

**Combined groups** (multiple `vehicle_type` codes → one flag):
| Flag | Codes |
|---|---|
| `has_motorcycle` | 2,3,4,5,23,97 |
| `has_car` | 8,9 |
| `has_hgv` | 20,21,98 |

**Individual codes** (one flag per remaining `vehicle_type` code, no combining):
| Flag | Code |
|---|---|
| `has_pedal_cycle` | 1 |
| `has_minibus` | 10 |
| `has_bus_coach` | 11 |
| `has_ridden_horse` | 16 |
| `has_agricultural` | 17 |
| `has_tram` | 18 |
| `has_van` | 19 |
| `has_mobility_scooter` | 22 |
| `has_other_vehicle` | 90 |
| `has_unknown_vehicle` | 99 |

**Other flags:**
| Flag | Definition |
|---|---|
| `has_towing` | any vehicle has `towing_and_articulation` in 1–5 |
| `has_escooter` | any vehicle has `escooter_flag == 1` |

### Output: `data/features/vehicle_realtime_features.csv`
82,491 rows (one per collision) × 16 columns (`collision_index` + 15 flags).
Flag prevalence on the 2020–2025 sample: `has_car` 84.8%, `has_motorcycle`
17.7%, `has_pedal_cycle` 15.3%, `has_van` 10.9%, `has_hgv` 4.8%, `has_bus_coach`
2.8%, `has_towing` 2.8%, `has_other_vehicle` 1.1%, `has_escooter` 1.1%, others
< 1%.

---

## 5. Next steps (not yet done)

- Join `collision_realtime_features.csv` with `vehicle_realtime_features.csv`
  on `collision_index` to produce the final model-ready feature table.
- Decide treatment for `light_conditions` / `weather_conditions` (replace with
  external API at inference; for training, decide whether to keep STATS19
  values or also source historical weather externally for consistency).
- Encode categorical/code columns as needed for the chosen model.
- Casualty-derived features (if any) — not yet built; `casualties` codes/labels
  CSVs are available but unused so far.
