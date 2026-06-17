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

### Output: `data/features/stats19/stats19_collision_realtime_features.csv`
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

### Output: `data/features/stats19/stats19_vehicle_realtime_features.csv`
82,491 rows (one per collision) × 16 columns (`collision_index` + 15 flags).
Flag prevalence on the 2020–2025 sample: `has_car` 84.8%, `has_motorcycle`
17.7%, `has_pedal_cycle` 15.3%, `has_van` 10.9%, `has_hgv` 4.8%, `has_bus_coach`
2.8%, `has_towing` 2.8%, `has_other_vehicle` 1.1%, `has_escooter` 1.1%, others
< 1%.

---

## 5. Combined Feature Table — `build_combined_features.R`

Left-joins the collision and vehicle realtime feature tables on `collision_index`
into a single model-ready file.

### Output: `data/features/stats19/stats19_combined_realtime_features.csv`
82,491 rows × 38 columns (23 collision columns + 15 vehicle flags, minus the
duplicate join key). Every sampled collision has a matching vehicle row, so no
NAs are introduced by the join.

---

## 6. AADF Traffic Volume Join — `join_aadf.R`

Enriches every accident with an annual average daily flow (AADF) traffic volume
figure from the DfT count-point dataset, using a spatial + road-name join.

### Source data
```
data/raw/aadf/dft_traffic_counts_aadf.csv
```
Columns used: `count_point_id`, `year`, `latitude`, `longitude`, `road_name`,
`all_motor_vehicles`.

### Road ID construction

Each accident's `first_road_class` and `second_road_class` are numeric codes in
Stats19. The script maps them to a letter prefix and concatenates with the road
number to build identifiers that match AADF's `road_name` field:

| Code | Class | Prefix | Example |
|------|-------|--------|---------|
| 1 | Motorway | M | M6 |
| 2 | A(M) | A | A1(M) → A1 |
| 3 | A | A | A461 |
| 4 | B | B | B4009 |
| 5 | C | C | C12 |
| 6 | Unclassified | U | U1234 |

Two IDs are built per accident:
- `accident_road_id` — from `first_road_class` + `first_road_number`
- `accident_road_id_2` — from `second_road_class` + `second_road_number`
  (NA when the accident is not at a junction or has no second road)

### Matching logic (per accident, same year only)

Three steps, tried in order, stopping at the first that succeeds:

1. **Same road (primary)** — filter AADF count points to those whose `road_name`
   equals `accident_road_id` exactly. If any exist, pick the geographically
   nearest one (no distance threshold). → `aadf_match_type = "same_road"`

2. **Same road (secondary)** — if step 1 found zero count points, repeat with
   `accident_road_id_2`. → `aadf_match_type = "same_road"`

3. **Nearest fallback** — if neither road exists in AADF at all, find the
   nearest count point on any road (no distance threshold).
   → `aadf_match_type = "nearest_fallback"`

No accident is left unmatched (the nearest-fallback always finds something).
The `aadf_distance_m` column records the actual distance in metres for every
match; the model can use this as a reliability signal or apply its own threshold.

### New columns added

| Column | Type | Description |
|--------|------|-------------|
| `aadf_traffic_volume` | integer | Annual average daily flow (all motor vehicles) from the matched count point |
| `aadf_distance_m` | double | Distance in metres from the accident to the matched count point |
| `aadf_match_type` | character | `"same_road"` or `"nearest_fallback"` |

### Output: `data/features/stats19+aadf/stats19+aadf_combined_realtime_features.csv`
Same rows as `stats19_combined_realtime_features.csv` plus the three new columns
above. Row order is preserved.

### Logging
Every run writes a full log to `data/features/stats19+aadf/join_aadf.log`, including match-type
counts per year and a distance-distribution table bucketed at
`<100 m / 100–200 m / 200–300 m / 300–500 m / 500 m–1 km / 1–2 km / 2–5 km / >5 km`.

### Debug mode
Set `DEBUG <- TRUE` at the top of the script to process only `DEBUG_N_ACC`
accidents from `DEBUG_YEAR`. Per-accident detail (road IDs tried, candidate
counts, distances) is written to `join_aadf.log` at `[DEBUG]` level.

---

## 7. Next steps (not yet done)

- Decide treatment for `light_conditions` / `weather_conditions` (replace with
  external API at inference; for training, decide whether to keep STATS19
  values or also source historical weather externally for consistency).
- Encode categorical/code columns as needed for the chosen model.
- Casualty-derived features (if any) — not yet built; `casualties` codes/labels
  CSVs are available but unused so far.
