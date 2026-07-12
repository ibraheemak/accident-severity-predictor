# Data Pipeline — Summary

This document summarizes the data acquisition and feature-extraction pipeline
built for the accident severity predictor project.

> **Revision note (July 2026):** the pipeline now samples years **2016–2025**
> with per-year caps **2,800 Serious / 5,600 Slight** (Fatal uncapped),
> yielding **89,791 collisions (15.8% Fatal)** — chosen deliberately to nearly
> double the Fatal class share vs. the original 2020–2024 run (9.1%), which
> was the project's main accuracy bottleneck. Row counts and NA percentages
> quoted in sections 2–8 below are from the original 82k-row run and are
> indicative only; the structure of each step is unchanged. Sections 9–13
> reflect the current run.

---

## 1. Download & Sampling — `stats19_download.R`

Downloads UK STATS19 road safety data via the official `stats19` R package.

### Config
```r
YEARS       <- 2016:2025
SERIOUS_CAP <- 2800   # per year
SLIGHT_CAP  <- 5600   # per year
RANDOM_SEED <- 42
```

Fatal is never capped, so extending the year range grows the Fatal class
"for free" while the shrunken Serious/Slight caps keep their 9-year totals
near the original 5-year totals — net effect: Fatal share 9.1% → 15.8%
without ballooning the dataset (82,491 → 89,791 rows).

### What it does
1. For each year, downloads three tables: **collisions**, **casualties**, **vehicles**
   - Note: DfT renamed "accident" → "collision" (post-2016 package); type strings
     are `"collision"`, `"casualty"`, `"vehicle"`; join key is `collision_index`.
2. **Severity-aware sampling** (applied to collisions only, then propagated):
   - Fatal: keep **all** rows
   - Serious: keep up to `SERIOUS_CAP` (2,800), all if fewer available
   - Slight: keep up to `SLIGHT_CAP` (5,600), all if fewer available
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
data/raw/stats19/stats19_<TIMESTAMP>/
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

## 7. OpenStreetMap Road Infrastructure Features — `add_osm_features.R`

Enriches every accident with road-network features pulled from OpenStreetMap
(OSM): speed cameras, traffic calming, traffic signals, lane count, and
median presence.

### LIMITATION
OSM data is a **current-era snapshot** (downloaded at run time), while
accident years span 2020–2024. Road infrastructure tags (lanes, signals,
etc.) reflect the network as it exists today, not as it was at the time of
each accident. Treat these features as a proxy, not a historically accurate
record.

### Data sources
- **Road ways + barriers** (national scale): a Great Britain PBF extract from
  [Geofabrik](https://download.geofabrik.de/europe/great-britain-latest.osm.pbf),
  read via the `osmextract` package. Geofabrik does not publish a
  separate England-only extract, so both debug and full runs that touch
  osmextract use the GB-wide file. The PBF (~2 GB) and its converted
  GeoPackage are cached in `data/raw/osm/cache/` after the first run.
- **Point features** (speed cameras, traffic calming, traffic signals):
  queried live from the Overpass API (3 mirror servers, rotated on retry)
  using the bounding box of the accidents being processed.

### Why two different sources
Overpass times out / OOMs on national-scale way queries (e.g. all driveable
roads, or all `barrier=*` ways, across Great Britain), so those go through
the offline PBF extract instead. Point features are sparse enough nationally
that Overpass handles them directly. In `--debug` mode, the bounding box is
small (50 accidents clustered around the dataset median, clamped to a 15 km
radius), so road ways and barriers are also fetched via Overpass — osmextract
is skipped entirely in debug mode.

### Road matching algorithm (two-stage, ref-first then nearest)
Each accident's `first_road_class` + `first_road_number` (STATS19 codes) is
converted into an OSM-style ref (e.g. class `2`/`3` + number `461` → `"A461"`).
For each accident:
1. **Stage 1 (ref match)** — among road ways within the search radius, keep
   only those whose OSM `ref` tag matches the accident's constructed ref
   (handles semicolon-separated refs like `"A1;A2"`); pick the nearest.
2. **Stage 2 (fallback)** — if no ref match (or the accident has no
   constructed ref), fall back to the nearest way within radius regardless
   of ref.

Speed cameras and traffic signals rarely carry a `ref` tag themselves, so
each point is first assigned the `ref` of its nearest road way (within
100 m) before the same two-stage matching is applied to points.

### Tag extraction (osmextract `other_tags` HSTORE)
For the osmextract path, `extra_tags` is intentionally **not** used — passing
non-default tags to `oe_read()` was found to corrupt GDAL's vectortranslate
step (`osmconf.ini` modification issue) in this environment. Instead, only
default columns are pulled (including the `other_tags` HSTORE field), and
`ref`, `lanes`, `divider` are parsed out of `other_tags` in R via a small
regex-based HSTORE parser.

### New columns added

| Column | Type | Description |
|--------|------|--------------|
| `has_speed_camera` | 0/1 | Speed camera within 500 m |
| `dist_to_speed_camera_m` | double | Distance to the matched camera (NA if none) |
| `has_traffic_calming` | 0/1 | Traffic calming feature within 50 m |
| `has_traffic_signals` | 0/1 | Traffic signal node within 50 m |
| `lane_count` | integer | Lane count of the matched road way (NA if untagged) |
| `has_median` | 0/1 | Matched way has a `divider` tag, or a `barrier=*` way passes within 15 m |

Columns considered and **dropped** after generation, due to near-total
sparsity in OSM Great Britain coverage: `road_width_m`, `road_surface_material`,
`overtaking`, `shoulder`. `is_bridge`/`is_tunnel` and a speed-display-sign
feature were also tried and dropped — bridges/tunnels are rare events that
added little signal, and OSM's `highway=speed_display` tag is used on only a
few hundred nodes nationally.

### Output: `data/features/stats19+aadf+osm/stats19+aadf+osm_combined_realtime_features.csv`
82,491 rows × 47 columns (the AADF-joined table plus the 6 columns above).
Run matching achieved 100% coverage (every accident with valid coordinates
matched to a road way within 50 m).

### Debug mode
`Rscript CODE/processing/osm/add_osm_features.R --debug` processes 50 accidents clustered around
the dataset's median coordinate, with verbose `[DBG]` logging, writing to
`stats19+aadf+osm_combined_realtime_features_debug.csv` and a separate debug
log/cache directory.

### Logging
Every run writes to `data/features/stats19+aadf+osm/add_osm_features.log`
(or `..._debug.log`), including per-column fill-rate statistics, match-rate
percentages at each stage, and distance distributions for matched cameras.

### Completeness check — `check_feature_completeness.R`
A companion script reports how many rows have **zero genuinely missing**
values across the full combined table. Some NAs are structurally valid
rather than missing data (e.g. `dist_to_speed_camera_m` is NA because there
is no camera nearby; `second_road_class`/`junction_control` are NA because
the accident wasn't at a junction) — these are excused from the count. As of
the current feature set, 40,565/82,491 rows (49.2%) have no unexcused
missing values; `lane_count` (43.9% missing — untagged in OSM) is the
largest remaining gap.

---

## 8. Open-Meteo Historical Weather Features — `add_openmeteo_features.R`

Enriches every accident with objective, continuous weather measurements at
its exact lat/lon and hour, sourced from Open-Meteo's Historical Weather API
(ERA5/ERA5-Land reanalysis, hourly, back to 1940). This is a deliberate
upgrade over STATS19's `weather_conditions` column, which is a subjective
category assessed by the attending officer at the scene rather than a
measurement.

### Source
[open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)

### Batching strategy
Open-Meteo's archive endpoint accepts a comma-separated list of locations in
a single request and returns the full day's hourly series for each one. The
script groups accidents by **date** (not by location — exact coordinate
collisions between two different accidents are effectively impossible), then
splits each date's accidents into batches of `MAX_BATCH_SIZE` (100, chosen
conservatively — empirically, GET requests for 200 locations still succeeded
but 500+ hit `414 Request-URI Too Large`, since this is a query-string length
limit rather than a documented per-request location cap). One API call is
made per (date, batch); the correct hour's values are then picked out of that
batch's 24-hour response for each accident.

This collapses what would be ~82,000 row-level calls down to roughly one
call per distinct accident date (~1,800 for the 2020–2024 sample).

### Caching
Each (date, batch) response is cached to `data/raw/openmeteo/cache/*.rds`,
keyed as `<date>_b<batch_index>.rds`. A single full run never re-issues the
same key, so the cache's value is purely for **resuming** an interrupted run
(crash, rate limit, network failure) without re-fetching dates already
fetched — not for deduplicating within one run.

### Rate limiting
The free, anonymous endpoint (`archive-api.open-meteo.com`) is shared by IP
and capped at 600 calls/min, 5,000/hour, 10,000/day. The script self-throttles
to roughly one request every `MIN_REQUEST_INTERVAL_SEC` (2s) to avoid tripping
this, and on a `429` response retries the same batch with escalating backoff
(60s → 180s → 300s) before falling back to a circuit breaker that cools down
for `CIRCUIT_BREAKER_COOLDOWN` (10 min) and retries indefinitely (up to
`MAX_COOLDOWN_CYCLES`, ~3.3 hours) rather than giving up and writing NA — a
429 is a transient throttle, not a real data error.

For reliable, fast runs, a paid Open-Meteo plan removes this entirely. Set
`OPENMETEO_API_KEY` and `OPENMETEO_BASE_URL` in a **gitignored** `.Renviron`
file at the project root (R loads this automatically on startup — never
commit credentials into a tracked file). One subtlety: the dedicated
customer endpoint is **API-specific**, not a single generic domain — the
Historical Weather/Archive API's customer endpoint is
`customer-archive-api.open-meteo.com`, not the generic
`customer-api.open-meteo.com` shown in the top-level docs example (that
generic form 404s for the archive API; each Open-Meteo API has its own
`customer-<api-name>-api` subdomain).

```
# .Renviron (gitignored, not committed)
OPENMETEO_API_KEY=your-key-here
OPENMETEO_BASE_URL=https://customer-archive-api.open-meteo.com/v1/archive
```

### Hourly variables requested
`temperature_2m`, `precipitation`, `wind_speed_10m`, `weather_code`,
`snowfall`, `snow_depth`. **Visibility was requested and dropped**: the
archive endpoint's ERA5 reanalysis does not model it and returns `null` for
every hour regardless of location or date (verified directly against the
API) — there is no objective visibility source available from this provider
for historical dates.

### New columns added

| Column | Type | Description |
|--------|------|--------------|
| `temperature_2m_c` | double | Air temperature at 2m, °C |
| `precipitation_mm` | double | Precipitation, mm/hour |
| `wind_speed_10m_kmh` | double | Wind speed at 10m, km/h |
| `weather_code` | integer | WMO weather code for the hour |
| `snowfall_cm` | double | Snowfall, cm |
| `snow_depth_m` | double | Snow depth, m |

### Output: `data/features/stats19+aadf+osm+meteo/stats19+aadf+osm+meteo_combined_realtime_features.csv`
82,481 rows × 53 columns. 10 rows from the upstream table had no valid
lat/lon (and therefore no possible weather lookup) and were removed by the
companion script below rather than kept as all-NA.

### Debug mode
`Rscript CODE/processing/meteo/add_openmeteo_features.R --debug` processes the first 50 rows,
writing to `..._debug.csv` and a separate debug log/cache directory.

### Companion: `drop_invalid_meteo_rows.R`
Removes the 10 accidents with no valid lat/lon/date/time from the output
table (`temperature_2m_c` is NA for exactly these rows — there was never any
weather to look up). Run once after `add_openmeteo_features.R`:
```
Rscript CODE/processing/meteo/drop_invalid_meteo_rows.R
```

### Companion: `analyze_meteo_features.R`
Reports the distribution of every weather column: missing rate, quantiles
(Q1/median/Q3/P90/P99), mean/sd, zero-rate, a text histogram, a labelled WMO
`weather_code` frequency table, and pairwise correlations between the
continuous variables. Findings from the current sample: precipitation
(76.8%), snowfall (99.5%), and snow depth (97.9%) are heavily zero-inflated;
`weather_code` is dominated by Overcast (38.7%) and Clear sky (17.4%); only
2.7% of accidents occurred during measurable rain/snow; pairwise correlations
among the continuous features are all weak (|r| ≤ 0.16), i.e. little
redundancy between them.
```
Rscript CODE/processing/qa/analyze_meteo_features.R
```

---

## 9. Spatial Risk-Density Features — `add_risk_density_features.R`

Adds four location-risk features answering *"how much accident history does
this place have?"* — dangerous locations stay dangerous, and no other
feature captures location-specific risk directly.

### Leakage design
For an accident in year Y, features are computed **only from accidents in
years strictly < Y** — never the accident itself, never same-year or future
accidents. Past accidents are public knowledge at 999-call time, so these
are legitimately available at inference. First-year rows (2016) have no
history in the dataset and get NA for all four columns (XGBoost handles NaN
natively).

### Known bias, accepted
The dataset is severity-sampled (all Fatal, capped Serious/Slight per
year), so `hist_severe_frac_500m` is inflated relative to the true
population fraction — but the sampling fraction is constant across *space*
within each year, so the relative variation between locations (what the
model actually uses) is preserved.

### New columns

| Column | Description |
|---|---|
| `hist_acc_rate_500m` | Prior-year accidents within 500 m ÷ number of prior years |
| `hist_severe_frac_500m` | Fraction of those that were Fatal or Serious (NA if none nearby) |
| `hist_fatal_rate_500m` | Prior-year Fatal accidents within 500 m ÷ number of prior years |
| `dist_to_nearest_hist_acc_m` | Distance to the nearest prior-year accident |

Implementation: points are projected once to EPSG:27700 (metres, same as
`join_aadf.R`) and matched per-year against the pooled prior-year accidents
with `sf::st_is_within_distance()` / `st_nearest_feature()`.

Sanity check from the current run: median `hist_acc_rate_500m` is 0.67/yr
for urban rows vs 0.00/yr for rural, as expected. Effect on the
Full-Acquisition classifier: ~+0.005 macro F1 — small but consistent on
both valid and test, for both the 3-class and binary targets. In the RL
framing these are **base** features (derived purely from lat/lon against a
precomputed history — zero acquisition cost), so they are never masked.

### Output: `data/features/stats19+aadf+osm+meteo+risk/stats19+aadf+osm+meteo+risk_combined_realtime_features.csv`
89,779 rows × 57 columns.

---

## 10. Dispatch-Time Casualty Features — `build_realtime_casualty_features.R`

Aggregates the casualties **codes** table to one row per `collision_index`,
keeping only what a 999 caller genuinely reports at dispatch time
(*"a pedestrian's been hit"*, *"three people hurt"*, *"an elderly woman"*):

| Column | Description |
|---|---|
| `n_casualties` | Number of casualty rows for the collision |
| `n_pedestrian_casualties` | Casualties with `casualty_class == 3` (pedestrian) |
| `youngest_casualty_age` | Minimum `age_of_casualty` (NA if none recorded) |
| `oldest_casualty_age` | Maximum `age_of_casualty` (NA if none recorded) |

Explicitly **excluded** as label-adjacent / post-hoc / officer-assessed:
`casualty_severity`, `enhanced_casualty_severity`,
`casualty_adjusted_severity_*`, `casualty_injury_based`,
`casualty_imd_decile`, `lsoa_of_casualty`, `pedestrian_location`,
`pedestrian_movement`.

In the Active Feature Acquisition framing these are **ask-the-dispatcher**
fields — knowable during the call but not necessarily volunteered up
front — so they are added to `ACQUIRABLE_COLS` in
`generate_masked_splits.R` (28 → 32 maskable fields), exactly like the
vehicle-type flags.

This was the single largest accuracy improvement in the project: 3-class
calibrated test macro F1 **0.478 → 0.501** (Fatal F1 0.40 → 0.45, precision
*and* recall both up), binary Slight-vs-Severe test macro F1
**0.631 → 0.651**.

An extended set of five demographic/mode-of-travel columns
(child/elderly/cyclist/motorcyclist counts, age span) was trialled on top of
these but removed: a controlled multi-seed ablation showed it worth only
+0.0012 macro F1 on the normalized variant (within noise) and +0.0052 on
masked — not enough to justify five extra columns, so the feature set was
reverted to the original four.

The script also left-joins the four columns onto the risk-feature table
(every STATS19 collision has ≥ 1 casualty row, so the join introduces no
NAs — asserted in the script).

### Output: `data/features/stats19+aadf+osm+meteo+risk+cas/stats19+aadf+osm+meteo+risk+cas_combined_realtime_features.csv`
89,779 rows × 61 columns — this is the table the split step consumes.

---

## 11. Train/Valid/Test Split — `split_train_valid_test.R`

Splits the final combined feature table (`stats19+aadf+osm+meteo+risk+cas_combined_realtime_features.csv`,
89,779 rows) into train/validation/test sets, stratified by
`collision_year` × `collision_severity` so every set contains data from
every year with the same severity mix.

### Config
```r
TRAIN_FRAC <- 0.70
VALID_FRAC <- 0.15
TEST_FRAC  <- 0.15
SEED       <- 42
```

### Method
Rows are grouped into `collision_year` × `collision_severity` strata; within
each stratum, rows are shuffled and cut at the `TRAIN_FRAC`/`VALID_FRAC`
boundaries (floor-based), with any remainder going to test. This guarantees
proportional representation per year/severity in every split, not just
overall.

### Output: `data/splits/{train,valid,test}.csv`
| Split | Rows | % of total |
|---|---|---|
| train | 62,828 | 70.0% |
| valid | 13,458 | 15.0% |
| test  | 13,493 | 15.0% |

Per-year counts are balanced within ~0.1 pp of the target split fractions in
every year (2016–2024); full breakdown in `data/splits/split_stats.txt`
(generated by the companion `check_split_stats.R`).

---

## 12. Feature Normalization — `normalize_splits.R`

Transforms every non-binary, non-metadata model feature before the data is
handed to the masked-copy generator. All statistics are computed from the
**train split only** and applied identically to valid and test — no leakage.

Three transformations are applied in order:

---

### Step 1 — Cyclic encoding

`day_of_week` (1–7) and the hour extracted from `time` (0–23) are periodic:
day 7 and day 1 are adjacent, hour 23 and hour 0 are adjacent. A linear scale
or z-score breaks that adjacency, so both are replaced by a sin/cos pair:

| Raw column   | Period | Replacement columns                     |
|---|---|---|
| `day_of_week` | 7 | `day_of_week_sin`, `day_of_week_cos` |
| `time` (HH:MM:SS) | 24 h | `hour_sin`, `hour_cos` |

Formula (0-based so first value maps to angle 0):

```
day_of_week_sin = sin(2π × (day_of_week − 1) / 7)
day_of_week_cos = cos(2π × (day_of_week − 1) / 7)
hour            = integer part of time string
hour_sin        = sin(2π × hour / 24)
hour_cos        = cos(2π × hour / 24)
```

The original `day_of_week` and `time` columns are dropped from the output.

---

### Step 2 — log1p for right-skewed columns

Before z-scoring, `log1p(x)` is applied to every column in `ZSCORE_COLS`
where the **train** skewness > 2 **and** train minimum ≥ 0. This compresses
the heavy right tail so the z-score is not dominated by rare extreme values.

12 columns met the threshold on the current run: `first_road_number`,
`second_road_number`, `aadf_traffic_volume`, `aadf_distance_m`,
`precipitation_mm`, `snowfall_cm`, `snow_depth_m`, `hist_acc_rate_500m`,
`hist_fatal_rate_500m`, `dist_to_nearest_hist_acc_m`, `n_casualties`,
`n_pedestrian_casualties`.

The threshold is computed at runtime from the actual train distribution, so
it self-updates if the input data changes.

---

### Step 3 — Z-score standardization

All 17 columns in `ZSCORE_COLS` (post log1p where applicable) are
standardized to mean 0 / SD 1 using train statistics:

```
x_norm = (x − mean_train) / sd_train
```

Columns normalized (binary flags, metadata, target, cyclic columns, and
nominal codes are excluded):

`number_of_vehicles`, `first_road_number`, `speed_limit`,
`second_road_number`, `aadf_traffic_volume`, `aadf_distance_m`,
`dist_to_speed_camera_m`, `lane_count`, `temperature_2m_c`,
`precipitation_mm`, `wind_speed_10m_kmh`, `snowfall_cm`, `snow_depth_m`,
`hist_acc_rate_500m`, `hist_severe_frac_500m`, `hist_fatal_rate_500m`,
`dist_to_nearest_hist_acc_m`, `n_casualties`, `n_pedestrian_casualties`,
`youngest_casualty_age`, `oldest_casualty_age`.

**Nominal codes are deliberately NOT z-scored** (`NOMINAL_COLS`:
`first_road_class`, `road_type`, `junction_detail`, `junction_control`,
`second_road_class`, `pedestrian_crossing`, `urban_or_rural_area`,
`light_conditions`, `weather_conditions`, `weather_code`). They pass
through as raw integer codes so XGBoost can use native categorical splits
(`enable_categorical=True` + shared pandas `category` dtype in
`train_classifier.py`) instead of treating unordered codes as if they had
a numeric order.

A zero-SD guard fires a hard error if any column is constant on train —
such a column carries no information and should be removed, not silently
left as NaN.

---

### Output: `data/splits/normalized/`

| File | Description |
|---|---|
| `train_norm.csv` | Normalized training split (57,732 rows) |
| `valid_norm.csv` | Normalized validation split (12,367 rows) |
| `test_norm.csv` | Normalized test split (12,382 rows) |
| `normalization_stats.csv` | Per-column mean, SD, and `log1p_first` flag — apply these to any new data at inference time |

> **Columns not normalized** (passed through unchanged):  
> metadata (`collision_index`, `collision_year`, `longitude`, `latitude`,
> `date`, `data_year`), target (`collision_severity`), binary flags
> (`has_*`, `has_speed_camera`, `has_traffic_calming`, `has_traffic_signals`,
> `has_median`, `trunk_road_flag`), and the string column `aadf_match_type`.

---

## 13. Masked-Copy Generation for Partial-Observability Training — `generate_masked_splits.R`

The model needs to perform well when only some STATS19 fields have been
acquired at inference time (a "999 call" scenario where the responder
doesn't yet know e.g. `junction_detail` or `second_road_class`). To train
for this, every row in each split is expanded into multiple copies with
varying amounts of feature masking.

Reads from `data/splits/normalized/` (output of step 12); masking is applied
to already-normalized values so masked-out NAs carry no unnormalized scale.

### Column categories
- **`XBASE_COLS`** (10, never masked — always available at call time):
  `collision_index`, `collision_year`, `longitude`, `latitude`, `date`,
  `data_year`, `collision_severity`, `day_of_week_sin`, `day_of_week_cos`,
  `hour_sin`, `hour_cos` (the raw `day_of_week` and `time` columns were
  replaced by their cyclic encodings in step 12).
- **`ACQUIRABLE_COLS`** (32, candidates for masking): `number_of_vehicles`,
  `first_road_class`, `first_road_number`, `road_type`, `speed_limit`,
  `junction_detail`, `junction_control`, `second_road_class`,
  `second_road_number`, `pedestrian_crossing`, `urban_or_rural_area`,
  `trunk_road_flag`, `light_conditions`, the 15 `has_*` vehicle-type flags,
  and the 4 casualty columns from step 10 (`n_casualties`,
  `n_pedestrian_casualties`, `youngest_casualty_age`,
  `oldest_casualty_age`) — ask-the-dispatcher fields, maskable like the
  vehicle-type flags.
- All other columns (AADF, OSM, Open-Meteo, risk-density features,
  `weather_conditions`) are left untouched in every copy — they're out of
  scope for this call-time partial-observability simulation, either because
  they're external-source features (their own acquisition cost governs
  availability) or, for risk-density, because they're free/always-available
  base features.

### Method (per row, independently)
1. Sample 10 distinct integers from `0:32` without replacement — these are
   the mask counts assigned to `copy_id` 0–9.
2. For each copy, randomly pick exactly that many of the 32 acquirable
   columns and set them to `NA`.
3. The original, fully-observed row is kept as `copy_id = -1`,
   `mask_rate = 0.0`.
4. Every copy gets a `mask_rate` column = `masked_count / 32` (0.0 for the
   original, otherwise matches the sampled count for that copy).

Each original row therefore becomes 11 rows (1 unmasked + 10 masked), all
columns preserved, seed 42 for reproducibility.

### Output: `data/splits/masked/{train,valid,test}_masked.csv`
| Split | Original rows | Expanded rows |
|---|---|---|
| train | 62,828 | 691,108 |
| valid | 13,458 | 148,038 |
| test  | 13,493 | 148,423 |

Note: `mask_rate` reflects only the fields *deliberately* masked by this
script. Some acquirable fields already had genuine NAs in the source data
(e.g. `junction_control` 44.6%, `second_road_number` 40.3% missingness from
section 3) — if a deliberately-masked field happened to already be NA, it
doesn't change the row's actual NA count beyond what `mask_rate` records as
intentional.

### Companion stats: `data/splits/masked/mask_stats/`
Computed directly from the same mask matrix used to generate the files
(not by re-inspecting NAs in the output, which would be confounded by the
pre-existing NAs above):

- **`{split}_mask_count_distribution.csv`** / **`combined_mask_count_distribution.csv`**
  — for each mask count 0–32, how many masked copies have exactly that many
  fields masked (count + % of all masked copies). Roughly uniform, as
  expected from sampling without replacement across 33 possible values.
- **`{split}_feature_mask_frequency.csv`** / **`combined_feature_mask_frequency.csv`**
  — for each of the 32 acquirable fields, how many times it was masked
  across all masked copies and the %. All land around ~50%, since each
  masked copy masks roughly half the acquirable fields on average.

---

## 14. Next steps (not yet done)

- Decide treatment for `light_conditions` (replace with an external API at
  inference, or keep the STATS19 value) — `weather_conditions` is now
  superseded by the Open-Meteo features above.
- ~~Casualty-derived features~~ — done, see section 10.
- Extend `YEARS` further back (e.g. to ~2010) to grow Fatal further, following
  the same recipe as the current 2016 extension — untested, diminishing
  returns likely as accident patterns drift over a longer window.
- De-bias `hist_severe_frac_500m` by the known per-year sampling fractions
  (currently accepted as-is; see section 9).
- ~~Re-tune XGBoost hyperparameters for the current 54-column feature set~~ —
  done. Re-ran `tune_full_acquisition_classifier.py` (60 trials 3-class, 100
  trials binary) on the current feature set; the search converged to
  essentially the same params as before (old config wasn't actually stale),
  but retraining all 8 subset classifiers with `--use-best-params` still
  lifted the flagship (aadf+osm+meteo) 3-class test macro F1 from 0.4845 to
  0.4928 uncalibrated (0.503 → 0.5028 calibrated), driven mostly by the
  weak "Serious" class (F1 0.339 → 0.39). Binary flagship macro F1 stayed
  flat (0.651 → 0.6517).
