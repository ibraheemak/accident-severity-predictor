# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A UK road-accident severity predictor (STATS19 data). The core deliverable is 8 frozen XGBoost classifiers (one per subset of 3 optional external data sources: AADF traffic, OSM road infrastructure, Open-Meteo weather), trained to predict Fatal/Serious/Slight (or binary Slight-vs-Severe) from only what a 999 dispatcher would know. The eventual goal (per README) is a Reinforcement Learning "Active Feature Acquisition" agent that decides at runtime which of the 3 external sources are worth acquiring before invoking the matching frozen classifier — the classifiers here are the environment that agent will sit on top of, not yet implemented.

**Current focus:** improving the macro F1 of the binary Slight-vs-Severe classifier on the `normalized` (unmasked, clean one-row-per-collision) data variant — i.e. `models/classifier_aadf_meteo_osm_normalized_binary.json`. Prefer changes/experiments that target this specific model+variant over generic changes to the masked/3-class models unless asked otherwise.

**Dataset-difficulty history (17 July 2026) — read before comparing any macro F1 across dates.** Macro F1 on this task is primarily a function of how Fatal-heavy the Severe class is (Fatal collisions are easy; Serious-vs-Slight is the genuinely hard boundary), which `SERIOUS_CAP`/`SLIGHT_CAP` in `stats19_download.R` directly control. Three configs have been tried:
  1. **2,800/5,600 caps → 89,791 collisions, 36% Fatal-of-Severe → test macro F1 ≈ 0.6967.** The original config; superseded below, but still the "moderate" reference point and what most historical experiment notes in this repo (feature ablations, model-family sweeps, ensemble results) were measured against.
  2. **25,000/22,400 caps → 415,550 collisions, 6.6% Fatal-of-Severe (near the real ~7% population rate) → test macro F1 ≈ 0.6704.** Tried and reverted same day — see `data/archive_2026-07-17_scaleup_415k/` / `models/archive_2026-07-17_scaleup_415k/` (~1.9 GB, gitignored) and branch `scale-up-415k` git history if you want it back. A `cap_sweep.py`-style experiment (script not committed; logic: subsample archived splits per-year, all Fatal + capped Serious/Slight) traced macro F1 across 7 cap configs on this data and found it moves **monotonically with Fatal-of-Severe share, not with dataset size** — bigger datasets scored *worse* here because every added row was a hard Serious case.
  3. **1,400/2,800 caps → 51,989 collisions, 53% Fatal-of-Severe → test macro F1 = 0.7322 (current, active).** Built 17 July 2026 by subsampling the archived 415k data (not a fresh download) down to half the original caps. This is the currently active dataset/model. Its higher score is bought by Fatal enrichment, not better modelling or new information — AUC actually *fell* slightly relative to config 1 in the cap-sweep's own numbers at this size trend. `experiments/evaluate_population.py`'s population-accuracy reconstruction becomes more important, not less, at this config: the *reported* macro F1 is measured on a benchmark roughly 8× more Fatal-heavy than reality.

The 89,791-config data/models are archived in `data/archive_2026-07-17_90k_default/` / `models/archive_2026-07-17_90k_default/` (gitignored, uncommitted) if you need to restore config 1 as active. **Never compare a macro F1 across these three configs without naming which one it's from** — 0.70 means something different in each.

## Repo layout

```
CODE/
  run_pipeline.R              # orchestrator: source()s every step below, in order
  downloading/                # step 1: STATS19 download + severity-aware sampling
  processing/
    stats19/                  # steps 2,3,10: STATS19-only feature builders
    aadf/, osm/, meteo/       # steps 5,6,7: per-source external joins
    build_combined_features.R, add_risk_density_features.R,
    split_train_valid_test.R, normalize_splits.R,
    generate_masked_splits.R  # steps 4,9,11,12,13: source-agnostic pipeline stages
    qa/                       # standalone diagnostic scripts, not part of run_pipeline.R
  training/
    train/                    # train_classifier.py, calibrate_thresholds.py
    tune/                     # tune_full_acquisition_classifier.py (Optuna)
  experiments/                # analysis/ablation scripts, one-off, not part of the pipeline
data/
  raw/{stats19,aadf,osm,openmeteo}/   # one subfolder per source
  features/stats19+aadf+osm+...+cas/  # sequential pipeline-stage tables (name = sources folded in so far)
  splits/{,normalized,masked}/        # train/valid/test at each transform stage
models/                       # trained model .json + metrics .json (gitignored)
docs/data_pipeline.md         # the authoritative, very detailed pipeline reference — read before touching any processing/ script
```

**Path resolution rule (important, and easy to get wrong):** every `data/...`/`models/...` string in every R and Python script is resolved relative to the **process's current working directory**, not the script's own location. Always run everything from the repo root. `run_pipeline.R`'s `source()` calls and Python's `from train_classifier import ...` follow the same rule — see below.

## Commands

Python (repo has a `.venv`; activate it first):
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Train a classifier (defaults to the masked, 3-class variant):
```bash
python CODE/training/train/train_classifier.py --sources aadf osm meteo
python CODE/training/train/train_classifier.py --sources aadf osm meteo --binary
python CODE/training/train/train_classifier.py --sources aadf osm meteo --data-variant normalized --use-best-params
```
`--sources` accepts any subset of `aadf osm meteo` (empty = base-only model). `--data-variant` is `masked` (default, partial-observability training) | `normalized` (clean, one row per accident) | `raw` (unnormalized). `--use-best-params` loads Optuna-tuned hyperparameters from `models/best_hyperparams[_<variant>][_binary].json` if present, else falls back to the masked-variant tune.

Tune hyperparameters (writes `models/best_hyperparams*.json`):
```bash
python CODE/training/tune/tune_full_acquisition_classifier.py --n-trials 60 [--binary] [--data-variant normalized]
```

Calibrate decision thresholds on an already-trained model (no retraining):
```bash
python CODE/training/train/calibrate_thresholds.py --sources aadf osm meteo [--binary] [--data-variant normalized]
```

Run the R data pipeline end-to-end (edit `START_FROM` at the top of the script to resume partway through):
```bash
Rscript CODE/run_pipeline.R
```
Or run a single stage directly, e.g. `Rscript CODE/processing/osm/add_osm_features.R --debug` (most processing scripts support `--debug` for a fast ~50-row smoke test).

There is no test suite, linter, or CI config in this repo.

## Architecture notes that span multiple files

**The R pipeline is a strict linear chain of CSV-in/CSV-out stages**, each named after the sources folded in so far: `stats19` → `stats19+aadf` → `stats19+aadf+osm` → `...+meteo` → `...+risk` → `...+risk+cas`, then split into `train/valid/test.csv`, then `normalized/*_norm.csv`, then `masked/*_masked.csv`. `run_pipeline.R`'s `steps` list is the index of this chain — if you add/reorder/rename a stage script, update that list (paths there are relative to repo root, e.g. `CODE/processing/osm/add_osm_features.R`, not to `CODE/`).

**`PIPELINE_DATA_DIR`** is how `run_pipeline.R` passes the current STATS19 download's location to the collision/vehicle/casualty feature builders — those scripts check `if (!exists("PIPELINE_DATA_DIR"))` and fall back to a hardcoded default when run standalone. When adding a new script that needs the raw STATS19 dir, follow this pattern rather than hardcoding a path.

**`train_classifier.py` is the shared library every other Python script imports from** (`feature_columns()`, `load_split()`, `prepare_xy()`, `unify_categorical_dtypes()`, `evaluate()`, `MODEL_DIR`, `SEVERITY_LABELS`/`BINARY_LABELS`, `DATA_VARIANT_DIRS`). It defines the feature set via `BASE_COLS` (always-on) + `SOURCE_COLS[source]` (added when that source is acquired) — this is the single place to add/remove a model feature. Since Python resolves `from train_classifier import ...` via the importing script's own directory (not CWD), every script outside `training/train/` (i.e. everything in `training/tune/` and `experiments/`) has a `sys.path.insert(0, ...)` shim at the top pointing at `training/train/` — keep this pattern if you add new scripts elsewhere, or co-locate the script in `training/train/` instead.

**Masking simulates partial dispatch-time information.** `generate_masked_splits.R` expands each accident into 11 copies (1 unmasked + 10 with 0–32 random "acquirable" fields blanked, per `ACQUIRABLE_COLS`). This is why the masked variant has ~11x the rows of `normalized`/`raw`, and why headline masked-variant metrics are a pooled average across missingness levels — see `data/raw` size counts and `docs/data_pipeline.md` §13 before drawing conclusions from a masked-variant number alone.

**Three data variants exist for a reason, not by accident:** `raw` (untransformed) and `normalized` (cyclic-encoded, log1p'd, z-scored, unmasked) isolate whether normalization matters; `masked` isolates whether partial observability matters. A hyperparameter tune or feature change validated on one variant does not automatically transfer to another — the project's own history includes tuned hyperparameters that were optimal for `masked` but measurably worse than defaults on `normalized` (heavier regularization was needed for the noisier 691k-row masked set than the clean 62k-row normalized set). When evaluating a change, check the specific variant you actually care about, not just `masked` by default.

**Class balance is deliberately non-natural — currently very much so.** `stats19_download.R` keeps all Fatal collisions but caps Serious/Slight per year (`SERIOUS_CAP`/`SLIGHT_CAP`, currently 1,400/2,800 — see the dataset-difficulty history note above), so every train/valid/test split is severity-oversampled relative to the real population (currently ~52% Severe in the data vs ~21% in reality, and Fatal is ~53% of Severe here vs ~7% in reality — far more skewed than the original 2,800/5,600-cap config). Macro F1 and balanced accuracy are the metrics to trust; plain accuracy on these splits is systematically *lower* than field accuracy would be, and can be gamed upward by a model that just leans toward the majority class. `experiments/evaluate_population.py` reconstructs true population accuracy from the per-year sampling fractions in `data/raw/stats19/stats19_<TIMESTAMP>/*_meta.txt` if you need that number — always report it alongside minority-class recall, since natural accuracy alone hides how many severe accidents get missed.
