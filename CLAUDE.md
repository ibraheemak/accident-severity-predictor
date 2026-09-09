# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A UK road-accident severity predictor (STATS19 data). The core deliverable is a bank of frozen XGBoost classifiers — one per **acquisition subset** — trained to predict Fatal/Serious/Slight (or binary Slight-vs-Severe) from only what a 999 dispatcher would know. On top of that bank sits a Reinforcement Learning "Active Feature Acquisition" agent that decides at runtime which information is worth acquiring before invoking the matching frozen classifier — implemented in `CODE/training/agent/` (September 2026), see that folder's README.

**Action-space revision (August 2026) — the "8 frozen classifiers" design is superseded.** It was 8 classifiers, one per subset of the 3 external sources, trained on randomly masked STATS19 fields. Two things changed:
- **Masking is no longer used.** It cost accuracy at every missingness level versus a dedicated per-subset model, and made every headline number a pooled average across missingness levels. `generate_masked_splits.R` and the `masked` variant still exist and still run, but no current work uses them. Don't default to `--data-variant masked`.
- **STATS19 fields are now acquirable too**, since the 3 external sources carry very little of the signal. One action per field would mean 2^31 models, so features are grouped into **10 bundles** (`BUNDLE_COLS`/`BUNDLE_TIERS` in `train_classifier.py`), each = one real acquisition act, in three cost tiers: *ask_caller* (`vehicles`, `casualties`, `casualty_demographics`), *map_lookup* (`road_class`, `road_environment`, `junction`, `crash_history`), *external_api* (`aadf`, `osm`, `meteo`). Only the call timestamp is free. That gives 2^10 = 1024 classifiers, pre-trained by `CODE/training/train/bundle_sweep.py` into `models/subsets/<variant>_<task>/<mask>.json` — the RL environment's model cache.

**Bundle invariants (easy to break):** bundle order in `BUNDLE_COLS` *is* the bitmask bit order, and `bundle_feature_columns()` emits columns in that order. XGBoost binds features positionally, so **reordering bundles, or changing any bundle's contents, silently invalidates every cached model** — wipe `models/subsets/` and re-run the sweep. An import-time assert checks the bundles still cover `BASE_COLS + SOURCE_COLS + weather_conditions`, so a feature added to `BASE_COLS` without a bundle home fails loudly rather than dropping out of the action space.

**Current focus:** the RL Active Feature Acquisition agent over the 10-bundle action space, on the `normalized` (unmasked, clean one-row-per-collision) data variant, binary Slight-vs-Severe. Prefer changes/experiments that target this variant+task over generic changes to the masked/3-class models unless asked otherwise. The single-model reference point remains `models/classifier_aadf_meteo_osm_normalized_binary.json` (full-information, test macro F1 0.7367).

**RL agent status (6 September 2026) — read `CODE/training/agent/README.md` before touching it.** The agent lives in `CODE/training/agent/`: tabular Q-learning whose state is the acquisition mask **plus the classifier's current confidence** `max(p_t)`, discretised into bins. Four things are easy to get wrong:
- **There is only one meaningful reward parameter, λ (`--cost-scale`)** — the exchange rate pricing one second of acquisition time in accuracy points, and a deployment decision rather than something to maximise. β (`--terminal-weight`) is *provably* redundant: scaling α, β and λ together leaves the policy identical, verified to four decimals by `sweeps/tune_rewards.py --scale-check`. α (`--step-weight`) telescopes to `α·(final − initial confidence)` and moved accuracy by 0.015 across a 20× sweep, inside single-seed noise. Do not grid-search α/β.
- **Two different things were both called `alpha` until Sept 2026.** `lr` is the Q-learning rate; `step_weight` is the report's α. `QLearner.load()` still accepts the old `"alpha"` key.
- **Tune with `sweeps/optuna_tune.py`** (TPE, same method as the classifier's Optuna tune), not the grids. Objective is `accuracy − λ·seconds`: accuracy because it matches the agent's own terminal reward, minus time because quality rises monotonically with acquisitions so ranking on quality alone selects for "acquire everything". Macro F1 is reported but not optimised — the two agree to 0.0001 here, and would *not* on the 3-class task where Fatal is rare. Trials persist to SQLite and resume. Trials use a reduced episode budget, so **retrain the winner at full budget before quoting numbers.**
- **Acquisition costs are operational difficulty, never value delivered.** `bundle_costs.py` holds per-bundle seconds (call time / query latency). Deriving cost from a bundle's accuracy gain is circular and was removed once already.
- **Cold-start Q-learning does not work here** and plateaus after one bundle (0.597 vs 0.739 seeded, same λ) — ε-greedy almost never discovers a specific multi-bundle acquisition sequence from a 0/1 reward delivered only at episode end. `--warm-start` seeds from measured one-step subset values (`core/seeding.py`).
- **`p_t` must stay in the state.** Keying the Q-table on the mask alone makes every collision look identical, so the policy collapses to one fixed acquisition set for every call (1 distinct set vs 13). Fixed 6 Sept 2026; a reward-wiring defect (α/β/costs never reaching the reward function) invalidated every tuning result before 1 Sept 2026.
- **Report numbers from `analysis/final_model.py` (3 seeds), never a single run.** Seed variance is ±0.018 — larger than any hyperparameter effect. The widely-quoted 0.7392 was one lucky seed of a distribution whose mean is ~0.72; the same config gave 0.7040 on the next seed.

Best trained agent (3 seeds, 12k episodes, λ=0.01): **0.7175 ± 0.0090 accuracy in 4.83 s** of acquisition, against a timestamp-only floor of 0.5408 and a best fixed subset of 0.7396. **Adaptivity does not improve accuracy** (vs mask-only ablation: +0.0020, pooled sd 0.0097 — inside noise) but **saves 1.43 s per call** for the same accuracy (~5σ). The λ sweep (`analysis/lambda_curve.py`, 19 values, 3 seeds) is the headline result: acquisition time falls 2,000× while accuracy falls 0.16. **The Best Fixed Subset baseline is not yet built** — filter the 1,024-subset sweep to a time budget and take the argmax — so the agent's number currently has nothing rigorous to be judged against. An earlier DP-based planner was removed 6 Sept 2026 (commit `d7c3d4e`): brute-force enumeration over the sweep is simpler and strictly better, reaching all 68 Pareto-optimal subsets where the DP's λ-sweep reached 9.

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
    train/                    # train_classifier.py, calibrate_thresholds.py, bundle_sweep.py
    tune/                     # tune_full_acquisition_classifier.py (Optuna)
    agent/                    # RL Active Feature Acquisition agent (own README + EVALUATION.md)
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

Train one classifier (defaults to the `normalized`, 3-class variant):
```bash
python CODE/training/train/train_classifier.py --bundles vehicles casualties --binary
python CODE/training/train/train_classifier.py --mask 7 --binary --use-best-params
python CODE/training/train/train_classifier.py --bundles            # free-information model
python CODE/training/train/train_classifier.py --sources aadf osm meteo --binary   # legacy
```
The action space is selected by exactly one of three mutually exclusive flags:
- `--bundles` — current 10-bundle action space; any subset of `vehicles casualties casualty_demographics road_class road_environment junction crash_history aadf osm meteo`. Passing it with no values trains the timestamp-only model. Writes `classifier_bundles<mask:04d>_*.json`.
- `--mask` — the same thing as an integer bitmask 0..1023, which is the RL state encoding and the subset-cache key. Convenient for reproducing one subset from a sweep.
- `--sources` — **legacy** 3-external-source action space, kept because `CODE/experiments/` is written against it. Used (with all three sources) when none of the three flags is given.

`--data-variant` is `normalized` (default) | `masked` (deprecated, see above) | `raw` (unnormalized). `--use-best-params` loads Optuna-tuned hyperparameters from `models/best_hyperparams[_<variant>][_binary].json` if present, else falls back to the masked-variant tune.

Tune hyperparameters (writes `models/best_hyperparams*.json`):
```bash
python CODE/training/tune/tune_full_acquisition_classifier.py --n-trials 60 [--binary] [--data-variant normalized]
```

Pre-train the 1024 acquisition-subset classifiers into the RL model cache (resumable; ~35 min for a full sweep, ~2 s per fit):
```bash
python CODE/training/train/bundle_sweep.py                  # all 1024, binary + normalized
python CODE/training/train/bundle_sweep.py --max-bundles 1  # 11-model smoke test
python CODE/training/train/bundle_sweep.py --masks 0 1023   # specific subsets only
```
Writes `models/subsets/<variant>_<task>/<mask>.json` plus a per-subset metrics CSV in `docs/figures/bundle_sweep/`. Skips masks already cached unless `--force`.

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

**Masking is deprecated (August 2026) — the section below documents how it worked, not what to use.** Partial observability is now handled by training a dedicated classifier per acquisition subset (see the action-space revision above), not by masking. `generate_masked_splits.R` still runs as part of `run_pipeline.R` and the historical masked-variant results stand, but new work should not use `--data-variant masked`.

**How masking worked.** `generate_masked_splits.R` expands each accident into 11 copies (1 unmasked + 10 with 0–32 random "acquirable" fields blanked, per `ACQUIRABLE_COLS`). This is why the masked variant has ~11x the rows of `normalized`/`raw`, and why headline masked-variant metrics are a pooled average across missingness levels — see `data/raw` size counts and `docs/data_pipeline.md` §13 before drawing conclusions from a masked-variant number alone.

**Three data variants exist for a reason, not by accident:** `raw` (untransformed) and `normalized` (cyclic-encoded, log1p'd, z-scored, unmasked) isolate whether normalization matters; `masked` isolates whether partial observability matters. A hyperparameter tune or feature change validated on one variant does not automatically transfer to another — the project's own history includes tuned hyperparameters that were optimal for `masked` but measurably worse than defaults on `normalized` (heavier regularization was needed for the noisier 691k-row masked set than the clean 62k-row normalized set). When evaluating a change, check the specific variant you actually care about — which is now `normalized`, since masking is deprecated.

**Class balance is deliberately non-natural — currently very much so.** `stats19_download.R` keeps all Fatal collisions but caps Serious/Slight per year (`SERIOUS_CAP`/`SLIGHT_CAP`, currently 1,400/2,800 — see the dataset-difficulty history note above), so every train/valid/test split is severity-oversampled relative to the real population (currently ~52% Severe in the data vs ~21% in reality, and Fatal is ~53% of Severe here vs ~7% in reality — far more skewed than the original 2,800/5,600-cap config). Macro F1 and balanced accuracy are the metrics to trust; plain accuracy on these splits is systematically *lower* than field accuracy would be, and can be gamed upward by a model that just leans toward the majority class. `experiments/evaluate_population.py` reconstructs true population accuracy from the per-year sampling fractions in `data/raw/stats19/stats19_<TIMESTAMP>/*_meta.txt` if you need that number — always report it alongside minority-class recall, since natural accuracy alone hides how many severe accidents get missed.
