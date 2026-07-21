# Full-STATS19 upper-bound experiment

**What this measures:** how much accuracy the project's *dispatch-time-only*
constraint costs. The deployment classifiers use only what a 999 dispatcher
could know at call time (`BASE_COLS` + the AADF/OSM/meteo sources). These
scripts build an **upper-bound** model that additionally uses every pre-label
STATS19 field — forensic vehicle/casualty detail, driver demographics, road
surface, whether police attended, etc. — to see the ceiling.

**These are reference models, NOT deployment candidates.** They use
post-collision / officer-recorded information no dispatcher has, and
`police_attended` is leakage-adjacent (attendance is itself driven by how
severe the collision looks). Never ship one of these.

## Scripts (run from the repo root)

| Script | Does |
|---|---|
| `build_full_stats19_features.py` | Joins every extra pre-label STATS19 field onto a set of `*_norm.csv` splits, writing `*_full.csv`. Two high-cardinality/redundant fields (`generic_make_model`, `local_authority_highway`) are excluded — an ablation found them net-negative. |
| `full_stats19_upper_bound.py` | Trains dispatch-legit vs full-STATS19 side by side across several dataset sizes. |
| `tune_full_stats19_model.py` | 80-trial Optuna tune dedicated to the 84-feature set (it otherwise borrows the dispatch model's params). |
| `train_full_stats19_model.py` | Trains and **saves** the full model to `models/` (local-only — `models/` is gitignored repo-wide). |

## Key findings (1400/2800-cap config, 2026-07-20)

- **Full model: test macro F1 0.7625 / accuracy 0.7627** on the balanced
  benchmark, vs the dispatch model's 0.7367 — a **+0.026** ceiling.
- **The gap is size-invariant** (~+0.027 across a 6× range of dataset
  sizes/severity mixes): the dispatch-time constraint is a fixed structural
  information cost, not something more STATS19 data would close. This supports
  the project's Active Feature Acquisition framing — the gap must be closed by
  *external* sources, not more STATS19 rows.
- **Dedicated tuning didn't help**: the 80-trial tune lost to the
  dispatch-tuned params (0.7610 vs 0.7625) — the 5th independent search on this
  project all won by the same params.
- On a **natural ~79%-Slight distribution** (clean split, no leakage) the full
  model reaches **~0.84 accuracy** — but with severe recall only ~0.375. That
  is the honest version of the "90%+ accuracy" reported by some published
  STATS19 studies: majority-class accuracy on imbalanced data (and, in at least
  one paper, SMOTE applied before the train/test split — a leakage artifact we
  reproduced separately). Report accuracy only alongside per-class recall.

The experiment output CSVs (`data/splits/full_stats19{,_v2}/`) are gitignored;
regenerate them with `build_full_stats19_features.py`.
