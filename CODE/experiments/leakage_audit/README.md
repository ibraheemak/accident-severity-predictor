# Leakage audit / full-STATS19 upper bound

*(This folder was `full_stats19/` until 24 Aug 2026; the `data/splits/full_stats19{,_v2}/`
output directories keep their original names.)*

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

The two models trained side by side are the **dispatch-time-only** model (`BASE_COLS` + AADF/OSM/meteo — everything a 999 dispatcher could know within
seconds of the call) and the **full-STATS19** model (the same plus every
pre-label officer-recorded field). The gap between them is the accuracy cost of
real-world deployability.

## Scripts (run from the repo root)

| Script | Does |
|---|---|
| `build_full_stats19_features.py` | Joins every extra pre-label STATS19 field onto a set of `*_norm.csv` splits, writing `*_full.csv`. Two high-cardinality/redundant fields (`generic_make_model`, `local_authority_highway`) are excluded — an ablation found them net-negative. |
| `full_stats19_upper_bound.py` | Trains dispatch-legit vs full-STATS19 side by side across several dataset sizes. |
| `tune_full_stats19_model.py` | 80-trial Optuna tune dedicated to the 84-feature set (it otherwise borrows the dispatch model's params). |
| `train_full_stats19_model.py` | Trains and **saves** the full model to `models/` (local-only — `models/` is gitignored repo-wide). |
| `plot_leakage_audit.py` | Scores both frozen models on the same test split and writes the comparison figures + `leakage_audit_results.json` to `docs/figures/leakage_audit/`. |

## Figures (`docs/figures/leakage_audit/`, regenerated 2026-08-24)

`fig1_leakage_gap.png` — the two models' test scores side by side.
`fig2_feature_budget.png` — where the +31 added features come from.

| Test metric | Dispatch-time (52 feat.) | Full STATS19 (83 feat.) | Gap |
|---|---|---|---|
| Macro F1 | 0.7369 | 0.7624 | **+0.0254** |
| Balanced accuracy | 0.7372 | 0.7624 | +0.0252 |
| ROC-AUC | 0.8117 | 0.8456 | +0.0338 |
| Severe recall | 0.7305 | 0.7705 | +0.0400 |

Both models use the same canonical tuned hyperparameters
(`models/best_hyperparams_normalized_binary.json`), so the gap is an
information gap, not a tuning artefact.

**Feature budget: 52 → 83, i.e. +31 features (+60%)** — 5 collision-record
fields, 1 police-attendance flag, 18 vehicle/driver, 7 casualty-detail. Every
one is officer-recorded after the fact; none is knowable when the 999 call
lands. 60% more features buy 3.5% relative macro F1.

## Key findings (1400/2800-cap config, 2026-07-20)

- **Full model: test macro F1 0.7624 / accuracy 0.7626** on the balanced
  benchmark, vs the dispatch model's 0.7369 — a **+0.025** ceiling.
  *(Originally measured at 0.7625 with 84 features on 2026-07-20; the model was
  retrained 2026-08-24 after `dist_to_speed_camera_m` was dropped from the
  dispatch feature set on 2026-07-21, so both models share the current
  definition. The dropped column changed the score by 0.0001.)*
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
