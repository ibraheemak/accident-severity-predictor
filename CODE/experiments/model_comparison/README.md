# Model-comparison experiments

Every script in this folder answers a version of the same question: **how much of the
headline score comes from XGBoost specifically, and how much is just what the data
supports?** They all train competing model families on the *same* split, features and
protocol as the deployment classifier, and report them on the same metrics.

These are **analysis scripts, not part of the pipeline** — nothing here writes to
`models/`, and none of them is invoked by `run_pipeline.R`. The prose write-up of the
results lives in [docs/model_family_comparison.md](../../../docs/model_family_comparison.md).

Common benchmark unless a script says otherwise: `normalized` data variant, binary
Slight-vs-Severe target, full acquisition (`aadf + osm + meteo`), balanced class weights
wherever the family supports them, 3 seeds (42/43/44), scored on `test`.

## Scripts

| Script | Does |
|---|---|
| `compare_model_families.py` | The main comparison: 17 families (boosters, ensembles, linear, k-NN, trees, naive Bayes) plus three baselines — majority-class, stratified-random, and a one-rule stump — at library defaults, except XGBoost which uses this project's Optuna params. Writes the results CSVs and calls `plot_model_families.py` for the figures. |
| `tune_model_families.py` | The fair-fight version: the same random-search budget (N trials, scored on `valid`) for **every** family, so nobody wins on tuning budget alone. XGBoost's canonical params are injected as trial 0 and compete like any other candidate. |
| `plot_model_families.py` | Figures only, regenerated from the saved CSVs — no retraining. Also imported by `compare_model_families.py`. |

## Running them

All paths inside these scripts resolve against the **process working directory**, so run
everything **from the repo root** with the venv active:

```bash
source .venv/bin/activate
```

```bash
# 1. main comparison — all families, 3 seeds (~5 min); writes CSVs + figures
python CODE/experiments/model_comparison/compare_model_families.py --seeds 42 43 44
python CODE/experiments/model_comparison/compare_model_families.py --quick     # 4 models, smoke test

# 2. equal-budget tuning (slow — 120 trials x every family)
python CODE/experiments/model_comparison/tune_model_families.py --trials 120
python CODE/experiments/model_comparison/tune_model_families.py --trials 30 --only XGBoost "Random forest"
python CODE/experiments/model_comparison/tune_model_families.py --reeval        # re-score saved winners, no search

# 3. figures only, from the CSVs step 1 or 2 already wrote
python CODE/experiments/model_comparison/plot_model_families.py
```

Order matters only between 1/2 and 3: `plot_model_families.py` reads the CSVs the other
two write.

Prerequisites: the R pipeline must have produced `data/splits/normalized/`, and
`models/best_hyperparams*.json` must exist for the XGBoost rows to use the project's tuned params.

## Outputs

All under `docs/figures/model_families/`:

| File | From |
|---|---|
| `model_families_results.csv` / `model_families_summary.csv` | `compare_model_families.py` — one row per (model, seed), and per-model means/sds |
| `tuned_results.csv` / `tuned_summary.csv` / `tuning_trials.csv` | `tune_model_families.py` |
| `fig1_macro_f1.png` … `fig7_tuning_gain.png` | `plot_model_families.py` |

## Reading the numbers

- **Report macro F1 and balanced accuracy, not plain accuracy** — the splits are
  deliberately severity-oversampled (see the class-balance note in `CLAUDE.md`).
- **Never compare a macro F1 across dataset-cap configs** without naming which one it is
  from; the current numbers are the 1,400/2,800-cap config (test macro F1 0.7367 for the
  full-information XGBoost reference model).
- The headline finding is the gap to the **one-rule stump baseline (+0.132)**, not the gap
  to the next-best family — among learning algorithms the spread is only ~0.02, because
  the engineered features have already absorbed most of the nonlinearity.
