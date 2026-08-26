# Temporal Drift

**What this measures:** Do accident severity patterns change over time? We train on 2016-2017 STATS19 data and test on 2024 to examine whether a model's predictive power degrades across a 7-year gap. This simulates a scenario where deployed classifiers become stale and are never updated.

**Setup:** normalized variant, binary Slight-vs-Severe, Optuna-tuned hyperparameters, class-balanced sample weights. Compares against a control model trained on all years (mixed) and tested on 2024 data.

## Scripts (run from repo root, venv active)

| Script | Does |
|---|---|
| `temporal_drift_test.py` | Trains on 2016-2017, tests on 2024. Compares against control (mixed years). Outputs CSV tables of metrics. |
| `plot_temporal_drift.py` | Generates visualization from the saved CSVs. |

```bash
source .venv/bin/activate
python CODE/experiments/temporal_drift/temporal_drift_test.py
python CODE/experiments/temporal_drift/plot_temporal_drift.py
```

## Results (2026-08-25)

### Key Finding: Minimal Drift Over 7 Years

Training a classifier on 2016-2017 data and testing on 2024 data shows **small but measurable performance degradation:**

| Metric | Drift Model | Control (2024) | Delta |
|---|---|---|---|
| **Macro F1** | 0.7283 | 0.7440 | **-0.0157** |
| **Balanced Acc** | 0.7301 | 0.7449 | **-0.0148** |
| **ROC AUC** | 0.8018 | 0.8160 | **-0.0142** |
| Recall: Severe | 0.6674 | 0.7041 | **-0.0367** |
| Recall: Slight | 0.7929 | 0.7857 | +0.0071 |

**Interpretation:**
- General discrimination ability (AUC) declines ~1.4 percentage points over 7 years
- Recall for Severe class drops more (~3.7 points), suggesting the model becomes worse at catching serious accidents in newer data
- Recall for Slight class actually improves slightly, indicating the model may be drifting toward higher precision on negative examples
- Overall the drift is modest: **a 2016-2017 model remains reasonably useful on 2024 data**, though slightly less reliable than one trained on all years

### Performance Across Test Years

The visualization shows both the drift model (2016-2017) and control model (all years) evaluated on test data broken down by incident year:

| Year | Macro F1 | Balanced Acc | Notes |
|---|---|---|---|
| 2016 | 0.7343 | 0.7352 | Early training years |
| 2017 | 0.7431 | 0.7431 | |
| 2018 | 0.7331 | 0.7336 | |
| 2019 | 0.7246 | 0.7248 | |
| 2020 | 0.7083 | 0.7083 | COVID year? Lowest performance |
| 2021 | 0.7265 | 0.7267 | Recovery |
| 2022 | 0.7164 | 0.7175 | Dip again |
| 2023 | 0.7299 | 0.7306 | Recovery |
| **2024** | **0.7440** | **0.7449** | **Highest performance** |

**Key observations:**
1. **2020 is the low point** (Macro F1 = 0.7083), possibly due to COVID-related anomalies in traffic patterns, accident types, or data collection practices
2. **2024 actually has the best performance** on a mixed-year model, suggesting the underlying accident severity patterns may have become **easier to predict** in recent years rather than harder
3. Trend is U-shaped: dip in 2020-2022, recovery in 2023-2024
4. This explains why the 2016-2017 model doesn't catastrophically fail on 2024 data — they may share more pattern similarity than intermediate years

## Implications for the RL Agent

1. **Model freshness matters, but not urgently.** A 7-year-old model loses ~1.4% AUC. If the agent's production deployment window is 3-5 years before retraining, expect ~1% performance decay.

2. **Watch 2020-like years.** If COVID-style disruptions recur (major policy changes, infrastructure work, new enforcement patterns), your classifiers will need faster retraining.

3. **Recent data is easier.** The fact that 2024 is the easiest year to predict suggests that either:
   - The feature set has become more discriminative over time (improved data collection, new signals), or
   - Accident patterns have genuinely become more separable (fewer borderline cases)
   - This is good news: your deployed models won't be fighting an increasingly difficult problem

## Caveats

- **No exploration of feature drift.** We don't examine *which* features' distributions changed most. A follow-up could isolate feature-level drift (e.g., do vehicle types differ, are road conditions more variable?) to pinpoint what drives the small overall delta.
- **Class balance is not natural.** The dataset is ~52% Severe vs ~21% in reality. Year-to-year drift may be different in production, where Fatal/Serious ratios naturally shift.
- **Single 7-year gap.** We look at 2016-17→2024 as one experiment; a sliding-window analysis (2016→2017, 2016-17→2018, etc.) would show drift accumulation rates more clearly.
- **Evaluation on test split only.** We don't separately evaluate on train/valid/test to check whether the drift is in the distribution or just in unseen test examples.

## Outputs (`docs/figures/temporal_drift/`)

| File | Contents |
|---|---|
| `temporal_drift_results.csv` | One row: metrics for drift model (2016-17→2024) and control model on 2024 test |
| `temporal_drift_by_year.csv` | One row per year (2016-2024): test metrics for both models, broken down by test year |
| `fig2_by_year_trend.png` | Line plot: Macro F1, Balanced Acc., and AUC trend across all test years (2016-2024) |
