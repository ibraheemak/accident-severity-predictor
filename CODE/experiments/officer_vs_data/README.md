# Officer vs. Data Gap

**What this measures:** STATS19 carries a single categorical weather field
(`weather_conditions`) that the **attending officer** fills in after the fact,
from the roadside, from memory. Open-Meteo gives an **objective** hourly
reconstruction at the same coordinate and hour (temperature, precipitation,
wind speed, WMO weather code, snowfall, snow depth). This experiment compares
the two directly to quantify what replacing a subjective officer assessment with
objective sensor data is actually worth for predicting collision severity.

Benchmark: `normalized` variant, binary Slight-vs-Severe, the 1,400/2,800-cap
dataset (51,989 collisions — see CLAUDE.md's dataset-difficulty note before
comparing any macro F1 here against another date). Agreement statistics are
computed on the **raw** splits, where the objective columns are still in
physical units (mm, km/h).

These are **analysis scripts, not part of the pipeline** — nothing here writes
to `models/`, and none of them is invoked by `run_pipeline.R`.

## Scripts (run from the repo root, venv active)

| Script | Does |
|---|---|
| `weather_agreement.py` | Part 1 — do the two sources describe the same sky? Officer-category profiles against Open-Meteo, the officer × objective cross-tab, Cohen's κ on shared claims, mutual information with the target. |
| `weather_predictive_test.py` | Part 2 — which source predicts severity better? Matched XGBoost models differing **only** in the weather block, added to the full dispatch feature set. 5 seeds, paired per-seed deltas. |
| `plot_officer_vs_data.py` | Figures only, regenerated from the saved CSVs — no retraining. |

```bash
source .venv/bin/activate
python CODE/experiments/officer_vs_data/weather_agreement.py
python CODE/experiments/officer_vs_data/weather_predictive_test.py          # 5 seeds, ~4 min
python CODE/experiments/officer_vs_data/weather_predictive_test.py --quick  # 1 seed, smoke test
python CODE/experiments/officer_vs_data/plot_officer_vs_data.py
```

Order matters only for the plotting script, which reads the CSVs the other two write.

## Results (2026-08-25)

### 1. The two sources disagree about half the time

`fig1_agreement_matrix.png`

Officer weather field vs. Open-Meteo objective conditions at the same location
and hour, cross-tabulated (row %). Where both make the *same* claim, agreement
is poor:

| Claim | Officer rate | Sensor rate | Cohen's κ |
|---|---|---|---|
| Precipitation falling | 13.1% | 14.0% | **0.43** |
| High winds | 2.7% | 6.6% | **0.26** |

When an officer writes "Raining", Open-Meteo records dry conditions **35% of the
time**; only 20% of the sensor's windy hours get a high-wind note. Open-Meteo
also has no fog channel: **93% of "Fog or mist" collisions are logged as Dry**.

### 2. Neither source adds value to the full dispatch feature set

`fig4_model_deltas.png` — test scores (Macro F1, Balanced Acc., ROC AUC) with
and without each weather source, mean over 5 seeds, compared to the no-weather
baseline:

| Weather block | Δ Macro F1 | Δ Balanced Acc. | Δ ROC AUC |
|---|---|---|---|
| Officer field | +0.0002 | +0.00009 | +0.0003 |
| Open-Meteo | −0.0033 | −0.00135 | −0.0005 |
| Both | −0.0018 | −0.00084 | −0.0007 |

Every delta is at or below seed noise. Open-Meteo is consistently negative:
six additional columns of near-zero signal cost more in variance than they
return. The officer field adds nothing measurable.

## Conclusion

**Replacing the subjective officer assessment with objective sensor data is
worth nothing** for this task:

- Neither source improves predictions beyond the dispatcher's existing feature
  set (call timestamp, vehicle/casualty data, road/location context).
- Their disagreement (κ 0.26–0.43) shows they measure different things — hourly
  grid-cell average vs. conditions a human standing there and reporting from
  memory.
- The officer's field is not uniformly more predictive; it just looks that way
  on univariate severity rates because non-weather values ("Unknown"/"Other")
  encode reporting process, not sky conditions.
- On wind and fog, the subjective call tracks severity where the objective
  sensor is flat — swapping one in would lose information rather than gain it.

**For the RL acquisition agent:** the `meteo` bundle (an `external_api` action
with network latency risk) buys an effect indistinguishable from zero. Strong
candidate to be the first action the agent learns never to take.

## Outputs (`docs/figures/officer_vs_data/`)

| File | Contents |
|---|---|
| `agreement_officer_profile.csv` | Open-Meteo conditions per officer category |
| `agreement_crosstab.csv` | officer category × objective condition, counts |
| `agreement_binary.csv` | precipitation + high-wind agreement, κ, error directions |
| `agreement_mutual_info.csv` | mutual information with the binary target |
| `weather_model_results.csv` | one row per (weather block, seed) |
| `weather_model_summary.csv` | means + deltas vs. no-weather control |
| `fig1_agreement_matrix.png` | officer category × Open-Meteo condition, row % |
| `fig4_model_deltas.png` | test scores (3 metrics) vs. no-weather baseline |

## Caveats

- **Thresholds are stated, not tuned.** "Wet" is >0.1 mm/h and "high winds" is
  ≥29 km/h (Beaufort 5). Both are in the constants block of
  `weather_agreement.py` if you want to test sensitivity.
- **Neither source is treated as ground truth.** Both directions of every
  disagreement are reported separately.
- **Class balance is not natural.** Splits are severity-oversampled (~52%
  Severe vs ~21% in the population) — see CLAUDE.md.
