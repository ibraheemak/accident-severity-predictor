# Value-of-Information Sweep

What each acquisition bundle is actually worth, measured across all 1,024 pre-trained subset classifiers.

This is an analysis of the model bank, not a training step. It needs `CODE/training/train/bundle_sweep.py` to have been run to completion first (`docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv`, 1,024 rows).

```bash
python CODE/training/agent/voi/compute_voi.py    # tables
python CODE/training/agent/voi/plot_voi.py       # figures
```

Everything lands in `docs/figures/voi_sweep/`.

---

## The central idea: marginal value

The obvious way to score a bundle is to ask what it achieves **alone** — train on just that bundle and see how much it beats the timestamp-only floor. That number is misleading for an acquisition decision, because you are never choosing bundles starting from nothing. You are always deciding whether to add *one more* to what you already hold.

So the number reported here is **marginal value**: how much the score improves when the bundle is added to a set you already have. That depends on which set, so it is averaged over **all 512 subsets that lack the bundle**.

The two can differ sharply. Adding `casualties` to nothing gains +0.0959; adding it to a set that already holds five other bundles gains only +0.0213. Averaged over every starting point it is worth **+0.0389**.

The gap between "alone" and "marginal" measures **redundancy**:

| Bundle | Alone | Marginal | |
|---|---|---|---|
| casualties | +0.0959 | +0.0389 | |
| road_environment | +0.0860 | +0.0157 | most of its signal is available elsewhere |
| meteo | +0.0031 | −0.0010 | fully redundant, and slightly harmful |

`road_environment` looks almost as strong as `casualties` in isolation but carries less than half the marginal value — once you hold a few other bundles it stops earning its place.

**Method note.** This is an *unweighted* mean over all subsets lacking the bundle, which is the **Banzhaf index**. It is a close relative of the better-known **Shapley value**, which weights subsets by size instead. The two usually rank features similarly but are not identical — cite it as Banzhaf-style, not Shapley.

---

## Outputs

| File | Contents |
|---|---|
| `bundle_value.csv` | per bundle: mean/std/min/max marginal value, value alone, fraction of subsets where it helps, acquisition seconds, value per second |
| `saturation.csv` | per subset size k: best / mean / worst achievable score, the best subset at that size, and the gain over k−1 |
| `tier_value.csv` | marginal value rolled up by acquisition kind |
| `voi_marginal.png` | marginal value per bundle, coloured by acquisition kind |
| `voi_saturation.png` | best achievable score against number of bundles held, with the worst-to-best band |
| `voi_cost_benefit.png` | marginal value against acquisition cost in seconds (log x) |

`--metric` selects the score column (default `test_macro_f1`; `valid_macro_f1`, `test_accuracy`, `valid_accuracy` also available).

---

## What the current run found

Floor (timestamp only) **0.5408** · full acquisition **0.7374** · whole action space worth **+0.1966** test macro F1.

**Caller information dominates.** Mean marginal value by kind: ask-the-caller **+0.0340**, map lookup **+0.0087**, external API **+0.0016** — a 3.9× and 20.9× gap. The three external sources that formed the *entire original action space* are the least valuable third of the current one, which is what motivated widening it to the STATS19 fields.

**Value saturates at five bundles.** `vehicles + casualties + casualty_demographics + road_class + road_environment` reaches 0.7374 — matching full acquisition. Past that, successive best-at-size gains are +0.0009, +0.0013, −0.0001, −0.0024, +0.0004: two of the five are negative, and all are within retrain noise.

**`meteo` is not merely redundant but harmful**: −0.0010 mean marginal, helping in only 28% of subsets. The exact planner in `../plan_optimal_policy.py` independently drops it even when acquisition is free, which is the same finding by a different route.

**Value and cost are positively correlated.** The three most valuable bundles are also the three slowest to acquire, by two orders of magnitude (seconds vs milliseconds). There is no free lunch in the action space, which is precisely why the λ exchange rate has to be an explicit choice rather than an assumption — see `../README.md`.

---

## Two things done deliberately

**Selection happens on validation, reporting on test.** `compute_voi.py` picks the "best subset" by the validation column and reports its test score. Taking the argmax of the test column instead would be selecting on the test set: it returns a different subset with a flattering score (0.7396) that means nothing. Marginal *values* are reported on test because describing what a bundle is worth is not a selection.

**Costs never come from accuracy.** The seconds in `../bundle_costs.py` are operational difficulty — call time, query latency. Deriving them from a bundle's measured value would be circular, since the entire question is whether a bundle's cost buys enough accuracy to justify paying it.

---

## Figures

Colours are categorical slots 1–3 from the project's validated palette (all-pairs CVD ΔE 9.2, normal-vision ΔE 24.0 in light mode). The aqua slot falls below 3:1 contrast on the chart surface, so every mark carries a direct label as the required relief — do not remove the value labels or the bundle names when editing.

Titles name what is plotted rather than assert what it shows; the findings belong in the report prose and figure captions.
