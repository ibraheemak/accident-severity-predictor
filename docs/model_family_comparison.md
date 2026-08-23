# Model-family comparison

**Question:** how much of the classifier's score comes from XGBoost specifically, and
how much is simply what the data supports?

**Setup.** Every family is trained and scored on exactly the same thing: the
`normalized` data variant, binary Slight-vs-Severe target, full acquisition
(`aadf + osm + meteo`, 52 features), the same train/valid/test split, balanced class
weighting wherever the family supports it, 3 seeds (42/43/44), reported on the test
split. Gradient boosters use their own native categorical and missing-value handling;
every other family gets median-imputed + z-scored numerics and most-frequent-imputed +
one-hot categoricals. Reproduce with:

```bash
python CODE/experiments/model_comparison/compare_model_families.py --seeds 42 43 44
python CODE/experiments/model_comparison/tune_model_families.py --trials 120   # equal-budget tuning
python CODE/experiments/model_comparison/plot_model_families.py      # figures only, from the saved CSVs
```

## Results

| Model | Family | Macro F1 | ±sd | Balanced acc. | Severe recall | AUC | Fit time |
|---|---|---|---|---|---|---|---|
| **XGBoost** | boosting | **0.7367** | 0.0010 | 0.7370 | 0.731 | **0.8119** | 1.82 s |
| CatBoost | boosting | 0.7300 | 0.0017 | 0.7302 | 0.730 | 0.8068 | 9.06 s |
| HistGradientBoosting | boosting | 0.7300 | 0.0027 | 0.7302 | 0.727 | 0.8054 | 0.71 s |
| LightGBM | boosting | 0.7269 | 0.0000 | 0.7270 | 0.727 | 0.8059 | 2.83 s |
| Random forest | ensemble | 0.7245 | 0.0008 | 0.7244 | 0.736 | 0.7986 | 2.24 s |
| Neural net (MLP 128-64) | ensemble | 0.7244 | 0.0028 | 0.7247 | 0.723 | 0.8003 | 1.22 s |
| Extra trees | ensemble | 0.7200 | 0.0017 | 0.7200 | 0.728 | 0.7930 | 1.93 s |
| Linear SVM | linear | 0.7176 | 0.0000 | 0.7180 | 0.708 | 0.7913 | 0.47 s |
| Logistic regression | linear | 0.7168 | 0.0000 | 0.7171 | 0.710 | 0.7917 | 0.25 s |
| Linear regression (OLS, thresholded) | linear | 0.7165 | 0.0000 | 0.7169 | 0.705 | 0.7908 | 0.13 s |
| k-NN (k=25) | simple | 0.7082 | 0.0000 | 0.7087 | 0.692 | 0.7742 | 0.06 s |
| Decision tree (depth 5) | simple | 0.6707 | 0.0000 | 0.6707 | 0.695 | 0.7298 | 0.19 s |
| Decision tree (unpruned) | simple | 0.6232 | 0.0011 | 0.6232 | 0.634 | 0.6232 | 0.53 s |
| **One-rule (stump)** | baseline | **0.6047** | 0.0000 | 0.6102 | 0.512 | 0.6102 | 0.10 s |
| Gaussian naive Bayes | simple | 0.5748 | 0.0000 | 0.5930 | 0.818 | 0.6966 | 0.08 s |
| **Random (stratified)** | baseline | **0.5032** | 0.0084 | 0.5033 | 0.516 | 0.5033 | 0.06 s |
| **Majority class** | baseline | **0.3403** | 0.0000 | 0.5000 | 1.000 | 0.5000 | 0.07 s |

Full per-seed numbers: `docs/figures/model_families/model_families_results.csv`.

## What it shows

**1. Against a proper baseline, the model earns +0.132.** Three reference points bracket
the problem: predicting the majority class scores 0.3403, predicting at random by the
class prior scores 0.5032, and the best single feature-threshold rule (a decision stump,
i.e. Holte's OneR) scores 0.6047. The tuned classifier reaches 0.7367 — **+0.132 macro F1
over the one-rule baseline and +0.231 over random**, with AUC going 0.610 → 0.812. That is
the comparison to quote: it measures what the model contributes over a non-learned
decision rule of the kind a dispatcher could apply from a laminated card.

**1b. Among learning algorithms, the spread is small — and that is the interesting part.**
Once a model is fitting the full engineered feature set, the family barely matters:
ordinary least squares thresholded at 0.5 reaches 0.7165, within 0.020 of the tuned
booster. The engineered features (cyclic time encodings, log1p'd and z-scored numerics,
and especially the risk-density features that carry local historical fatality rates)
have already absorbed most of the nonlinearity, leaving little interaction structure for
a booster to discover. The pipeline, not the algorithm, is doing the work — which is a
result about the feature engineering, and worth stating plainly rather than hiding.

**2. Every capable family lands in a 0.72–0.73 band.** Four gradient boosters, a random
forest and a neural net are separated by 0.010 macro F1, and the ROC curves (fig 2) are
visually indistinguishable. That is the signature of an *information ceiling* — the
data, not the model, is the binding constraint. It also reproduces an earlier sweep run
on a different dataset configuration, where the same families clustered the same way.

**3. XGBoost's win over the other learners is real but small.** +0.0067 over the next
family (CatBoost) at library defaults, with a seed sd of ~0.002, and it is best on AUC too
(0.8119 vs 0.8068). But see finding 5: once every family is given an equal tuning budget
the margin shrinks to +0.0015 over LightGBM, which is inside seed noise.

**4. The simple models fail in informative ways.** The unpruned decision tree (0.623) is
*worse* than one capped at depth 5 (0.671): with 52 features it memorises the training
split. Gaussian naive Bayes gets the highest Severe recall of any model (0.818) purely by
over-predicting Severe, which drops its macro F1 to 0.575 — a reminder that recall alone
is not evidence of a good model on these severity-oversampled splits.

**5. With an equal tuning budget, XGBoost still leads — but only just.** The table above
runs every family at library defaults, which flatters the deployed model, since XGBoost alone
carries hyperparameters from this project's Optuna searches. `tune_model_families.py`
removes that asymmetry: 120 random-search trials per family (60 for CatBoost, which trains
~10× slower), each trial trained on `train` and scored on `valid`, with the winner
retrained over 3 seeds and reported on `test` — which is touched only at that final step.
XGBoost's project-tuned configuration is injected as trial 0 of its own search, so it
competes as one candidate among 120 and is selected by the same validation rule as
everything else. It won that selection.

| Family | Tuned macro F1 | ±sd | AUC | vs default |
|---|---|---|---|---|
| **XGBoost** (project-tuned config won its own search) | **0.7367** | 0.0010 | **0.8119** | +0.0000 |
| LightGBM | 0.7352 | 0.0013 | 0.8110 | +0.0083 |
| CatBoost | 0.7338 | 0.0016 | 0.8080 | +0.0037 |
| HistGradientBoosting | 0.7333 | 0.0001 | 0.8071 | +0.0033 |
| Extra trees | 0.7283 | 0.0016 | 0.8041 | +0.0083 |
| Random forest | 0.7280 | 0.0003 | 0.8018 | +0.0035 |
| Neural net (MLP) | 0.7276 | 0.0004 | 0.8051 | +0.0032 |
| Logistic regression | 0.7172 | 0.0000 | 0.7909 | +0.0004 |
| Linear SVM | 0.7168 | 0.0000 | 0.7913 | −0.0008 |
| k-NN | 0.7125 | 0.0000 | 0.7870 | +0.0044 |
| Decision tree | 0.6982 | 0.0000 | 0.7725 | **+0.0275** |
| Gaussian naive Bayes | 0.5850 | 0.0000 | 0.6997 | +0.0102 |

Three things fall out of this. **(a)** The ranking is essentially unchanged — tuning moves
scores, not order. **(b)** Tuning helps the weak models most: the decision tree gains
+0.0275 and naive Bayes +0.0102, while the linear models gain nothing (+0.0004 and
−0.0008), because they were already at their ceiling. **(c)** XGBoost's lead over the
field narrows from +0.0067 to **+0.0015 over LightGBM**, which is inside the ±0.002 seed
noise. The defensible claim is therefore that **the top four boosters are statistically
indistinguishable once all are tuned**, and XGBoost is retained on the strength of a
consistent-but-small edge plus its 11× faster training than tuned LightGBM.

**6. Caveat on the absolute numbers.** These splits are deliberately severity-enriched
(~52% Severe here vs ~21% in the real population, and Fatal is ~53% of Severe here vs
~7% in reality). Every score above is measured on that benchmark, so it is a comparison
between models, not an estimate of field performance. See `CLAUDE.md` for the
dataset-configuration history and `CODE/experiments/evaluate_population.py` for the
population-accuracy reconstruction.

## Figures

| File | Shows |
|---|---|
| `fig1_macro_f1.png` | Test macro F1 by model family, sorted, seed error bars |
| `fig2_roc.png` | ROC curves, five highest-AUC families + chance line |
| `fig3_pr.png` | Precision–recall curves, Severe class + no-skill baseline |
| `fig4_time_vs_score.png` | Training time by model family |
| `fig5_metric_panels.png` | Macro F1, balanced accuracy and AUC by model family |
| `fig6_confusion.png` | Row-normalised confusion matrices, four highest-scoring families |
| `fig7_tuning_gain.png` | Library defaults vs 120-trial tuning, by model family |

All figures are 200 dpi PNG in `docs/figures/model_families/` and are regenerable from
the saved CSVs without retraining.
