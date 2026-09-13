# Active Feature Acquisition Agent

An agent that decides *which information to acquire* before predicting collision severity, trading prediction accuracy against the real time cost of acquiring each piece of information.

Task: **normalized** data variant, **binary** Slight vs. Severe. Raw/masked variants and the 3-class task are not supported here.

---

## The MDP

- **State** — the acquisition mask (which of the 10 bundles are held) **and the classifier's current confidence**, `max(p_t)`, discretised into bins. The confidence term is what lets two collisions holding the same bundles be treated differently.
- **Actions** — `0` predict & terminate, or `1..10` acquire bundle `action-1`.
- **Transitions** — deterministic. Acquiring bundle *i* sets bit *i* and loads the pre-trained classifier for the new mask from `models/subsets/normalized_binary/`.
- **Reward** —

  ```
  per acquisition step :   α · ΔConfidence  −  λ · seconds_i
  on predict & terminate:  β · Accuracy(ŷ, y)
  ```

### The three reward parameters, and which one actually matters

| | what it scales | changes the optimal policy? |
|---|---|---|
| **β** `--terminal-weight` | value of a correct prediction | **No — measured.** Scaling β, α and λ together leaves every Q-value scaled equally and `argmax` unchanged. Verified: three configurations at k×(0.05, 1, 0.01) for k ∈ {0.5, 1, 2} gave accuracy 0.6980 and 8.25 bundles, *identical to four decimals*. β only fixes units. Leave at 1. |
| **λ** `--cost-scale` | price of one second of acquisition time, *in accuracy points* | **Yes — this is the real knob.** |
| **α** `--step-weight` | shaping bonus for confidence gained | **Barely — measured.** Summed over an episode the ΔConfidence terms telescope to `α·(final − initial confidence)`, so α reduces to a bonus on the final state. Sweeping α ∈ {0, 0.05, 0.2, 1.0} moved accuracy by 0.015 total, non-monotonically (0.7133 / 0.6980 / 0.7040 / 0.7107) — inside single-seed noise. α = 0 scored highest, so the shaping term may be droppable entirely. |

So there is really **one** parameter: **λ**.

**What λ means.** Every acquisition is a trade — "spend 2 seconds asking the caller about casualties, gain 9.6 accuracy points; worth it?" You cannot answer that without an exchange rate between seconds and accuracy points. λ *is* that exchange rate, in accuracy points per second.

- `λ = 0.01` → a second costs 1 percentage point. The 2-second casualties question buys +9.6 points for 2 points of cost: clearly worth it.
- `λ = 0.1` → a second costs 10 points. That same question costs 20 points to gain 9.6: a bad trade. Only the millisecond-cheap lookups survive.

λ is **not** a value to be maximised — it is a deployment decision about how much a 999 call-handler's time is worth. Each λ yields a different, equally correct policy. Sweep it with `sweeps/tune_rewards.py`.

---

## Quick start

```bash
# 1. What each bundle is worth (analysis only, no training)
python CODE/training/agent/voi/compute_voi.py && python CODE/training/agent/voi/plot_voi.py

# 2. Train
python CODE/training/agent/train_rl_agent.py \
  --warm-start --cost-scale 0.01 --step-weight 0.05 --conf-bins 5 \
  --gamma 1.0 --n-train 12000 --n-eval 2500 \
  --output-dir models/rl_agents/main

# 3. Sweep the knobs
python CODE/training/agent/sweeps/tune_rewards.py     # lambda: the price of time
python CODE/training/agent/sweeps/tune_learning.py    # confidence bins, episode budget

# Or run everything
LAMBDA=0.01 bash CODE/training/agent/run_all.sh
```

---

## Discretising the confidence

`p_t` is a continuous probability, but a lookup table needs keys you can land on more than once — `Q[(19, 0.7231947)]` would be created, used once, and never revisited, so nothing would ever be learned. `max(p_t)` is therefore binned ([core/ql_solver.py](core/ql_solver.py), `conf_bin`):

```python
lo = 0.5                                    # two-class max-prob floor
frac = (confidence - lo) / (1.0 - lo)       # rescale [0.5,1.0] -> [0,1]
int(np.clip(frac * n_bins, 0, n_bins - 1))  # -> bin index
```

With 3 bins: 0.50–0.67 → bin 0, 0.67–0.83 → bin 1, 0.83–1.00 → bin 2. The `clip` matters only at confidence exactly 1.0, which would otherwise index a bin that doesn't exist.

**How many bins.** A real trade-off: too few and the agent cannot tell an easy call from a hard one (1 bin *is* the mask-only agent); too many and the 1024×bins states each get too few visits to learn anything — at 12,000 episodes of ~7 steps, 5 bins gives ~16 visits per state and 20 bins ~4.

`b` was included in the Optuna search ([sweeps/optuna_tune.py](sweeps/optuna_tune.py)) over the range 2–12, which selected **5** and attributed 26% of the objective variance to it — second only to the learning rate. **5 is the default** in both `QLearner` and `--conf-bins`.

Differences across the searched range are within seed-to-seed variation (±0.018), so the setting is not critical and should not be reported as a tuned optimum. `--conf-bins 1` is outside the searched range and is used deliberately as the mask-only ablation.

---|---|---|---|
| 1 (mask only) | 1,024 | 0.7112 | ~94 |
| **3** | **3,072** | **0.7316** | ~31 |
| 5 | 5,120 | 0.7048 | ~19 |
| 8 | 8,192 | 0.7276 | ~12 |
| 12 | 12,288 | 0.7228 | ~8 |
| 20 | 20,480 | 0.7132 | ~5 |

**3 is the default** in both `QLearner` and `--conf-bins`. The decline from 8 → 12 → 20 is the visit-count effect; the 5-bin row breaking that trend is single-seed noise, so the 3-vs-8 ordering is not solid without repeated seeds.

---

## Seeding the Q-table

Q-learning from an all-zero table **does not work on this problem**. A good policy is a specific multi-bundle sequence, the only strong feedback arrives at the end of the episode, and ε-greedy essentially never produces that sequence by chance. Cold-start runs plateau after a single bundle at 0.597 accuracy.

`--warm-start` fills the table first, using `one_step_seed()` from [core/seeding.py](core/seeding.py):

```
Q[stop]      = β · score(m)                    what stopping here is worth
Q[acquire i] = β · score(m ∪ {i}) − Cᵢ         what one more bundle is worth
```

Those scores are the measured validation accuracies the bundle sweep already recorded. This is a **one-step lookahead, not a plan**: it prices one bundle at a time and says nothing about sequences, combinations, or when to stop — which is exactly the part the agent still has to learn. And the 1,024 classifiers *are* the environment the agent queries, so reading their known scores uses the environment's own structure rather than handing the agent a solution.


`QLearner.warm_start()` only loads values; it contains no logic for producing them, so any seed source drops in.

---

## Tuning

Two vocabularies collide here, so to be explicit:

| | what it is | flag |
|---|---|---|
| `lr` | the **Q-learning rate** — how far each experience moves an estimate | `--lr` |
| `step_weight` | the **report's α** — weight on the ΔConfidence shaping term | `--step-weight` |

Both were called `alpha` until September 2026. They are unrelated; `QLearner.load()` still accepts the old `"alpha"` key so earlier saved models open.

### Which script to use

**[sweeps/optuna_tune.py](sweeps/optuna_tune.py)** — the main tuner. TPE search over `lr` (0.005–0.5, log), `epsilon` (0.02–0.40), `conf_bins` (2–12) and `step_weight` (0–1). Same method §3.1 uses for the classifier. A grid tests a handful of hand-chosen points and cannot see between them or model interactions — the best `lr` at 3 bins need not be the best at 10, since 3,072 states versus 12,288 changes how often each is visited.

```bash
python CODE/training/agent/sweeps/optuna_tune.py --n-trials 40 --seeds 2
tail -f models/rl_agents/optuna/run.log     # one line per trial
```

Resumable: trials persist to `models/rl_agents/optuna/optuna.db` as they complete, and re-running the same command continues rather than restarting.

**[sweeps/tune_rewards.py](sweeps/tune_rewards.py)** — grid over the reward weights, plus `--scale-check`, which is the experiment proving β is redundant.

**[sweeps/tune_learning.py](sweeps/tune_learning.py)** — grid over confidence bins and episode budget, when you want the whole curve rather than an optimum.

### The objective, and what is deliberately not tuned

```
score = accuracy − λ · mean acquisition seconds
```

**Accuracy**, matching the agent's own terminal reward, so hyperparameters are selected against the same quantity the agent is trained on. Macro F1 is computed and printed for every trial — it is what the report quotes — but is not optimised. On this near-balanced binary task the two agree to 0.0001 across all 1,024 subsets, so the choice is nearly free. It would **not** be free on the 3-class task, where Fatal is rare enough that accuracy can be maximised by neglecting it.

**Minus the time**, because quality rises monotonically with acquisitions: the highest-F1 policy is always "acquire all ten," which is guaranteed rather than discovered. Rank on quality alone and the tuner selects hyperparameters that make the agent ignore λ, defeating the point of active acquisition.

**The α·ΔConfidence shaping term is excluded** even though it is part of the agent's reward, because `step_weight` is itself being tuned — including it would hand free objective to high-α trials and make them incomparable. Shaping is a training device, not a deployment goal.

**λ is held fixed, not tuned.** It is the price of call-handler time — a deployment decision, where every value is a valid operating point. Hand it to an optimiser maximising quality and it drives λ to zero, "discovering" that free information is best. **β is held at 1**, being provably redundant. **γ = 1**: episodes cap at 10 steps and only the final mask determines the score.

### Read the noise before the winner

Each trial is one training run scored on sampled episodes, and the terminal reward is 0/1 — a noisy objective. A sampler handed noise will confidently report whichever configuration drew luckiest. Every trial therefore averages over multiple seeds, and the summary prints the winner's own seed-to-seed spread beside its lead over the median, with a warning when the lead is smaller. If that fires, the honest conclusion is *"these hyperparameters barely matter"* — which is a result, not a failure.

One caveat that applies to any tuning run: to fit 40 trials in a few hours, trials use fewer training episodes than a final model would. The optimum found in an under-trained regime need not be the optimum at full budget, so **retrain the winning configuration at the full episode count before quoting its numbers.**

---

## Results

Reference points from the 1,024-subset sweep: timestamp-only floor **0.5408**, best fixed subset **0.7396** (7 bundles), full acquisition **0.7374** test macro F1.

### Headline models — 3 seeds, 12,000 episodes, λ = 0.01

| config | accuracy | macro F1 | seconds | distinct sets |
|---|---|---|---|---|
| tuned | 0.7175 ± 0.0090 | 0.7136 | **4.83** | 13 |
| default (pre-tuning) | 0.7223 ± 0.0176 | 0.7186 | 5.54 | 10 |
| mask-only (`--conf-bins 1`) | 0.7155 ± 0.0036 | 0.7113 | **6.26** | **1** |

Reproduce with `python CODE/training/agent/analysis/final_model.py`.

**Adaptivity does not improve accuracy.** tuned − mask-only = **+0.0020** against a pooled sd of 0.0097 — inside noise. An earlier single-seed comparison reported 0.7392 vs 0.7112 and that gap did not survive repetition; **0.7392 was the top of a wide distribution, not a typical result** (the same configuration gave 0.7040 on the next seed).

**What adaptivity buys is time.** 4.83 s versus 6.26 s — **1.43 seconds less per call for the same accuracy**, at roughly 5σ. For a dispatch setting that is arguably the more valuable axis.

**Tuning bought nothing measurable.** tuned − default = −0.0048 against a pooled sd of 0.0198. Seed variance (±0.018) is larger than any hyperparameter effect we could find, so the agent is best described as *insensitive to its hyperparameters within the ranges tested*, with the one caveat that `lr` above ~0.15 is genuinely worse.

### The operating curve — 19 values of λ, 3 seeds each

```bash
python CODE/training/agent/analysis/lambda_curve.py                    # λ = 0 … 0.10
python CODE/training/agent/analysis/lambda_curve.py --lambdas 0.2 0.3 0.5 1 2 3 5 10 \
       --out-dir docs/figures/lambda_curve_log                         # λ = 0.2 … 10
python CODE/training/agent/analysis/plot_lambda_curve.py               # combine + figures
```

| λ | accuracy | seconds |
|---|---|---|
| 0 | 0.7111 | 6.16 |
| **0.01** | **0.7175** | **4.83** |
| 0.02 | 0.7123 | 3.17 |
| 0.05 | 0.6753 | 1.69 |
| 0.10 | 0.6488 | 0.75 |
| 1.0 | 0.6203 | 0.09 |
| 10 | 0.5533 | 0.003 |

**Acquisition time falls 2,000× while accuracy falls 0.16.** That trade-off, not any single number, is the result: λ is a deployment decision, so the report should present the menu and name the operating point it assumes.

Two features worth stating:

- **λ = 0.01 dominates λ = 0** — higher accuracy in *less* time. When time is free the agent spends it on bundles that make the classifier worse (`meteo` has negative marginal value, three others are near zero). A small time charge prunes them.
- **The knee is λ ≈ 0.02** — half the acquisition time of λ = 0 for 0.005 of accuracy. Past 0.03 accuracy drops in earnest.

Figures in `docs/figures/lambda_curve/`: `lambda_operating_curve.png`, `lambda_bundle_heatmap.png`, `lambda_vs_metrics.png`. The heatmap shows which bundles the agent abandons as λ rises — the three ask-the-caller bundles go first (demographics by 0.02, casualties by 0.03, vehicles by 0.2), the millisecond lookups survive to λ ≈ 0.3, and `osm` is the last standing at λ ≈ 5, matching its break-even of value-alone / seconds = 4.2.

**Caveat for any figure taken from this sweep:** points are means over 3 seeds and small non-monotonicities (the wobble around λ = 0.2–0.5) are inside run-to-run variation. Error bars were removed for legibility; the standard deviations remain in `lambda_curve_combined.csv`.

### Two bugs worth knowing about

**Rewards were disconnected (fixed 1 Sept 2026).** `MDPEnvironment.step()` never passed the learner's α, β or bundle costs into the reward functions — it used the method defaults (α=1, β=1, **all costs 0**). Every reward-tuning experiment before that date measured nothing; identical results across configurations were the symptom. Discard any tuning conclusion predating the fix.

**The state ignored `p_t` (fixed 6 Sept 2026).** The MDP formulation defines the state as `(x_free, x_acquired, m_t, p_t)`, but the Q-table was keyed on `m_t` alone — the environment computed the confidence and the learner discarded it. That is why the policy was identical for every collision.

### Current limitations

**Confidence is discretised.** A tabular value function needs discrete states, so two collisions at 0.71 and 0.79 confidence fall in the same bin and are treated identically. Removing that means function approximation (a network taking `p_t` raw), which costs the readable policy and needs far more data — not warranted at 1,024 masks.

**The learning signal is far too weak to find the optimal subset.** The terminal reward is 0/1, so a single visit to a state carries a standard deviation of 0.449. Distinguishing two subsets that differ by 0.01 in accuracy needs ~8,000 visits to that state; at 12,000 episodes over 1024×5 states the agent gets about **16**. It can easily tell "acquire nothing" from "acquire something" (a 0.20 gap) but cannot resolve differences among the *good* subsets (0.002–0.03). Consequently the agent largely inherits the greedy choices in its seed rather than improving on them, and lands ~0.028 below the best fixed subset at λ = 0.

**Seed variance dominates.** ±0.018 across seeds for the same configuration, which is larger than any hyperparameter effect measured. Any single-run number from this agent should be treated as one draw, not a measurement.

**Untested:** whether Q-learning adds anything over the seed at all. Evaluating the seeded policy with zero training episodes would settle it, and given the visit-count arithmetic above it is a live possibility that training is mostly inert.

## The 10 bundles

| # | Bundle | Kind | Seconds | Features |
|---|--------|------|---------|----------|
| 0 | vehicles | ask caller | 1.0 | vehicle count + 7 type flags |
| 1 | casualties | ask caller | 2.0 | counts by pedestrian/driver/passenger/cyclist |
| 2 | casualty_demographics | ask caller | 3.0 | ages, child/elderly/female counts |
| 3 | road_class | map lookup | 0.050 | road class & number, type, trunk flag |
| 4 | road_environment | map lookup | 0.050 | speed limit, urban/rural, light, weather, crossing |
| 5 | junction | map lookup | 0.040 | junction detail & control |
| 6 | crash_history | map lookup | 0.060 | prior-year rates within 500 m |
| 7 | aadf | external API | 0.020 | traffic flow, distance to count point |
| 8 | osm | external API | 0.010 | signals, calming, lanes, median, camera |
| 9 | meteo | external API | 0.030 | temperature, precipitation, wind, weather code |

Only the call timestamp is free. **Bundle order is the bitmask bit order** — reordering invalidates every cached model in `models/subsets/`.

Costs are **operational difficulty** (how hard the data is to obtain), never derived from how much accuracy a bundle delivers — that would be circular. They live in [core/bundle_costs.py](core/bundle_costs.py) in seconds; λ converts them to reward units.

---

## Layout

```
agent/
├── train_rl_agent.py     ← the entry point: train and evaluate one agent
├── run_all.sh            ← runs everything below in order
│
├── core/                 the agent itself (you rarely edit these)
│   ├── mdp_environment.py   state, transitions, rewards; caches the 1,024 classifiers
│   ├── ql_solver.py         tabular Q-learning over (mask, confidence bin)
│   ├── seeding.py           initial Q-values for the table (one_step_seed)
│   ├── bundle_costs.py      per-bundle acquisition time in seconds
│   └── test_mdp.py          smoke tests
│
├── sweeps/               finding good settings
│   ├── optuna_tune.py       TPE search over the learning knobs  ← the main one
│   ├── tune_rewards.py      grid over the reward weights α, β, λ
│   └── tune_learning.py     grid over confidence bins and episode budget
│
├── analysis/             understanding a trained agent
│   ├── final_model.py          headline numbers: full budget, 3 seeds, error bars
│   ├── lambda_curve.py         sweep λ — the cost/accuracy operating curve
│   ├── plot_lambda_curve.py    figures for that sweep
│   ├── analyze_policy.py       which bundles the policy acquires
│   ├── compare_strategies.py   the policy against fixed baselines
│   └── plot_strategies.py      figures for that comparison
│
└── voi/                  what each bundle is worth (no agent needed)
    ├── compute_voi.py
    ├── plot_voi.py
    └── README.md
```

Everything runs from the **repo root**, not from this directory:
`python CODE/training/agent/train_rl_agent.py …`

## Not yet built

**The Best Fixed Subset baseline.** *"Always acquire these five bundles"* — computable directly from the bundle sweep by filtering the 1,024 subsets to those within a time budget and taking the argmax. This is the baseline the agent must beat to justify its complexity, and until it exists the agent's accuracy has nothing to be judged against.

An earlier version computed a related frontier by dynamic programming over the subset lattice (removed 6 Sept 2026; recoverable from commit `d7c3d4e`). Brute-force enumeration is both simpler and strictly better: sweeping λ through a DP is a Lagrangian relaxation that only recovers the *convex hull* of the cost/accuracy frontier, so it reached 9 of the 68 Pareto-optimal subsets. Enumeration finds all 68.

## External references

- MDP formulation — `docs/data_pipeline.md` §5
- Pre-trained subset models — `CODE/training/train/bundle_sweep.py`
- Bundle definitions — `CODE/training/train/train_classifier.py` (`BUNDLE_COLS`, `BUNDLE_NAMES`)
