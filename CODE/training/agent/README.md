# Active Feature Acquisition Agent

An agent that decides *which information to acquire* before predicting collision severity, trading prediction accuracy against the real time cost of acquiring each piece of information.

Task: **normalized** data variant, **binary** Slight vs. Severe. Raw/masked variants and the 3-class task are not supported here.

> This file describes how the agent works and where things live. **Measured results are not kept here** — they go stale and get quoted out of date. The numbers live in `docs/figures/` (`lambda_curve/`, `baselines/`, `bundle_sweep/`) and in the report.

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
| **β** `--terminal-weight` | value of a correct prediction | **No, provably.** Scaling β, α and λ together scales every Q-value equally and leaves `argmax` unchanged, so β only fixes units. Leave at 1. `sweeps/tune_rewards.py --scale-check` is the experiment that demonstrates it. |
| **λ** `--cost-scale` | price of one second of acquisition time, *in accuracy points* | **Yes — this is the real knob.** |
| **α** `--step-weight` | shaping bonus for confidence gained | **Barely.** Summed over an episode the ΔConfidence terms telescope to `α·(final − initial confidence)`, so α reduces to a bonus on the final state. Do not grid-search it. |

So there is really **one** parameter: **λ**.

**What λ means.** Every acquisition is a trade — "spend 2 seconds asking the caller about casualties, gain some accuracy; worth it?" You cannot answer that without an exchange rate between seconds and accuracy points. λ *is* that exchange rate, in accuracy points per second.

- A low λ makes a second cheap, so even the slow ask-the-caller bundles pay for themselves.
- A high λ makes a second expensive, and only the millisecond lookups survive.

λ is **not** a value to be maximised — it is a deployment decision about how much a 999 call-handler's time is worth. Each λ yields a different, equally correct policy.

---

## Quick start

```bash
# 1. What each bundle is worth (analysis only, no training)
python CODE/training/agent/voi/compute_voi.py && python CODE/training/agent/voi/plot_voi.py

# 2. Train
python CODE/training/agent/train_rl_agent.py \
  --warm-start --cost-scale 0.01 --conf-bins 5 \
  --gamma 1.0 --n-train 12000 --n-eval 2500 \
  --output-dir models/rl_agents/main

# 3. Sweep the knobs
python CODE/training/agent/sweeps/optuna_tune.py      # the learning knobs
python CODE/training/agent/sweeps/tune_rewards.py     # lambda: the price of time

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

The `clip` matters only at confidence exactly 1.0, which would otherwise index a bin that doesn't exist.

**How many bins** is a real trade-off: too few and the agent cannot tell an easy call from a hard one (1 bin *is* the mask-only agent); too many and the 1024×bins states each get too few visits to learn anything. **5 is the default** in both `QLearner` and `--conf-bins`, selected by the Optuna search over the range 2–12.

Differences across that range sit inside seed-to-seed variation, so the setting is not critical and should not be reported as a tuned optimum. `--conf-bins 1` is outside the searched range and is used deliberately as the mask-only ablation.

---

## Seeding the Q-table

Q-learning from an all-zero table **does not work on this problem**. A good policy is a specific multi-bundle sequence, the only strong feedback arrives at the end of the episode, and ε-greedy essentially never produces that sequence by chance — cold-start runs plateau after a single bundle.

`--warm-start` fills the table first, using `one_step_seed()` from [core/seeding.py](core/seeding.py):

```
Q[stop]      = β · score(m)                    what stopping here is worth
Q[acquire i] = β · score(m ∪ {i}) − Cᵢ         what one more bundle is worth
```

Those scores are the measured **validation** accuracies the bundle sweep already recorded — seeding from a test column would leak the test set into the agent's starting point.

This is a **one-step lookahead, not a plan**: it prices one bundle at a time and says nothing about sequences, combinations, or when to stop, which is exactly the part the agent still has to learn. And the 1,024 classifiers *are* the environment the agent queries, so reading their known scores uses the environment's own structure rather than handing the agent a solution.

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

**[sweeps/optuna_tune.py](sweeps/optuna_tune.py)** — the main tuner. TPE search over `lr`, `epsilon`, `conf_bins` and `step_weight`; the same method §3.1 uses for the classifier. A grid tests a handful of hand-chosen points and cannot see between them or model interactions — the best `lr` at 3 bins need not be the best at 10, since 3,072 states versus 12,288 changes how often each is visited.

```bash
python CODE/training/agent/sweeps/optuna_tune.py --n-trials 40 --seeds 2
tail -f models/rl_agents/optuna/run.log     # one line per trial
```

Resumable: trials persist to `models/rl_agents/optuna/optuna.db` as they complete, and re-running the same command continues rather than restarting.

**[sweeps/tune_rewards.py](sweeps/tune_rewards.py)** — grid over the reward weights, plus `--scale-check`, the experiment demonstrating β is redundant.

**[sweeps/tune_learning.py](sweeps/tune_learning.py)** — grid over confidence bins and episode budget, when you want the whole curve rather than an optimum.

### The objective, and what is deliberately not tuned

```
score = accuracy − λ · mean acquisition seconds
```

**Accuracy**, matching the agent's own terminal reward, so hyperparameters are selected against the same quantity the agent is trained on. Macro F1 is computed and printed for every trial but is not optimised. On this near-balanced binary task the two agree closely, so the choice is nearly free. It would **not** be free on the 3-class task, where Fatal is rare enough that accuracy can be maximised by neglecting it.

**Minus the time**, because quality rises monotonically with acquisitions: the highest-quality policy is always "acquire all ten," which is guaranteed rather than discovered. Rank on quality alone and the tuner selects hyperparameters that make the agent ignore λ, defeating the point of active acquisition.

**The α·ΔConfidence shaping term is excluded** even though it is part of the agent's reward, because `step_weight` is itself being tuned — including it would hand free objective to high-α trials and make them incomparable. Shaping is a training device, not a deployment goal.

**λ is held fixed, not tuned.** It is the price of call-handler time — a deployment decision, where every value is a valid operating point. Hand it to an optimiser maximising quality and it drives λ to zero, "discovering" that free information is best. **β is held at 1**, being provably redundant. **γ = 1**: episodes cap at 10 steps and only the final mask determines the score.

### Read the noise before the winner

Each trial is one training run scored on sampled episodes, and the terminal reward is 0/1 — a noisy objective. A sampler handed noise will confidently report whichever configuration drew luckiest. Every trial therefore averages over multiple seeds, and the summary prints the winner's own seed-to-seed spread beside its lead over the median, with a warning when the lead is smaller. If that fires, the honest conclusion is *"these hyperparameters barely matter"* — which is a result, not a failure.

Seed variance on this agent is larger than any hyperparameter effect measured, so **no single run is a measurement.** Report from `analysis/final_model.py`, which averages seeds, never from one run.

One caveat that applies to any tuning run: to fit 40 trials in a few hours, trials use fewer training episodes than a final model would. The optimum found in an under-trained regime need not be the optimum at full budget, so **retrain the winning configuration at the full episode count before quoting its numbers.**

---

## Two bugs worth knowing about

**Rewards were disconnected (fixed 1 Sept 2026).** `MDPEnvironment.step()` never passed the learner's α, β or bundle costs into the reward functions — it used the method defaults (α=1, β=1, **all costs 0**). Every reward-tuning experiment before that date measured nothing; identical results across configurations were the symptom. Discard any tuning conclusion predating the fix.

**The state ignored `p_t` (fixed 6 Sept 2026).** The MDP formulation defines the state as `(x_free, x_acquired, m_t, p_t)`, but the Q-table was keyed on `m_t` alone — the environment computed the confidence and the learner discarded it. That is why the policy was identical for every collision.

---

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

Only the call timestamp is free. **Bundle order is the bitmask bit order** — reordering, or changing any bundle's contents, silently invalidates every cached model in `models/subsets/`.

Costs are **operational difficulty** (how hard the data is to obtain), never derived from how much accuracy a bundle delivers — that would be circular, and was removed once already. They live in [core/bundle_costs.py](core/bundle_costs.py) in seconds; λ converts them to reward units.

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
│   ├── baselines.py            the agent against the non-adaptive policies
│   ├── plot_baselines.py       figures for that comparison
│   ├── convergence.py          how many episodes the agent actually needs
│   ├── seed_vs_trained.py      does training improve on its own seed
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

A browser simulator for inspecting the policy on single calls lives in `CODE/simulator/` — `python CODE/simulator/app.py`. It runs the real classifiers and the saved policies, and has its own README.

### The baselines

`analysis/baselines.py` scores the agent against four non-adaptive policies, each given **the same acquisition time the agent spent at that λ**, so every comparison holds cost constant:

| baseline | how it spends the budget |
|---|---|
| no acquisition | nothing; predicts from the call timestamp alone |
| full acquisition | all ten bundles |
| best fixed subset | the highest-scoring of the 1,024 subsets that fits the budget |
| static priority | a fixed order — marginal value ÷ acquisition time — stopping when the budget runs out |
| random acquisition | the ten bundles shuffled, taken until the next would exceed the budget, averaged over many shuffles |

The subset and the ordering are both selected on **validation** and reported on **test**, so no baseline uses information the agent did not also have.

Two things to know before reading its output. Static priority's accuracy is a **step function** of the budget — its prefixes cost 0.23 s, then 1.23 s, then 3.23 s and nothing between — so a small change in the agent's mean time can swing that comparison sharply; `baselines.csv` carries `static_matched_seconds` as the accuracy-matched reading, which does not have that sensitivity. And `--skip-lambdas` omits some λ from the figures by default; `--skip-lambdas none` restores the full table.

---

## Design decisions worth not re-litigating

**Dynamic programming over the subset lattice** was removed 6 Sept 2026 (recoverable from commit `d7c3d4e`). Brute-force enumeration is both simpler and strictly better: sweeping λ through a DP is a Lagrangian relaxation that only recovers the *convex hull* of the cost/accuracy frontier, so it reached 9 of the 68 Pareto-optimal subsets. Enumeration finds all 68.

**A confidence-threshold baseline** — same acquisition order as static priority, stopping once the classifier is sure enough — was built and then removed in October 2026 at the author's request. It is the only baseline that isolates *adaptivity*, since it differs from static priority in the stopping rule alone. Without it the baselines answer whether the agent picks a good acquisition *set*, not whether adapting per collision is worth anything.

**Several attempts to lift the agent off its ceiling were tried and dropped**, all within seed noise: longer training, ε-decay, conditional seeding, seeding from the best fixed subset, validation checkpointing, fitted Q iteration, a dense terminal reward, and pruning the negative-value bundle. Between them they rule out more data and more representational capacity as the limiting factor. Check the report before re-running any of them.

---

## External references

- MDP formulation — `docs/data_pipeline.md` §5
- Pre-trained subset models — `CODE/training/train/bundle_sweep.py`
- Bundle definitions — `CODE/training/train/train_classifier.py` (`BUNDLE_COLS`, `BUNDLE_NAMES`)
