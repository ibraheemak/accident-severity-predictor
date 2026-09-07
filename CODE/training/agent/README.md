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
  --warm-start --cost-scale 0.01 --step-weight 0.05 --conf-bins 3 \
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

**How many bins.** A real trade-off: too few and the agent cannot tell an easy call from a hard one (1 bin *is* the mask-only agent); too many and the 1024×bins states each get too few visits to learn anything. Swept at 12,000 training episodes / 2,500 eval:

| bins | states | accuracy | visits per state |
|---|---|---|---|
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

## Results

Reference points from the 1,024-subset sweep: timestamp-only floor **0.5408**, full acquisition **0.7374** test macro F1.

Trained agents, 2,500 eval episodes at λ = 0.01, 12,000 training episodes:

| state | accuracy | bundles | distinct acquisition sets |
|---|---|---|---|
| mask only (`--conf-bins 1`) | 0.7112 | 10.00 | **1** |
| mask + confidence (`--conf-bins 3`) | **0.7392** | 7.93 | 14 |

The `distinct sets` column is the point. With the mask alone the policy used **one** acquisition set for all 2,500 collisions — a fixed shopping list. With confidence in the state it used many, stopping early on calls the classifier was already sure about and pushing further on ambiguous ones. Higher accuracy *and* fewer questions.

Episode length tracks difficulty as it should: episodes stopping at 4–5 bundles are right ~93% of the time; those running to all 10 are the genuinely hard ones, right ~60% of the time.

### Two bugs worth knowing about

**Rewards were disconnected (fixed 1 Sept 2026).** `MDPEnvironment.step()` never passed the learner's α, β or bundle costs into the reward functions — it used the method defaults (α=1, β=1, **all costs 0**). Every reward-tuning experiment before that date measured nothing; identical results across configurations were the symptom. Discard any tuning conclusion predating the fix.

**The state ignored `p_t` (fixed 6 Sept 2026).** The MDP formulation defines the state as `(x_free, x_acquired, m_t, p_t)`, but the Q-table was keyed on `m_t` alone — the environment computed the confidence and the learner discarded it. That is why the policy was identical for every collision.

### Current limitation

Confidence is **discretised**, because a tabular value function needs discrete states. Two collisions at 0.71 and 0.79 fall in the same bin and are treated identically. Removing that approximation means replacing the table with function approximation (a network taking `p_t` raw), which costs the interpretable policy and needs far more data — not warranted at 1,024 masks.

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
│   ├── tune_rewards.py      sweep λ, the price of time
│   └── tune_learning.py     sweep confidence bins and episode budget
│
├── analysis/             understanding a trained agent
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
