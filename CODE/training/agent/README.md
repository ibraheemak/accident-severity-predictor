# Active Feature Acquisition Agent

An agent that decides *which information to acquire* before predicting collision severity, trading prediction accuracy against the real time cost of acquiring each piece of information.

Task: **normalized** data variant, **binary** Slight vs. Severe. Raw/masked variants and the 3-class task are not supported here.

---

## The MDP

- **State** — the acquisition mask (which of the 10 bundles are held) plus the current classifier's predicted probabilities.
- **Actions** — `0` predict & terminate, or `1..10` acquire bundle `action-1`.
- **Transitions** — deterministic. Acquiring bundle *i* sets bit *i* and loads the pre-trained classifier for the new mask from `models/subsets/normalized_binary/`.
- **Reward** —

  ```
  per acquisition step :   α · ΔConfidence  −  λ · seconds_i
  on predict & terminate:  β · Accuracy(ŷ, y)
  ```

### The three reward parameters, and which one actually matters

This is the part that is easy to get wrong, so it is worth being precise.

| | what it scales | changes the optimal policy? |
|---|---|---|
| **β** `--terminal-weight` | value of a correct prediction | **No.** Scaling β, α and λ together by any constant leaves the policy identical — only ratios matter. Fix β = 1. |
| **λ** `--cost-scale` | price of one second of acquisition time, *in accuracy points* | **Yes — this is the real knob.** |
| **α** `--step-weight` | shaping bonus for confidence gained | Barely. Summed over an episode the ΔConfidence terms telescope to `α·(final − initial confidence)`, so α nudges the agent toward confident stopping states and helps exploration, but does not redefine the objective. Keep it small (0.05). |

So there is really **one** parameter: **λ**.

**What λ means.** Every acquisition is a trade — "spend 2 seconds asking the caller about casualties, gain 9.6 accuracy points; worth it?" You cannot answer that without an exchange rate between seconds and accuracy points. λ *is* that exchange rate, in units of *accuracy points per second*.

- `λ = 0.01` → a second costs 1 percentage point. The 2-second casualties question buys +9.6 points for 2 points of cost: clearly worth it. The agent acquires generously.
- `λ = 0.1` → a second costs 10 points. That same question now costs 20 points to gain 9.6: a bad trade. Only the millisecond-cheap map and API lookups survive.
- `λ = 0` → time is free; acquire anything that helps at all.

λ is **not** a value to be maximized — it is a deployment decision about how much a 999 call-handler's time is worth. Each λ yields a different, equally correct policy. `plan_optimal_policy.py` sweeps λ and hands you the menu; you pick the row whose time budget matches reality.

---

## Quick start

```bash
# 1. Solve the MDP exactly and see the accuracy/time menu
python CODE/training/agent/plan_optimal_policy.py

# 2. Train at your chosen lambda, warm-started from that exact solution
python CODE/training/agent/train_rl_agent.py \
  --warm-start --cost-scale 0.01 --step-weight 0.05 \
  --n-train 1500 --n-eval 1000 --output-dir models/rl_agents/warmstart

# Or run everything
LAMBDA=0.01 bash CODE/training/agent/full_evaluation.sh
```

---

## Plan-then-learn

The mask-level MDP is **fully known in advance**: transitions are deterministic, and the expected terminal reward of stopping at any mask is that subset classifier's accuracy — already measured for all 1024 masks by `bundle_sweep.py`. So it can be solved in closed form by dynamic programming over the subset lattice:

```
V(m) = max( score(m),                                  # stop and predict
            max over i∉m [ −λ·seconds_i + V(m ∪ {i}) ] )  # acquire i
```

`plan_optimal_policy.py` does exactly this and sweeps λ to produce the Pareto frontier. Planning uses **valid** metrics (choosing a subset on test would leak); the table reports **test** metrics for the chosen subsets.

**Two details that are easy to misread in this recursion.**

*Where did α·ΔConfidence go?* It is deliberately absent, for two reasons. Practically, ΔConfidence is per-collision (it depends on `X_row`), while the planner reasons about per-mask aggregates over the whole split — there is no single ΔConfidence for a mask transition. More fundamentally, it would not change the answer: summed over an episode the shaping terms telescope to `α·(Conf(m_T) − Conf(m_0))`, and `Conf(m_0)` is the same constant for every policy, so the term reduces to a bonus `α·Conf(m_T)` on the final state alone. That is potential-based shaping with Φ = confidence, which is policy-invariant at γ=1. α's job is guiding *exploration* during learning; the planner enumerates rather than explores, so it has no job there.

*Why is there no γ here, when `warm_start()` has one?* Because they answer different questions, and both are right for their own. The planner reports the true undiscounted frontier — only the final mask determines the score and acquisition order does not change it, so discounting would penalise longer paths for no reason. `QLearner.warm_start()` instead has to seed Q-values on the same scale the learner will bootstrap with (γ=0.99), or the seed and the first TD updates would fight each other. Consequence worth knowing: γ¹⁰ ≈ 0.90, so a full-depth path is seeded about 10% low, biasing the warm start slightly toward shorter acquisition sequences than the exact optimum.

`--warm-start` seeds the Q-table with that DP solution, so training refines a correct policy instead of trying to discover a specific 5-step acquisition path from noisy 0/1 terminal rewards. This is what took the agent from 0.597 to 0.70+ accuracy — cold-start Q-learning plateaus at one bundle because random exploration almost never stumbles onto a good multi-bundle sequence.

---

## Results

Exact frontier (`plan_optimal_policy.py`, planning on valid, reporting test):

| λ | bundles acquired | time | test F1 |
|---|---|---|---|
| 0.002 | 9 — all except `meteo` | 6.23 s | **0.7371** |
| 0.01 | vehicles, casualties, road_environment, junction, crash_history | 3.15 s | 0.7088 |
| 0.05 | vehicles, road_environment, junction, crash_history, osm | 1.16 s | 0.6855 |
| 0.1 | road_environment, crash_history, aadf | 0.13 s | 0.6381 |
| 1.0 | osm only | 0.01 s | 0.5828 |

Baseline (timestamp only, mask 0) is **0.5408** test F1. Full acquisition is 0.7374 — note the optimum **excludes `meteo`**, which is net harmful.

Trained agents (1000 eval episodes):

| setup | accuracy |
|---|---|
| broken rewards (pre-fix, see below) | 0.55 – 0.63 |
| fixed rewards, cold start | 0.597 |
| warm start, λ = 0.01 | **0.700** |
| warm start, λ = 0.002 | **0.712** |

### A bug worth knowing about

Until 1 Sept 2026 `MDPEnvironment.step()` never passed the learner's α, β or bundle costs into the reward functions — it silently used the method defaults (α=1, β=1, **all costs 0**). Every reward-tuning experiment before that date was measuring nothing; identical results across configurations were the symptom. Any tuning conclusion recorded before that fix should be discarded.

### Current limitation

The state is only the acquisition mask, so the learned policy is **the same for every collision** — it acquires one fixed subset every time. That is a well-chosen *static* subset, not active acquisition: the RL currently reproduces the planner rather than beating it, and 0.737 is the ceiling for any fixed subset.

To go beyond it the state needs the **current prediction confidence**, so the agent can stop early when the free-information model is already sure and keep acquiring when it is not. That per-collision adaptivity is the only route past 0.737, and it is the natural next piece of work.

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

Costs are **operational difficulty** (how hard the data is to obtain), never derived from how much accuracy the bundle delivers — that would be circular. They live in [bundle_costs.py](bundle_costs.py) in seconds, and λ converts them to reward units.

---

## Files

**Core**
- [mdp_environment.py](mdp_environment.py) — state, transitions, rewards; caches the 1024 classifiers class-side
- [ql_solver.py](ql_solver.py) — tabular Q-learning + `warm_start()` DP seeding
- [bundle_costs.py](bundle_costs.py) — per-bundle acquisition time in seconds

**Planning & training**
- [plan_optimal_policy.py](plan_optimal_policy.py) — exact DP solution, λ-sweep Pareto frontier ← *start here*
- [train_rl_agent.py](train_rl_agent.py) — train and evaluate one agent

**Analysis**
- [tune_rewards.py](tune_rewards.py) — λ sweep: trained agent vs exact planner
- [analyze_policy.py](analyze_policy.py) — which bundles the policy acquires, and their value
- [analyze_voi.py](analyze_voi.py) — comparison against fixed baselines
- [visualize_voi.py](visualize_voi.py) — plots
- [full_evaluation.sh](full_evaluation.sh) — runs all of the above in order

**Other**
- [test_mdp.py](test_mdp.py) — smoke tests
- [EVALUATION.md](EVALUATION.md) — workflow, interpretation, troubleshooting

---

## Design notes

1. **Tabular, not neural.** 1024 states × 11 actions fits in a dict; function approximation would only add error. This stops being true once confidence enters the state.
2. **Models are pre-trained, never fitted at runtime.** The environment only does inference, from a class-level cache keyed by mask.
3. **Planning uses valid, reporting uses test.** Selecting a subset on test would leak.
4. **Costs are exogenous.** Difficulty of acquisition, not value of acquisition.

## External references

- MDP formulation — `docs/data_pipeline.md` §5
- Pre-trained subset models — `CODE/training/train/bundle_sweep.py`
- Bundle definitions — `CODE/training/train/train_classifier.py` (`BUNDLE_COLS`, `BUNDLE_NAMES`)
