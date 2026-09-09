# RL Agent Evaluation Guide

How to pick λ, train against it, and check the result is actually optimal.

See [README.md](README.md) for the MDP itself and what α/β/λ mean. Short version: **λ is the only parameter that changes the policy** — it prices one second of acquisition time in accuracy points. β rescales everything (policy-invariant, leave at 1.0) and α is small exploration shaping (leave at 0.05).

---

## The workflow

### Step 1 — Choose λ

λ prices one second of acquisition time in accuracy points, and it is the only reward parameter that changes which policy is optimal. It is a deployment decision, not a quantity to maximise: pick the acquisition time your dispatch scenario can absorb. `tune_rewards.py` (step 3) shows what the agent achieves at each setting.

### Step 2 — Train at that λ, seeded

```bash
python CODE/training/agent/train_rl_agent.py \
  --warm-start \
  --cost-scale 0.01 \
  --step-weight 0.05 \
  --terminal-weight 1.0 \
  --n-train 1500 --n-eval 1000 \
  --output-dir models/rl_agents/warmstart
```

`--warm-start` seeds the Q-table from measured one-step subset values (see `seeding.py`). **Use it.** Without it the learner has to discover a specific multi-bundle acquisition path from noisy 0/1 terminal rewards, which pure ε-greedy essentially never does — cold-start runs plateau at one bundle (0.597 vs 0.739 at the same λ).

`--cost-scale` is λ. Bundle seconds come from `bundle_costs.py`; λ multiplies them into reward units.

### Step 3 — Sweep λ

```bash
python CODE/training/agent/sweeps/tune_rewards.py --lambdas 0.002 0.01 0.05 0.2
```

Trains an agent at each λ and reports accuracy, bundles acquired, seconds spent, and how many *distinct* acquisition sets the policy used. Higher λ should buy fewer seconds at lower accuracy. `distinct_sets = 1` means the policy collapsed to a fixed shopping list and is ignoring `p_t`.

```bash
python CODE/training/agent/sweeps/tune_learning.py --conf-bins 1 3 5 8
```

Sweeps the learning side: how finely to discretise confidence, and the episode budget. **3 bins is the tuned default** — see the bin-count table in [README.md](README.md#discretising-the-confidence). `--conf-bins 1` is the mask-only ablation, which is the comparison that shows adaptivity is doing anything at all.

For a proper search rather than a grid, use the Optuna tuner — it searches continuous ranges and models interactions between parameters:

```bash
python CODE/training/agent/sweeps/optuna_tune.py --n-trials 40 --seeds 2
tail -f models/rl_agents/optuna/run.log
```

It optimises `accuracy − λ·seconds` (see [README.md](README.md#tuning) for why both terms), reports macro F1 alongside, resumes if interrupted, and warns you when the winning trial's lead is smaller than its own seed noise.

Reward weights have their own grid:

```bash
python CODE/training/agent/sweeps/tune_rewards.py                    # alpha x beta grid
python CODE/training/agent/sweeps/tune_rewards.py --scale-check      # proves beta is redundant
```

`--scale-check` runs k×(α, β, λ) for k ∈ {0.5, 1, 2}. All three share the same ratios, so if the reward is scale-invariant they must give identical policies — and they do, to four decimals. That is the evidence for fixing β = 1 rather than tuning it.

### Step 4 — Inspect and compare

```bash
python CODE/training/agent/analysis/analyze_policy.py --rl-model <model.json> --n-samples 500
python CODE/training/agent/analysis/compare_strategies.py   --rl-model <model.json> --n-samples 300
python CODE/training/agent/analysis/plot_strategies.py \
  --results-csv docs/figures/strategy_comparison/strategy_comparison.csv \
```

`analyze_policy` shows which bundles get acquired and their average confidence contribution. `compare_strategies` compares against three fixed baselines (predict-immediately, greedy-to-threshold, full acquisition).

**Still missing:** the Best Fixed Subset baseline — the best-scoring subset within the agent's time budget, a filter and argmax over the 1,024 already-scored subsets. Until it exists the agent's accuracy has nothing to be judged against.

### Everything at once

```bash
LAMBDA=0.01 bash CODE/training/agent/run_all.sh
```

---

## Reference numbers

Baseline (mask 0, timestamp only): **0.5408** test F1.
Full acquisition (mask 1023): **0.7374**. The unconstrained optimum is mask 511 at **0.7371** — i.e. *dropping `meteo` is free*, it contributes nothing and costs 30 ms.

| setup | accuracy | steps |
|---|---|---|
| cold start, λ=0.01 | 0.597 | 1.0 |
| warm start, λ=0.01 | 0.700 | 7.0 |
| warm start, λ=0.002 | 0.712 | 10.0 |

**0.737 is the ceiling for any mask-only policy.** The state carries no per-collision information beyond the mask, so the policy is necessarily the same subset for every call. Beating it requires putting prediction confidence into the state — see the limitation section in [README.md](README.md).

---

## Troubleshooting

**Every configuration returns identical numbers.**
This was a real bug (fixed 1 Sept 2026): `MDPEnvironment.step()` was not passing α, β or the bundle costs into the reward functions, so they defaulted to α=1, β=1, costs=0 regardless of what you asked for. If you see it again, check that `MDPEnvironment` is being constructed with `step_weight` / `terminal_weight` / `bundle_costs` — `QLearner` must forward its own values through.

**Agent acquires nothing (mean_steps = 0).**
Expected at large λ — time is priced so high nothing is worth acquiring. Check the planner: if it also stops at mask 0 for your λ, the agent is right. If the planner acquires and the agent does not, you forgot `--warm-start`.

**Agent acquires everything at high λ.**
Costs are not reaching the reward. Verify the `Cost scale (lambda)` line printed at startup is non-zero.

**Agent stalls at one bundle.**
Cold start. Add `--warm-start`.

**Score below the planner's number for that λ.**
More episodes, or check `--warm-start` is on. The planner's number is achievable by construction — the gap is a learning problem, never a λ problem.

---

## What NOT to do

- **Do not derive λ or bundle costs from how much accuracy a bundle delivers.** That is circular: cost must reflect operational difficulty (call time, query latency), and the whole point is to discover whether that cost buys enough accuracy. An earlier iteration of this repo did exactly that; it has been removed.
- **Do not tune β.** Scaling β, α and λ together leaves the policy unchanged, so β only rescales the reported returns.
- **Do not select a subset on the test split.** The planner deliberately plans on valid and reports test.
