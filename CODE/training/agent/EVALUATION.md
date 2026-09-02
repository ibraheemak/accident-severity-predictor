# RL Agent Evaluation Guide

How to pick λ, train against it, and check the result is actually optimal.

See [README.md](README.md) for the MDP itself and what α/β/λ mean. Short version: **λ is the only parameter that changes the policy** — it prices one second of acquisition time in accuracy points. β rescales everything (policy-invariant, leave at 1.0) and α is small exploration shaping (leave at 0.05).

---

## The workflow

### Step 1 — Solve the MDP exactly, then choose λ

```bash
python CODE/training/agent/plan_optimal_policy.py
```

This does **not** train anything. It solves the mask-level MDP in closed form by dynamic programming (transitions are deterministic and every mask's accuracy is already known from `bundle_sweep.py`) and sweeps λ to produce the accuracy/time menu:

```
 lambda  mask  n_bundles  seconds  valid  test_f1  acquired (in order)
  0.002   511          9     6.23 0.7401   0.7371  vehicles -> casualties -> ... -> osm
  0.010   115          5     3.15 0.7227   0.7088  vehicles -> casualties -> road_environment -> junction -> crash_history
  0.100   208          3     0.13 0.6377   0.6381  road_environment -> crash_history -> aadf
  1.000   256          1     0.01 0.5943   0.5828  osm
```

**Pick the row whose acquisition time your dispatch scenario can absorb.** Its λ is your setting and its `test_f1` is the score the trained agent should reach. There is no "best" row — the choice is about how much call-handler time is worth, which is your call, not the model's.

Options: `--metric valid_accuracy` to plan on accuracy instead of macro F1; `--lambdas ...` for a custom sweep.

### Step 2 — Train at that λ, warm-started

```bash
python CODE/training/agent/train_rl_agent.py \
  --warm-start \
  --cost-scale 0.01 \
  --step-weight 0.05 \
  --terminal-weight 1.0 \
  --n-train 1500 --n-eval 1000 \
  --output-dir models/rl_agents/warmstart
```

`--warm-start` seeds the Q-table from the step-1 DP solution. **Use it.** Without it the learner has to discover a specific multi-bundle acquisition path from noisy 0/1 terminal rewards, which pure ε-greedy essentially never does — cold-start runs plateau at one bundle (0.597 vs 0.700 at the same λ).

`--cost-scale` is λ. Bundle seconds come from `bundle_costs.py`; λ multiplies them into reward units.

### Step 3 — Check the agent matched the planner

```bash
python CODE/training/agent/tune_rewards.py --lambdas 0.002 0.01 0.05 0.2
```

Trains an agent at each λ and prints it beside the exact optimum:

```
 lambda  planner_bundles  planner_test_acc  agent_bundles  agent_test_acc    gap
  0.002                9            0.7371          10.00          0.7120  -0.03
  0.010                5            0.7089           7.00          0.7000  -0.01
```

A `gap` near zero means the learner reached the optimum. A large negative gap means **under-training or a bad warm start — not a bad λ.** λ cannot be "wrong"; it defines which optimum you are aiming at.

### Step 4 — Inspect and compare

```bash
python CODE/training/agent/analyze_policy.py --rl-model <model.json> --n-samples 500
python CODE/training/agent/analyze_voi.py   --rl-model <model.json> --n-samples 300
python CODE/training/agent/visualize_voi.py \
  --results-csv docs/figures/voi/strategy_comparison.csv \
  --policy-analysis-json docs/figures/policy_analysis/policy_analysis.json
```

`analyze_policy` shows which bundles get acquired and their average confidence contribution. `analyze_voi` compares against three fixed baselines (predict-immediately, greedy-to-threshold, full acquisition).

### Everything at once

```bash
LAMBDA=0.01 bash CODE/training/agent/full_evaluation.sh
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
