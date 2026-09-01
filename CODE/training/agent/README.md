# Active Feature Acquisition Agent

Reinforcement Learning agent that decides which features to acquire before making a severity prediction, balancing prediction accuracy against acquisition costs.

## Overview

The system is formulated as a Markov Decision Process (MDP) over the **normalized, binary (Slight vs. Severe)** task:

- **State**: Current acquisition mask (which of the 10 feature bundles have been acquired) + predicted class probabilities
- **Actions**: Predict & terminate (return prediction) or acquire one of the 10 remaining bundles
- **Transitions**: Acquiring bundle i loads the pre-trained classifier for the new mask
- **Rewards**: 
  - Per-step: α·ΔConfidence - C_i (reward for confidence improvement, penalize acquisition cost)
  - Terminal: β·Accuracy(y_true, ŷ) (reward for correct prediction)

The agent learns a policy that maps each acquisition state to the best next action, trading off confidence improvement against acquisition costs.

**Note**: This implementation is hardcoded for the normalized, binary task. Raw and masked variants, as well as multiclass, are not supported.

## Components

### `mdp_environment.py`

Implements the MDP environment. Manages:
- State transitions (acquiring bundles)
- Model loading from the pre-trained classifier cache
- Prediction and reward computation
- Episode bookkeeping

**Key classes:**
- `MDPEnvironment`: One-collision environment instance

**Example:**
```python
from mdp_environment import MDPEnvironment

env = MDPEnvironment(X_row, y_true)
state = env.get_state()
state, reward, done = env.step(1)  # Acquire bundle 0
```

### `ql_solver.py`

Tabular Q-learning solver. Learns an optimal policy over the 1024-state space (2^10 acquisition subsets).

**Key classes:**
- `QLearner`: Q-learning agent with epsilon-greedy exploration

**Example:**
```python
from ql_solver import QLearner

learner = QLearner(alpha=0.1, gamma=0.99, epsilon=0.1)
learner.train(X_train, y_train, n_episodes=1000)
results = learner.evaluate(X_test, y_test)
policy = learner.get_policy()  # Extract learned policy
```

## Quick Start

### Full Evaluation Pipeline (10 minutes)

```bash
# Train, analyze, tune, and visualize everything
bash full_evaluation.sh
```

This runs:
1. ✅ Train agent on 1000 episodes
2. ✅ Analyze what bundles it prefers
3. ✅ Test 9 reward weight configurations
4. ✅ Compare against baselines
5. ✅ Generate publication-quality plots

Outputs go to `docs/figures/` and `models/rl_agents/tuning/`

### Quick Validation (2 minutes)

```bash
# Train and check if agent learns
python train_rl_agent.py --n-train 500 --n-eval 200
python analyze_policy.py --rl-model models/rl_agents/*.json --n-samples 200
```

---

## Usage

### Train and evaluate an agent

```bash
python train_rl_agent.py \
  --n-train 1000 \
  --n-eval 500 \
  --step-weight 1.0 \
  --terminal-weight 1.0
```

**Parameters:**
- `--n-train`: Training episodes (default 1000)
- `--n-eval`: Evaluation episodes (default 500)
- `--alpha`: Learning rate (default 0.1)
- `--gamma`: Discount factor (default 0.99)
- `--epsilon`: Exploration rate (default 0.1)
- `--step-weight`: Weight for per-step confidence rewards α (default 1.0)
- `--terminal-weight`: Weight for terminal accuracy reward β (default 1.0)
- `--output-dir`: Save directory (default `models/rl_agents`)

**Output:**
- `models/rl_agents/ql_normalized_binary_*.json`: Trained agent (Q-table)
- `models/rl_agents/ql_normalized_binary_*_results.csv`: Evaluation results

### Analyze what the agent learned

```bash
python analyze_policy.py \
  --rl-model models/rl_agents/ql_normalized_binary_a0.1_g0.99_e0.1.json \
  --n-samples 500
```

Shows:
- Which bundles the agent acquires
- Value-of-information per bundle
- How often each bundle is used

Example output:
```
Value-of-Information per Bundle:
  [8] osm                  +0.0623 ± 0.0456  (acquired   250 times)
  [9] meteo                +0.0412 ± 0.0389  (acquired   180 times)
  [7] aadf                 +0.0198 ± 0.0134  (acquired    85 times)
  [0] vehicles             -0.0012 ± 0.0089  (acquired    12 times)
```

### Tune reward weights (α, β)

```bash
python tune_rewards.py \
  --alpha-values 0.5 1.0 2.0 \
  --beta-values 0.5 1.0 2.0 \
  --n-train 500
```

Tests different reward configurations and recommends the best one based on accuracy.

### Compare against baselines

```bash
python analyze_voi.py \
  --n-samples 1000 \
  --rl-model models/rl_agents/ql_normalized_binary_*.json
```

Compares learned policy against:
- **Predict immediately**: No acquisitions (baseline accuracy)
- **Full acquisition**: Acquire all bundles (upper bound)
- **Greedy confidence**: Acquire until confidence > 0.8
- **Learned policy**: Your trained agent

### Visualize results

```bash
python visualize_voi.py \
  --results-csv docs/figures/voi/strategy_comparison.csv \
  --policy-analysis-json docs/figures/policy_analysis/policy_analysis.json
```

Generates:
- Bundle costs by tier
- Accuracy vs acquisition cost curves
- Value-of-information rankings

### Test the environment

```bash
python test_mdp.py
```

Verifies MDP environment and Q-learner work correctly.

## Architecture Details

### The 10 Acquisition Bundles

| Bundle | Kind | Features | Cost Tier |
|--------|------|----------|-----------|
| vehicles | Ask caller | number_of_vehicles, vehicle type flags | ask_caller |
| casualties | Ask caller | n_casualties, pedestrian/driver/etc counts | ask_caller |
| casualty_demographics | Ask caller | Age ranges, female/child/elderly counts | ask_caller |
| road_class | Map lookup | Road class, numbers, type, trunk flag | map_lookup |
| road_environment | Map lookup | Speed limit, urban/rural, light, weather | map_lookup |
| junction | Map lookup | Junction detail, control | map_lookup |
| crash_history | Map lookup | Historical accident rate/severity/distance | map_lookup |
| aadf | External API | Annual Average Daily Flow, distance | external_api |
| osm | External API | Traffic signals, calming, lane count | external_api |
| meteo | External API | Temperature, precipitation, wind, weather code | external_api |

Bundle order (0-9) is the bit order in the acquisition mask — do not reorder without invalidating all cached models.

### Model Cache

The RL environment uses pre-trained classifiers cached in:
```
models/subsets/normalized_binary/000.json  (free info only, timestamp)
                                  001.json  (bundle 0 acquired)
                                  ...
                                  1023.json (all 10 bundles acquired)
```

All 1024 classifiers are pre-trained once by `bundle_sweep.py` and reused by the RL agent. No model training happens at runtime — only inference.

### Q-Learning Details

**State space:** 1024 states (one per acquisition bitmask)
**Action space:** 11 actions (predict, or acquire one of 10 bundles)
**Update rule:** Standard tabular Q-learning with epsilon-greedy exploration
**Convergence:** Guaranteed since state/action spaces are finite and transitions are deterministic

The learner updates the Q-table during training:
```
Q[s][a] ← Q[s][a] + α · (r + γ·max Q[s'][a'] - Q[s][a])
```

At evaluation time, it uses the greedy policy:
```
π(s) = argmax_a Q[s][a]
```

## What's New (vs Initial Implementation)

✅ **Realistic Bundle Costs** — Set based on real-world latencies:
- External APIs: 10-30ms (cheapest)
- Map lookups: 40-60ms (medium)
- Caller questions: 1-3 seconds (most expensive)

✅ **Reward Tuning** — Tools to find optimal α, β parameters:
- `tune_rewards.py`: Grid search over configurations
- Outputs ranked results with recommendations

✅ **Policy Analysis** — Understand what the agent learned:
- `analyze_policy.py`: Shows bundle value-of-information
- `analyze_voi.py`: Compares against baselines

✅ **Visualizations** — Publication-quality plots:
- `visualize_voi.py`: Cost/accuracy trade-offs
- Bundle value rankings
- Strategy comparisons

✅ **Evaluation Guide** — Complete workflow documentation:
- `EVALUATION.md`: Usage patterns, troubleshooting, expected results

---

## Key Design Decisions

1. **Tabular Q-learning**: The state space (1024 states) is small enough for exact tabular representation. No function approximation needed.

2. **Deterministic transitions**: The MDP is fully observable — there's no randomness in transitions. The agent's only uncertainty is about optimal actions.

3. **No per-sample training data**: The agent trains on random samples from the training set. Each sample is independent; there's no sequential decision-making over time.

4. **Pre-trained model cache**: All classification models are pre-trained offline. The agent only does inference and policy learning, not model training.

5. **Episodic structure**: Each collision's features form one episode. The agent acquires bundles until it terminates with a prediction.

6. **Realistic costs**: Bundle costs reflect actual dispatch center latencies. Agent learns to prefer cheap bundles when benefit is similar.

## Bundle Acquisition Costs

Realistic costs are set in [bundle_costs.py](bundle_costs.py):

**Ask Caller (Most Expensive):**
- vehicles: 1.0 sec
- casualties: 2.0 sec
- casualty_demographics: 3.0 sec (hardest question)

**Map Lookup (Medium):**
- junction: 40 ms
- road_class: 50 ms
- road_environment: 50 ms
- crash_history: 60 ms (radius query)

**External API (Cheapest):**
- osm: 10 ms
- aadf: 20 ms
- meteo: 30 ms

The agent learns to prefer cheap bundles (external APIs) over expensive ones (caller questions) when the benefit is similar.

---

## Evaluation & Tuning

See [EVALUATION.md](EVALUATION.md) for:
- Complete evaluation workflow
- How to interpret results
- Troubleshooting guide
- Expected performance ranges
- Parameter tuning strategies

Quick summary:
- `tune_rewards.py`: Grid search over α, β
- `analyze_policy.py`: Understand what agent learned
- `analyze_voi.py`: Compare against baselines
- `visualize_voi.py`: Generate plots
- `full_evaluation.sh`: Run everything

---

## Limitations and Future Work

1. **Reward function tuning**: The α and β parameters can be tuned with `tune_rewards.py`. See [EVALUATION.md](EVALUATION.md) for workflow.

2. **Multi-sample training**: Currently each episode is independent. Future work could train a policy optimal for a batch of collisions simultaneously.

3. **Function approximation**: If action space grows beyond 10 bundles, tabular Q-learning won't scale. A neural network policy could handle that.

4. **Exploration strategies**: Currently uses epsilon-greedy. Could try UCB or other strategies if exploration is a bottleneck.

5. **Cost recalibration**: Bundle costs in [bundle_costs.py](bundle_costs.py) should be validated against actual dispatch center measurements.

## File Reference

**Core Implementation:**
- [mdp_environment.py](mdp_environment.py) — MDP state/action/reward logic
- [ql_solver.py](ql_solver.py) — Q-learning algorithm
- [bundle_costs.py](bundle_costs.py) — Acquisition costs by bundle

**Training & Evaluation:**
- [train_rl_agent.py](train_rl_agent.py) — Main training script
- [analyze_policy.py](analyze_policy.py) — Understand learned behavior
- [analyze_voi.py](analyze_voi.py) — Compare vs baselines
- [tune_rewards.py](tune_rewards.py) — Grid search over α, β
- [visualize_voi.py](visualize_voi.py) — Generate plots

**Orchestration & Documentation:**
- [full_evaluation.sh](full_evaluation.sh) — Run entire pipeline
- [EVALUATION.md](EVALUATION.md) — Complete usage guide
- [README.md](README.md) — This file

**Testing:**
- [test_mdp.py](test_mdp.py) — Unit tests

---

## External References

- MDP formulation: `docs/data_pipeline.md` section 5 (Active Feature Acquisition)
- Pre-trained models: Generated by `CODE/training/train/bundle_sweep.py` (1024 cached classifiers)
- Feature definitions: `CODE/training/train/train_classifier.py` (BUNDLE_COLS, BUNDLE_NAMES)
