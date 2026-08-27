# Active Feature Acquisition Agent

Reinforcement Learning agent that decides which features to acquire before making a severity prediction, balancing prediction accuracy against acquisition costs.

## Overview

The system is formulated as a Markov Decision Process (MDP):

- **State**: Current acquisition mask (which of the 10 feature bundles have been acquired) + predicted class probabilities
- **Actions**: Predict & terminate (return prediction) or acquire one of the 10 remaining bundles
- **Transitions**: Acquiring bundle i loads the pre-trained classifier for the new mask
- **Rewards**: 
  - Per-step: α·ΔConfidence - C_i (reward for confidence improvement, penalize acquisition cost)
  - Terminal: β·Accuracy(y_true, ŷ) (reward for correct prediction)

The agent learns a policy that maps each acquisition state to the best next action, trading off confidence improvement against acquisition costs.

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

## Usage

### Train and evaluate an agent

```bash
python train_rl_agent.py \
  --n-train 1000 \
  --n-eval 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.1 \
  --step-weight 1.0 \
  --terminal-weight 1.0
```

**Parameters:**
- `--n-train`: Number of training episodes
- `--n-eval`: Number of evaluation episodes
- `--alpha`: Learning rate (default 0.1)
- `--gamma`: Discount factor (default 0.99)
- `--epsilon`: Exploration rate (default 0.1)
- `--step-weight`: Weight for per-step rewards (default 1.0)
- `--terminal-weight`: Weight for terminal reward (default 1.0)
- `--variant`: Data variant: `normalized` (default), `raw`, `masked`
- `--task`: Task type: `binary` (default) or `multiclass`
- `--output-dir`: Directory to save models (default `models/rl_agents`)
- `--evaluate-only`: Load and evaluate an existing model

**Output:**
- `models/rl_agents/ql_<variant>_<task>_*.json`: Trained agent (Q-table)
- `models/rl_agents/ql_<variant>_<task>_*_results.csv`: Evaluation results

### Test the environment

```bash
python test_mdp.py
```

Runs basic tests to verify the MDP environment and Q-learner work correctly.

### Analyze value-of-information

```bash
python analyze_voi.py \
  --n-samples 1000 \
  --output-dir docs/figures/voi
```

Compares the learned RL agent against simpler baselines (always predict immediately, greedy confidence-maximizing, etc.) and generates plots showing value-of-information curves.

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

## Key Design Decisions

1. **Tabular Q-learning**: The state space (1024 states) is small enough for exact tabular representation. No function approximation needed.

2. **Deterministic transitions**: The MDP is fully observable — there's no randomness in transitions. The agent's only uncertainty is about optimal actions.

3. **No per-sample training data**: The agent trains on random samples from the training set. Each sample is independent; there's no sequential decision-making over time.

4. **Pre-trained model cache**: All classification models are pre-trained offline. The agent only does inference and policy learning, not model training.

5. **Episodic structure**: Each collision's features form one episode. The agent acquires bundles until it terminates with a prediction.

## Limitations and Future Work

1. **Reward function tuning**: The α and β parameters need empirical tuning. Current defaults (1.0, 1.0) may not reflect real-world cost/accuracy trade-offs.

2. **Bundle costs**: All bundles currently cost 0. Real costs should be set based on:
   - Call time (ask_caller bundles)
   - API latency (external_api bundles)
   - GIS query time (map_lookup bundles)

3. **Multi-sample training**: Currently each episode is independent. Future work could train a policy that's optimal for a batch of collisions simultaneously.

4. **Function approximation**: If the action space grows beyond 10 bundles, tabular Q-learning won't scale. A neural network policy could handle that.

5. **Exploration strategies**: Currently uses epsilon-greedy. Could try UCB or other strategies if exploration is a bottleneck.

## References

- MDP formulation: See `docs/data_pipeline.md` section 5 (Active Feature Acquisition)
- Pre-trained models: Generated by `CODE/training/train/bundle_sweep.py`
- Feature definitions: `CODE/training/train/train_classifier.py` (BUNDLE_COLS, BUNDLE_NAMES)
