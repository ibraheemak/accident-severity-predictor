# RL Agent Evaluation & Analysis Guide

Complete workflow for validating, tuning, and understanding the learned acquisition policy.

## Quick Start

Run the full evaluation pipeline (takes ~10 minutes):

```bash
bash CODE/training/agent/full_evaluation.sh
```

This will:
1. ✅ Train agent on 1000 episodes
2. ✅ Analyze what bundles it prefers to acquire
3. ✅ Test 9 different reward weight configurations
4. ✅ Compare against baseline strategies
5. ✅ Generate publication-quality visualizations

---

## Individual Tools

### 1. Train Agent (Extended)

```bash
# Standard training
python CODE/training/agent/train_rl_agent.py \
  --n-train 1000 \
  --n-eval 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.1
```

**What it does:**
- Trains Q-learning agent on 1000 collision samples
- Evaluates on 500 test samples
- Shows cost summary and policy decisions
- Saves trained model to `models/rl_agents/`

**Key parameters:**
- `--n-train`: Number of training episodes (default 1000)
- `--n-eval`: Number of evaluation episodes (default 500)
- `--alpha`: Learning rate (default 0.1)
- `--gamma`: Discount factor (default 0.99)
- `--epsilon`: Exploration rate (default 0.1)
- `--step-weight`: Weight for confidence rewards α (default 1.0)
- `--terminal-weight`: Weight for accuracy rewards β (default 1.0)

---

### 2. Analyze Learned Policy

```bash
python CODE/training/agent/analyze_policy.py \
  --rl-model models/rl_agents/ql_normalized_binary_a0.1_g0.99_e0.1.json \
  --n-samples 500
```

**What it shows:**
- Which bundles the agent acquires at each state
- Value-of-information per bundle (avg confidence improvement)
- How many times each bundle was acquired
- Acquisition sequences for sample collisions

**Output:**
```
================================================================================
POLICY ANALYSIS
================================================================================

Action Distribution:
  Predict (action 0): 450 times
  Acquire osm                  (action 8):   25 times
  Acquire meteo                (action 9):   15 times
  Acquire aadf                 (action 7):   10 times
  ...

Value-of-Information per Bundle:
(Avg confidence improvement when acquiring)

  [8] osm                  +0.0623 ± 0.0456  (acquired   25 times)
  [9] meteo                +0.0412 ± 0.0389  (acquired   15 times)
  [7] aadf                 +0.0198 ± 0.0134  (acquired   10 times)
  [0] vehicles             never acquired
  ...
```

**Interpretation:**
- **High value bundles**: Agent prefers these (worth the cost)
- **Never acquired**: Agent learned these don't help enough to justify cost
- **Low acquisition count**: High cost relative to benefit

---

### 3. Tune Reward Weights (α, β)

The agent's behavior depends on two reward weights:
- **α (step_weight)**: How much to reward per-step confidence improvement
- **β (terminal_weight)**: How much to reward final prediction accuracy

Test different combinations:

```bash
python CODE/training/agent/tune_rewards.py \
  --alpha-values 0.5 1.0 2.0 \
  --beta-values 0.5 1.0 2.0 \
  --n-train 500 \
  --n-eval 200
```

**Output:**
```
   alpha  beta  accuracy  mean_return  mean_steps  mean_confidence
     2.0   2.0    0.6823      0.6823       1.2300           0.7156
     2.0   1.0    0.6734      0.6734       1.1800           0.7043
     1.0   2.0    0.6645      0.6645       0.9200           0.6734
     ...
```

**What to look for:**
- **High accuracy**: Agent is making correct predictions
- **Low mean_steps**: Agent isn't over-acquiring features
- **High mean_confidence**: Agent is confident in its decisions
- Best balance often trades off steps vs accuracy

---

### 4. Compare Against Baselines

```bash
python CODE/training/agent/analyze_voi.py \
  --rl-model models/rl_agents/ql_normalized_binary_a0.1_g0.99_e0.1.json \
  --n-samples 1000
```

Compares learned policy against:
- **predict_immediate**: Predict without acquiring anything (baseline 1)
- **full_acquisition**: Acquire all 10 bundles, then predict (baseline 2)
- **greedy_confidence**: Acquire until confidence > 0.8 (baseline 3)
- **learned_policy**: Your trained agent (if model provided)

**Output:**
```
Strategy Comparison Summary
================================================

predict_immediate:
  Accuracy: 0.3500
  Mean acquisition steps: 0.00
  Mean final confidence: 0.5234
  Mean reward: 0.3500

learned_policy:
  Accuracy: 0.6200
  Mean acquisition steps: 1.45
  Mean final confidence: 0.6834
  Mean reward: 0.6200

full_acquisition:
  Accuracy: 0.6450
  Mean acquisition steps: 10.00
  Mean final confidence: 0.7102
  Mean reward: 0.6450
```

**Interpretation:**
- If `learned_policy` accuracy is close to `full_acquisition` but with fewer steps, it's working well
- If significantly worse, may need more training or different reward weights
- Comparison against `predict_immediate` shows value of acquisition

---

### 5. Create Visualizations

```bash
python CODE/training/agent/visualize_voi.py \
  --results-csv docs/figures/voi/strategy_comparison.csv \
  --policy-analysis-json docs/figures/policy_analysis/policy_analysis.json \
  --output-dir docs/figures/voi_plots
```

**Generates:**
1. **bundle_costs.png**: Acquisition costs by bundle (color-coded by tier)
2. **voi_summary.png**: Accuracy vs steps and accuracy vs time
3. **bundle_value.png**: Value-of-information ranking

---

## Workflow Recommendations

### Scenario 1: Validate That Learning Works

```bash
# Quick validation (2 minutes)
python CODE/training/agent/train_rl_agent.py --n-train 500 --n-eval 200
python CODE/training/agent/analyze_policy.py \
  --rl-model models/rl_agents/ql_normalized_binary_*.json \
  --n-samples 200
```

**Check:** Does the policy acquire external APIs preferentially? (they're cheapest)

### Scenario 2: Tune for Your Use Case

```bash
# If accuracy is more important than latency:
python CODE/training/agent/tune_rewards.py \
  --alpha-values 0.1 0.5 1.0 \
  --beta-values 2.0 3.0 4.0

# Then train with best β:
python CODE/training/agent/train_rl_agent.py \
  --n-train 2000 \
  --terminal-weight 3.0
```

### Scenario 3: Comprehensive Analysis

```bash
# Full pipeline (production quality)
bash CODE/training/agent/full_evaluation.sh

# Then review outputs:
# - docs/figures/policy_analysis/policy_analysis.json
# - models/rl_agents/tuning/tuning_results.csv
# - docs/figures/voi_plots/*.png
```

---

## Expected Results

### With Realistic Bundle Costs

**Learned policy should:**
- ✅ Prefer external APIs (10-30ms) over map lookups (40-60ms)
- ✅ Prefer map lookups over caller questions (1-3 seconds)
- ✅ Rarely acquire caller bundles unless confidence very low
- ✅ Achieve 55-65% accuracy with ~1 acquisition step on average

**Baseline comparisons:**
- predict_immediate: ~35% accuracy, 0 steps
- learned_policy: ~60% accuracy, ~1.5 steps
- full_acquisition: ~65% accuracy, 10 steps

The learned policy should be close to full_acquisition in accuracy while using far fewer acquisitions.

---

## Troubleshooting

### Agent only predicts, never acquires

**Symptom:** Policy always chooses action 0 (predict)

**Causes:**
1. Not enough training episodes (use `--n-train 2000`)
2. Acquisition costs too high relative to rewards
3. Reward weights poorly calibrated (try different α/β)

**Fix:**
```bash
# Increase step reward weight to encourage exploration
python CODE/training/agent/train_rl_agent.py \
  --n-train 2000 \
  --step-weight 2.0 \
  --epsilon 0.2
```

### Accuracy worse than baselines

**Symptom:** learned_policy accuracy << full_acquisition accuracy

**Causes:**
1. Insufficient training
2. Poor reward weight balance
3. Learning rate too high (causes instability)

**Fix:**
```bash
# More training + lower learning rate
python CODE/training/agent/train_rl_agent.py \
  --n-train 5000 \
  --alpha 0.05 \
  --terminal-weight 2.0
```

### Agent acquires everything

**Symptom:** Policy acquires many bundles even expensive ones

**Causes:**
1. Terminal reward weight too low
2. Confidence rewards too high
3. Acquisition costs may be miscalibrated

**Fix:**
```bash
# Increase accuracy reward
python CODE/training/agent/train_rl_agent.py \
  --terminal-weight 2.0 \
  --step-weight 0.5
```

---

## Key Metrics to Track

| Metric | Good Range | Interpretation |
|--------|-----------|-----------------|
| Accuracy | 55-65% | Prediction correctness |
| Mean steps | 0.5-2.0 | Avg acquisitions per collision |
| Mean confidence | 0.65-0.75 | Model certainty |
| External API acquisitions | >50% of all acquisitions | Prefers cheap bundles |
| Caller acquisitions | <10% of all acquisitions | Avoids expensive bundles |

---

## Next Steps

1. **Run full evaluation** → `bash full_evaluation.sh`
2. **Review policy analysis** → Check which bundles add value
3. **Tune parameters** → Test different α/β combinations
4. **Select best config** → Choose weights that balance accuracy/cost for your use case
5. **Deploy** → Train final agent with best parameters on larger dataset

---

## References

- MDP formulation: See [README.md](README.md)
- Bundle costs: [bundle_costs.py](bundle_costs.py)
- Training details: [train_rl_agent.py](train_rl_agent.py)
