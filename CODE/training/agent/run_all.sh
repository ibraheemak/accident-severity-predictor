#!/bin/bash
# Full evaluation pipeline: plan exactly, train against the plan, analyse, visualise.
#
# The 1,024 subset classifiers are pre-trained by bundle_sweep.py; everything
# here reads them. The agent's Q-table is seeded from their measured one-step
# values (seeding.py) because a cold start plateaus after a single bundle.

set -e

echo "================================================"
echo "RL Agent Full Evaluation Pipeline"
echo "================================================"

LAMBDA=${LAMBDA:-0.01}          # score points per second of acquisition time
TRAIN_EPISODES=${TRAIN_EPISODES:-1500}
EVAL_EPISODES=${EVAL_EPISODES:-1000}
OUTPUT_DIR="models/rl_agents/full_eval"

mkdir -p "$OUTPUT_DIR"

# Step 1: what each bundle is worth (analysis only, no training).
echo ""
echo "STEP 1: Value-of-information sweep over the 1,024 subsets..."
python CODE/training/agent/voi/compute_voi.py
python CODE/training/agent/voi/plot_voi.py

# Step 2: train the agent at the chosen lambda, seeded from measured values.
echo ""
echo "STEP 2: Training agent at lambda=$LAMBDA ($TRAIN_EPISODES episodes)..."
python CODE/training/agent/train_rl_agent.py \
  --n-train "$TRAIN_EPISODES" \
  --n-eval "$EVAL_EPISODES" \
  --step-weight 0.05 \
  --terminal-weight 1.0 \
  --cost-scale "$LAMBDA" \
  --warm-start \
  --output-dir "$OUTPUT_DIR"

MODEL_PATH=$(ls -t "$OUTPUT_DIR"/ql_*.json | head -1)
echo "✓ Model saved: $MODEL_PATH"

# Step 3: does the trained agent actually reach the planner's optimum?
echo ""
echo "STEP 3: Sweeping lambda -- what the agent does at each price of time..."
python CODE/training/agent/sweeps/tune_rewards.py \
  --lambdas 0.002 0.01 0.05 0.2 \
  --n-train 1000 \
  --n-eval 500 \
  --output-dir "$OUTPUT_DIR/lambda_sweep"

# Step 4: what did the policy actually learn to acquire?
echo ""
echo "STEP 4: Analysing the learned policy..."
python CODE/training/agent/analysis/analyze_policy.py \
  --rl-model "$MODEL_PATH" \
  --n-samples 500 \
  --output-dir "docs/figures/policy_analysis"

# Step 5: compare against fixed baselines.
echo ""
echo "STEP 5: Baseline comparison (predict-now / greedy / full acquisition)..."
python CODE/training/agent/analysis/compare_strategies.py \
  --n-samples 300 \
  --rl-model "$MODEL_PATH" \
  --output-dir "docs/figures/strategy_comparison"

# Step 6: plots.
echo ""
echo "STEP 6: Creating visualisations..."
python CODE/training/agent/analysis/plot_strategies.py \
  --results-csv "docs/figures/strategy_comparison/strategy_comparison.csv" \
  --output-dir "docs/figures/strategy_comparison"

echo ""
echo "================================================"
echo "EVALUATION COMPLETE"
echo "================================================"
echo ""
echo "Key outputs:"
echo "  1. Bundle values:    docs/figures/voi_sweep/"
echo "  2. Trained model:    $MODEL_PATH"
echo "  3. Lambda sweep:     $OUTPUT_DIR/lambda_sweep/lambda_sweep.csv"
echo "  4. Policy analysis:  docs/figures/policy_analysis/"
echo "  5. Baselines:        docs/figures/strategy_comparison/"
echo "  6. Plots:            docs/figures/strategy_comparison/"
echo ""
echo "Pick LAMBDA from the step-3 sweep: the row whose acquisition time your"
echo "dispatch scenario can absorb, then re-run step 2 with that value."
echo ""
