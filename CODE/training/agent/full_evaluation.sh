#!/bin/bash
# Full evaluation pipeline: plan exactly, train against the plan, analyse, visualise.
#
# Order matters. The mask-level MDP is fully known (deterministic transitions,
# per-mask accuracies already measured by bundle_sweep.py), so we SOLVE it first
# and use that solution both to pick lambda and to warm-start the learner.

set -e

echo "================================================"
echo "RL Agent Full Evaluation Pipeline"
echo "================================================"

LAMBDA=${LAMBDA:-0.01}          # score points per second of acquisition time
TRAIN_EPISODES=${TRAIN_EPISODES:-1500}
EVAL_EPISODES=${EVAL_EPISODES:-1000}
OUTPUT_DIR="models/rl_agents/full_eval"

mkdir -p "$OUTPUT_DIR"

# Step 1: exact solution of the mask-level MDP, for every lambda.
echo ""
echo "STEP 1: Solving the mask-level MDP exactly (Pareto frontier over lambda)..."
python CODE/training/agent/plan_optimal_policy.py

# Step 2: train the agent at the chosen lambda, warm-started from that solution.
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
echo "STEP 3: Sweeping lambda -- agent vs exact planner..."
python CODE/training/agent/tune_rewards.py \
  --lambdas 0.002 0.01 0.05 0.2 \
  --n-train 1000 \
  --n-eval 500 \
  --output-dir "$OUTPUT_DIR/lambda_sweep"

# Step 4: what did the policy actually learn to acquire?
echo ""
echo "STEP 4: Analysing the learned policy..."
python CODE/training/agent/analyze_policy.py \
  --rl-model "$MODEL_PATH" \
  --n-samples 500 \
  --output-dir "docs/figures/policy_analysis"

# Step 5: compare against fixed baselines.
echo ""
echo "STEP 5: Baseline comparison (predict-now / greedy / full acquisition)..."
python CODE/training/agent/analyze_voi.py \
  --n-samples 300 \
  --rl-model "$MODEL_PATH" \
  --output-dir "docs/figures/voi"

# Step 6: plots.
echo ""
echo "STEP 6: Creating visualisations..."
python CODE/training/agent/visualize_voi.py \
  --results-csv "docs/figures/voi/strategy_comparison.csv" \
  --policy-analysis-json "docs/figures/policy_analysis/policy_analysis.json" \
  --output-dir "docs/figures/voi_plots"

echo ""
echo "================================================"
echo "EVALUATION COMPLETE"
echo "================================================"
echo ""
echo "Key outputs:"
echo "  1. Exact frontier:   docs/figures/policy_analysis/exact_policy_frontier.csv"
echo "  2. Trained model:    $MODEL_PATH"
echo "  3. Agent vs planner: $OUTPUT_DIR/lambda_sweep/lambda_sweep.csv"
echo "  4. Policy analysis:  docs/figures/policy_analysis/"
echo "  5. Baselines:        docs/figures/voi/"
echo "  6. Plots:            docs/figures/voi_plots/"
echo ""
echo "Read the frontier CSV first: pick the row whose time budget you accept,"
echo "then re-run step 2 with LAMBDA set to that row's lambda."
echo ""
