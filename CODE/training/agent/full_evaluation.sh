#!/bin/bash
# Full evaluation pipeline: train, analyze, tune, visualize

set -e  # Exit on error

echo "================================================"
echo "RL Agent Full Evaluation Pipeline"
echo "================================================"

# Configuration
TRAIN_EPISODES=1000
EVAL_EPISODES=500
OUTPUT_DIR="models/rl_agents/full_eval"

mkdir -p "$OUTPUT_DIR"

# Step 1: Train longer
echo ""
echo "STEP 1: Training agent on 1000 episodes..."
echo "This may take 1-2 minutes..."
python CODE/training/agent/train_rl_agent.py \
  --n-train $TRAIN_EPISODES \
  --n-eval $EVAL_EPISODES \
  --output-dir "$OUTPUT_DIR"

MODEL_PATH=$(ls "$OUTPUT_DIR"/ql_*.json | head -1)
RESULTS_CSV=$(ls "$OUTPUT_DIR"/ql_*_results.csv | head -1)

echo "✓ Model saved: $MODEL_PATH"
echo "✓ Results saved: $RESULTS_CSV"

# Step 2: Analyze policy
echo ""
echo "STEP 2: Analyzing learned policy..."
python CODE/training/agent/analyze_policy.py \
  --rl-model "$MODEL_PATH" \
  --n-samples 500 \
  --output-dir "docs/figures/policy_analysis"

# Step 3: Tune reward weights
echo ""
echo "STEP 3: Tuning reward weights (α, β)..."
echo "This will test 9 configurations × 500 train × 200 eval = 4500 episodes"
echo "This will take ~5 minutes..."
python CODE/training/agent/tune_rewards.py \
  --alpha-values 0.5 1.0 2.0 \
  --beta-values 0.5 1.0 2.0 \
  --n-train 300 \
  --n-eval 200 \
  --output-dir "$OUTPUT_DIR/tuning"

# Step 4: Value-of-information analysis
echo ""
echo "STEP 4: Running value-of-information comparison..."
python CODE/training/agent/analyze_voi.py \
  --n-samples 300 \
  --rl-model "$MODEL_PATH" \
  --output-dir "docs/figures/voi"

# Step 5: Visualize
echo ""
echo "STEP 5: Creating visualizations..."
python CODE/training/agent/visualize_voi.py \
  --results-csv "docs/figures/voi/strategy_comparison.csv" \
  --policy-analysis-json "docs/figures/policy_analysis/policy_analysis.json" \
  --output-dir "docs/figures/voi_plots"

# Summary
echo ""
echo "================================================"
echo "EVALUATION COMPLETE"
echo "================================================"
echo ""
echo "Key outputs:"
echo "  1. Trained model: $MODEL_PATH"
echo "  2. Policy analysis: docs/figures/policy_analysis/"
echo "  3. Parameter tuning: $OUTPUT_DIR/tuning/tuning_results.csv"
echo "  4. VOI comparison: docs/figures/voi/"
echo "  5. Visualizations: docs/figures/voi_plots/"
echo ""
echo "Next steps:"
echo "  - Review policy_analysis results to see what bundles agent prefers"
echo "  - Check tuning_results.csv for best α/β parameters"
echo "  - View plots in docs/figures/voi_plots/"
echo "  - Retrain with best parameters if desired"
echo ""
