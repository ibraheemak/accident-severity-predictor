#!/usr/bin/env python3
"""Train and evaluate the RL Active Feature Acquisition agent.

This script trains a Q-learning agent on the normalized, binary (Slight vs. Severe)
task. The agent learns when to acquire additional features (bundles) versus making
a prediction, balancing accuracy against acquisition costs.

Usage:
    python train_rl_agent.py                    # Train on 1000 samples
    python train_rl_agent.py --n-train 5000     # More training episodes
    python train_rl_agent.py --alpha 0.05       # Custom learning rate
    python train_rl_agent.py --evaluate-only    # Load and evaluate existing model
"""

import argparse
import json

import numpy as np
import pandas as pd

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[0]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES, CATEGORICAL_COLS,
    unify_categorical_dtypes,
)

from mdp_environment import MDPEnvironment
from ql_solver import QLearner
from seeding import one_step_seed
from bundle_costs import get_costs_dict, print_costs_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Number of evaluation episodes")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Exploration rate")
    parser.add_argument("--step-weight", type=float, default=1.0,
                        help="α parameter for step rewards")
    parser.add_argument("--terminal-weight", type=float, default=1.0,
                        help="β parameter for terminal reward")
    parser.add_argument("--conf-bins", type=int, default=3,
                        help="confidence bins in the state (p_t discretised); "
                             "3 was best in the sweep, 1 = mask-only ablation")
    parser.add_argument("--cost-scale", type=float, default=1.0,
                        help="lambda: multiplier converting bundle seconds into "
                             "reward units; sweep it with tune_rewards.py)")
    parser.add_argument("--warm-start", action="store_true",
                        help="seed the Q-table from measured one-step subset "
                             "values (see seeding.py); strongly recommended -- "
                             "a cold start plateaus after one bundle")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl_agents"),
                        help="Directory to save models")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Load existing model and evaluate only")
    args = parser.parse_args()

    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = (f"ql_normalized_binary_"
                  f"a{args.alpha}_g{args.gamma}_e{args.epsilon}"
                  f"_c{args.cost_scale}_b{args.conf_bins}")
    model_path = output_dir / f"{model_name}.json"

    # Load data (normalized, binary)
    print("Loading normalized binary data...")
    cols_all = bundle_feature_columns(BUNDLE_NAMES, "normalized")

    train_df = load_split("train", cols_all, variant="normalized")
    X_train, y_train = prepare_xy(train_df, cols_all, binary=True)

    test_df = load_split("test", cols_all, variant="normalized")
    X_test, y_test = prepare_xy(test_df, cols_all, binary=True)

    # Ensure categorical columns are properly typed
    unify_categorical_dtypes([X_train, X_test], cols_all)

    print(f"  Train: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"  Test: {len(X_test)} samples")

    # Load realistic bundle costs (seconds), scaled into reward units by lambda
    bundle_costs = {i: c * args.cost_scale for i, c in get_costs_dict().items()}
    print_costs_summary()
    print(f"Cost scale (lambda): {args.cost_scale} reward units per second")

    # Initialize learner
    learner = QLearner(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        step_weight=args.step_weight,
        terminal_weight=args.terminal_weight,
        bundle_costs=bundle_costs,
        n_conf_bins=args.conf_bins,
        seed=args.seed,
    )

    if args.warm_start:
        sweep_csv = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")
        sweep = pd.read_csv(sweep_csv).set_index("mask")
        learner.warm_start(one_step_seed(
            sweep["valid_accuracy"].to_dict(), bundle_costs,
            n_conf_bins=args.conf_bins, terminal_weight=args.terminal_weight))
        print(f"Seeded Q-table from measured one-step values over "
              f"{len(sweep)} masks ({sweep_csv})")

    if args.evaluate_only:
        print(f"\nLoading model from {model_path}...")
        learner.load(model_path)
    else:
        # Train
        print(f"\nTraining for {args.n_train} episodes...")
        train_stats = learner.train(X_train, y_train, n_episodes=args.n_train)
        print(f"  Mean return (last 100): {train_stats['mean_return']:.4f}")
        print(f"  Mean episode length: {train_stats['mean_length']:.2f}")
        print(f"  Q-table states learned: {train_stats['q_table_size']}")

        # Save
        print(f"\nSaving model to {model_path}...")
        learner.save(model_path)

    # Evaluate
    print(f"\nEvaluating on {args.n_eval} test samples...")
    eval_stats = learner.evaluate(X_test, y_test, n_episodes=args.n_eval)
    print(f"  Accuracy: {eval_stats['accuracy']:.4f}")
    print(f"  Mean return: {eval_stats['mean_return']:.4f}")
    print(f"  Mean steps: {eval_stats['mean_steps']:.2f}")
    print(f"  Mean confidence: {eval_stats['mean_confidence']:.4f}")

    # Save results
    results_path = output_dir / f"{model_name}_results.csv"
    eval_stats["results"].to_csv(results_path, index=False)
    print(f"  Results saved to {results_path}")

    # Does the policy actually differ across confidence bins? If every bin of a
    # mask picks the same action, the agent is not using p_t and has collapsed
    # to a fixed shopping list.
    policy = learner.get_policy()
    nbins = max(args.conf_bins, 1)
    varying = sum(1 for m in range(1 << 10)
                  if len({policy[(m, b)] for b in range(nbins)}) > 1)
    print(f"\nMasks whose action depends on confidence: {varying}/1024")
    print("Policy at mask 0 (start of every episode), by confidence bin:")
    for b in range(nbins):
        a = policy[(0, b)]
        lo = 0.5 + 0.5 * b / nbins
        print(f"  confidence {lo:.2f}-{lo + 0.5/nbins:.2f}: "
              f"{'Predict now' if a == 0 else f'Acquire {BUNDLE_NAMES[a-1]}'}")


if __name__ == "__main__":
    main()
