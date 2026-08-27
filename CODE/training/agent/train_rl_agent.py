#!/usr/bin/env python3
"""Train and evaluate the RL Active Feature Acquisition agent.

This script trains a Q-learning agent that learns when to acquire additional
features (bundles) versus making a prediction. The agent learns to balance
prediction accuracy against acquisition costs.

Usage:
    python train_rl_agent.py                    # Train on 1000 samples
    python train_rl_agent.py --n-train 5000     # More training episodes
    python train_rl_agent.py --alpha 0.05       # Custom learning rate
    python train_rl_agent.py --evaluate-only    # Load and evaluate existing model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES, CATEGORICAL_COLS,
    unify_categorical_dtypes,
)

from mdp_environment import MDPEnvironment
from ql_solver import QLearner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="normalized",
                        choices=["normalized", "raw", "masked"],
                        help="Data variant to use")
    parser.add_argument("--task", default="binary",
                        choices=["binary", "multiclass"],
                        help="Classification task")
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
    model_name = (f"ql_{args.variant}_{args.task}_"
                  f"a{args.alpha}_g{args.gamma}_e{args.epsilon}")
    model_path = output_dir / f"{model_name}.json"

    # Load data
    print(f"Loading {args.variant} {args.task} data...")
    cols_all = bundle_feature_columns(BUNDLE_NAMES, args.variant)

    train_df = load_split("train", cols_all, variant=args.variant)
    X_train, y_train = prepare_xy(train_df, cols_all, binary=(args.task=="binary"))

    test_df = load_split("test", cols_all, variant=args.variant)
    X_test, y_test = prepare_xy(test_df, cols_all, binary=(args.task=="binary"))

    # Ensure categorical columns are properly typed
    unify_categorical_dtypes([X_train, X_test], cols_all)

    print(f"  Train: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"  Test: {len(X_test)} samples")

    # Initialize learner
    learner = QLearner(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        step_weight=args.step_weight,
        terminal_weight=args.terminal_weight,
        variant=args.variant,
        task=args.task,
        seed=args.seed,
    )

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

    # Print policy summary
    print("\nPolicy summary (first 10 masks):")
    policy = learner.get_policy()
    for mask in range(min(10, 1 << 10)):
        action = policy[mask]
        if action == 0:
            action_str = "Predict"
        else:
            action_str = f"Acquire bundle {action-1}"
        print(f"  Mask {mask:04b}: {action_str}")


if __name__ == "__main__":
    main()
