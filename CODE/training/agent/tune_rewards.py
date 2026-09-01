#!/usr/bin/env python3
"""Tune reward weights (α and β) to find optimal balance.

Tests different α (step reward weight) and β (terminal reward weight)
combinations and evaluates which gives best accuracy/cost trade-off.

α: weight for per-step confidence improvement (acquisition benefit)
β: weight for terminal accuracy (prediction correctness)
"""

import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES,
    unify_categorical_dtypes,
)

from ql_solver import QLearner
from bundle_costs import get_costs_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0],
                        help="α values to try (step reward weight)")
    parser.add_argument("--beta-values", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0],
                        help="β values to try (terminal reward weight)")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Training episodes per configuration")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Evaluation episodes per configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl_agents/tuning"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    train_df = load_split("train", cols, variant="normalized")
    X_train, y_train = prepare_xy(train_df, cols, binary=True)

    test_df = load_split("test", cols, variant="normalized")
    X_test, y_test = prepare_xy(test_df, cols, binary=True)

    unify_categorical_dtypes([X_train, X_test], cols)

    bundle_costs = get_costs_dict()

    # Grid search
    results = []
    configs = list(product(args.alpha_values, args.beta_values))

    print(f"\nTesting {len(configs)} configurations...")
    print(f"  α (step weight): {args.alpha_values}")
    print(f"  β (terminal weight): {args.beta_values}")
    print(f"  Train: {args.n_train} episodes, Eval: {args.n_eval} episodes\n")

    for alpha, beta in tqdm(configs, desc="Tuning"):
        # Train
        learner = QLearner(
            alpha=0.1,  # learning rate, not the reward weight
            gamma=0.99,
            epsilon=0.1,
            step_weight=alpha,  # This is the α parameter
            terminal_weight=beta,  # This is the β parameter
            bundle_costs=bundle_costs,
        )

        learner.train(X_train, y_train, n_episodes=args.n_train, show_progress=False)

        # Evaluate
        eval_stats = learner.evaluate(X_test, y_test, n_episodes=args.n_eval, show_progress=False)

        results.append({
            "alpha": alpha,
            "beta": beta,
            "accuracy": eval_stats["accuracy"],
            "mean_return": eval_stats["mean_return"],
            "mean_steps": eval_stats["mean_steps"],
            "mean_confidence": eval_stats["mean_confidence"],
        })

    # Results to DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("accuracy", ascending=False)

    # Print results
    print("\n" + "=" * 80)
    print("REWARD WEIGHT TUNING RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Save results
    results_path = output_dir / "tuning_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Best configuration
    best = df_results.iloc[0]
    print(f"\nBest configuration:")
    print(f"  α={best['alpha']:.1f}, β={best['beta']:.1f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  Mean return: {best['mean_return']:.4f}")
    print(f"  Mean steps: {best['mean_steps']:.2f}")

    # Recommendation
    print("\nTo train with best configuration:")
    print(f"  python train_rl_agent.py \\")
    print(f"    --n-train 2000 \\")
    print(f"    --step-weight {best['alpha']:.1f} \\")
    print(f"    --terminal-weight {best['beta']:.1f}")


if __name__ == "__main__":
    main()
