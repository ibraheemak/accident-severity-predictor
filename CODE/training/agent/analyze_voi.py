#!/usr/bin/env python3
"""Analyze value of information: compare RL agent vs. baseline strategies.

Evaluates multiple strategies for deciding when to acquire features:
  1. Immediate prediction (acquire nothing)
  2. Full acquisition (acquire everything)
  3. Greedy confidence (acquire bundles until confidence > threshold)
  4. Learned RL policy (from Q-learning)

Produces:
  - Summary statistics comparing strategies
  - Plots showing accuracy/cost trade-offs
  - Per-bundle value-of-information curves
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES,
    unify_categorical_dtypes, mask_bundles,
)

from mdp_environment import MDPEnvironment
from ql_solver import QLearner


class StrategyEvaluator:
    """Evaluate different acquisition strategies."""

    @staticmethod
    def strategy_predict_immediate(
        env: MDPEnvironment,
    ) -> Dict:
        """Strategy 1: Predict immediately (no acquisitions)."""
        state, reward, done = env.step(0)
        info = env.get_optimal_path_info()
        return {
            "strategy": "predict_immediate",
            "n_steps": 0,
            "reward": reward,
            "acquired_bundles": [],
            "is_correct": info["is_correct"],
            "final_confidence": info["final_confidence"],
            "y_pred": info["y_pred"],
        }

    @staticmethod
    def strategy_full_acquisition(
        env: MDPEnvironment,
    ) -> Dict:
        """Strategy 2: Acquire all bundles, then predict."""
        while len(env.get_acquirable_actions()) > 0:
            # Acquire the next available bundle
            action = env.get_acquirable_actions()[0]
            env.step(action)

        # Predict
        state, reward, done = env.step(0)
        info = env.get_optimal_path_info()
        return {
            "strategy": "full_acquisition",
            "n_steps": 10,
            "reward": reward,
            "acquired_bundles": list(BUNDLE_NAMES),
            "is_correct": info["is_correct"],
            "final_confidence": info["final_confidence"],
            "y_pred": info["y_pred"],
        }

    @staticmethod
    def strategy_greedy_confidence(
        env: MDPEnvironment,
        threshold: float = 0.8,
    ) -> Dict:
        """Strategy 3: Acquire bundles greedily until confidence > threshold."""
        steps = 0
        while env.current_proba.max() < threshold and len(env.get_acquirable_actions()) > 0:
            # Acquire the bundle that most improves confidence (greedy)
            best_action = None
            best_delta = -np.inf

            for action in env.get_acquirable_actions():
                env_copy = MDPEnvironment(env.X_row, env.y_true)
                env_copy.current_mask = env.current_mask
                env_copy._load_model(env.current_mask)
                env_copy._update_proba()

                state, reward, done = env_copy.step(action)
                delta = state["max_proba"] - env.current_proba.max()
                if delta > best_delta:
                    best_delta = delta
                    best_action = action

            if best_action is None:
                break

            env.step(best_action)
            steps += 1

        # Predict
        state, reward, done = env.step(0)
        info = env.get_optimal_path_info()
        return {
            "strategy": "greedy_confidence",
            "n_steps": steps,
            "reward": reward,
            "acquired_bundles": mask_bundles(env.current_mask),
            "is_correct": info["is_correct"],
            "final_confidence": info["final_confidence"],
            "y_pred": info["y_pred"],
        }

    @staticmethod
    def strategy_learned_policy(
        env: MDPEnvironment,
        policy: Dict[int, int],
    ) -> Dict:
        """Strategy 4: Use learned Q-learning policy."""
        steps = 0
        while True:
            mask = env.current_mask
            if mask in policy:
                action = policy[mask]
            else:
                # Fallback if mask not in policy
                action = 0

            if action == 0:
                # Predict & terminate
                state, reward, done = env.step(action)
                info = env.get_optimal_path_info()
                return {
                    "strategy": "learned_policy",
                    "n_steps": steps,
                    "reward": reward,
                    "acquired_bundles": mask_bundles(mask),
                    "is_correct": info["is_correct"],
                    "final_confidence": info["final_confidence"],
                    "y_pred": info["y_pred"],
                }
            else:
                # Acquire a bundle
                env.step(action)
                steps += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of test samples to evaluate")
    parser.add_argument("--rl-model", type=Path, default=None,
                        help="Path to trained RL model; if None, uses default")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/voi"),
                        help="Directory for output plots")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                        help="Confidence threshold for greedy strategy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (normalized binary)
    print("Loading normalized binary data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    test_df = load_split("test", cols, variant="normalized")
    X_test, y_test = prepare_xy(test_df, cols, binary=True)
    unify_categorical_dtypes([X_test], cols)
    print(f"  Loaded {len(X_test)} samples")

    # Load or initialize RL policy
    if args.rl_model:
        print(f"Loading RL model from {args.rl_model}...")
        learner = QLearner()
        learner.load(args.rl_model)
        rl_policy = learner.get_policy()
    else:
        print("No RL model provided; will evaluate strategies without learned policy")
        rl_policy = None

    # Evaluate strategies
    n_samples = min(args.n_samples, len(X_test))
    sample_indices = np.random.choice(len(X_test), size=n_samples, replace=False)

    results = []
    print(f"\nEvaluating {len(sample_indices)} samples across strategies...")
    pbar = tqdm(sample_indices)

    for idx in pbar:
        for strategy_name in [
            "predict_immediate",
            "full_acquisition",
            "greedy_confidence",
        ]:
            env = MDPEnvironment(X_test.iloc[idx], y_test.iloc[idx])

            if strategy_name == "predict_immediate":
                result = StrategyEvaluator.strategy_predict_immediate(env)
            elif strategy_name == "full_acquisition":
                result = StrategyEvaluator.strategy_full_acquisition(env)
            elif strategy_name == "greedy_confidence":
                result = StrategyEvaluator.strategy_greedy_confidence(
                    env, threshold=args.confidence_threshold
                )
            result["sample_idx"] = idx
            results.append(result)

        # Learned policy (if available)
        if rl_policy:
            env = MDPEnvironment(X_test.iloc[idx], y_test.iloc[idx])
            result = StrategyEvaluator.strategy_learned_policy(env, rl_policy)
            result["sample_idx"] = idx
            results.append(result)

    df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*70)
    print("Strategy Comparison Summary")
    print("="*70)

    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        acc = subset["is_correct"].mean()
        mean_steps = subset["n_steps"].mean()
        mean_conf = subset["final_confidence"].mean()
        mean_reward = subset["reward"].mean()

        print(f"\n{strategy}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Mean acquisition steps: {mean_steps:.2f}")
        print(f"  Mean final confidence: {mean_conf:.4f}")
        print(f"  Mean reward: {mean_reward:.4f}")

    # Save results
    results_path = output_dir / "strategy_comparison.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Summary statistics table
    summary_rows = []
    for strategy in sorted(df["strategy"].unique()):
        subset = df[df["strategy"] == strategy]
        summary_rows.append({
            "strategy": strategy,
            "n_samples": len(subset),
            "accuracy": subset["is_correct"].mean(),
            "mean_steps": subset["n_steps"].mean(),
            "mean_confidence": subset["final_confidence"].mean(),
            "mean_reward": subset["reward"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path}")

    # Print summary
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
