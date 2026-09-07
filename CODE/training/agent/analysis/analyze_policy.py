#!/usr/bin/env python3
"""Analyze a learned acquisition policy.

Shows:
- Which bundles the policy acquires at each state
- Value-of-information per bundle
- Confidence improvement from each acquisition
- When the agent chooses to predict vs acquire
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES,
    unify_categorical_dtypes, mask_bundles,
)

from mdp_environment import MDPEnvironment
from ql_solver import QLearner


def analyze_policy(policy, learner, X_test, y_test, n_samples=500):
    """Analyze what the learned policy does.

    Returns:
        Dictionary with analysis results
    """
    n = min(n_samples, len(X_test))
    sample_indices = np.random.choice(len(X_test), size=n, replace=False)

    # Track decisions
    action_counts = defaultdict(int)  # which actions taken at each mask
    bundle_value = defaultdict(list)  # confidence improvement per bundle
    predict_rates = defaultdict(float)  # predict vs acquire at each mask
    acquisition_sequences = []  # full sequences of acquisitions per episode

    print("\nAnalyzing policy over {} samples...".format(n))
    for sample_idx in tqdm(sample_indices, desc="Analyzing"):
        env = MDPEnvironment(X_test.iloc[sample_idx], y_test.iloc[sample_idx])
        mask = env.current_mask
        sequence = []

        while not env.is_terminal:
            old_confidence = env.current_proba.max()
            action = learner.act(mask, old_confidence)
            action_counts[action] += 1

            if action == 0:
                # Predict
                state, reward, done = env.step(action)
                predict_rates[mask] = predict_rates.get(mask, 0) + 1
                sequence.append((mask, 0, reward))
            else:
                # Acquire
                bundle_idx = action - 1
                state, reward, done = env.step(action)
                confidence_improvement = state["max_proba"] - old_confidence
                bundle_value[bundle_idx].append(confidence_improvement)
                mask = state["mask"]
                sequence.append((mask, action, confidence_improvement))

        acquisition_sequences.append(sequence)

    # Compute statistics
    results = {
        "n_samples": n,
        "action_distribution": dict(action_counts),
        "bundle_value_mean": {
            BUNDLE_NAMES[i]: float(np.mean(vals)) if vals else 0.0
            for i, vals in bundle_value.items()
        },
        "bundle_value_std": {
            BUNDLE_NAMES[i]: float(np.std(vals)) if vals else 0.0
            for i, vals in bundle_value.items()
        },
        "bundle_value_count": {
            BUNDLE_NAMES[i]: len(vals)
            for i, vals in bundle_value.items()
        },
    }

    return results, bundle_value, acquisition_sequences


def print_analysis(results, bundle_value):
    """Print policy analysis nicely."""
    print("\n" + "=" * 80)
    print("POLICY ANALYSIS")
    print("=" * 80)

    print("\nAction Distribution:")
    print(f"  Predict (action 0): {results['action_distribution'].get(0, 0)} times")
    for i in range(10):
        action = i + 1
        count = results['action_distribution'].get(action, 0)
        print(f"  Acquire {BUNDLE_NAMES[i]:20s} (action {action}): {count:5d} times")

    print("\nValue-of-Information per Bundle:")
    print("(Avg confidence improvement when acquiring)")
    print()
    sorted_bundles = sorted(
        results["bundle_value_mean"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for bundle_name, mean_value in sorted_bundles:
        idx = BUNDLE_NAMES.index(bundle_name)
        count = results["bundle_value_count"][bundle_name]
        std = results["bundle_value_std"][bundle_name]

        if count > 0:
            print(f"  [{idx}] {bundle_name:20s}: "
                  f"{mean_value:+.4f} ± {std:.4f} "
                  f"(acquired {count:4d} times)")
        else:
            print(f"  [{idx}] {bundle_name:20s}: never acquired")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model", type=Path, required=True,
                        help="Path to trained RL model")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/policy_analysis"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.rl_model}...")
    learner = QLearner()
    learner.load(args.rl_model)
    policy = learner.get_policy()

    # Load data
    print("Loading normalized binary data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    test_df = load_split("test", cols, variant="normalized")
    X_test, y_test = prepare_xy(test_df, cols, binary=True)
    unify_categorical_dtypes([X_test], cols)

    # Analyze
    results, bundle_value, sequences = analyze_policy(
        policy, learner, X_test, y_test, n_samples=args.n_samples
    )

    # Print results
    print_analysis(results, bundle_value)

    # Save results
    results_path = output_dir / "policy_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save detailed sequence data
    sequences_path = output_dir / "acquisition_sequences.json"
    with open(sequences_path, "w") as f:
        json.dump(
            [{
                "sequence": [(m, a, float(r)) for m, a, r in seq]
                for seq in sequences[:100]  # First 100 for brevity
            }],
            f,
            indent=2
        )
    print(f"Sample sequences saved to {sequences_path}")


if __name__ == "__main__":
    main()
