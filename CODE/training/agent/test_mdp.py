#!/usr/bin/env python3
"""Quick test of MDP environment and QL solver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES,
    unify_categorical_dtypes,
)

from mdp_environment import MDPEnvironment
from ql_solver import QLearner


def test_environment():
    """Test that environment loads and runs basic episodes."""
    print("Testing MDPEnvironment...")

    # Load minimal test data
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    test_df = load_split("test", cols, variant="normalized")
    X_test, y_test = prepare_xy(test_df, cols, binary=True)
    unify_categorical_dtypes([X_test], cols)

    # Test one collision
    env = MDPEnvironment(X_test.iloc[0], y_test.iloc[0])
    print(f"  Initial state mask: {env.current_mask}")
    print(f"  Predicted probabilities: {env.current_proba}")
    print(f"  Acquirable actions: {env.get_acquirable_actions()}")

    # Take one acquisition step
    state, reward, done = env.step(1)
    print(f"  After acquiring bundle 0:")
    print(f"    New mask: {state['mask']}")
    print(f"    Confidence: {state['max_proba']:.4f}")
    print(f"    Step reward: {reward:.4f}")

    # Predict & terminate
    env.reset()
    state, reward, done = env.step(0)
    print(f"  After predict & terminate:")
    print(f"    Terminal reward: {reward:.4f}")
    print(f"    Is done: {done}")
    print(f"    Episode info: {env.get_optimal_path_info()}")
    print("  ✓ MDPEnvironment tests passed\n")


def test_qlearner():
    """Test that QL solver initializes and runs."""
    print("Testing QLearner...")

    # Load minimal test data
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    train_df = load_split("train", cols, variant="normalized")
    X_train, y_train = prepare_xy(train_df, cols, binary=True)
    unify_categorical_dtypes([X_train], cols)

    # Just test first 10 samples to verify setup
    X_train = X_train.head(10)
    y_train = y_train.head(10)

    learner = QLearner(alpha=0.1, gamma=0.99, epsilon=0.1)
    print(f"  Q-table initialized")

    # Train for a few episodes
    print(f"  Training for 5 episodes...")
    train_stats = learner.train(X_train, y_train, n_episodes=5, show_progress=False)
    print(f"    Mean return: {train_stats['mean_return']:.4f}")
    print(f"    Q-table size: {train_stats['q_table_size']}")

    # Get policy
    policy = learner.get_policy()
    print(f"  Policy for mask=0 (free info only): action={policy[0]}")
    print(f"  Policy for mask=1 (bundle 0 acquired): action={policy[1]}")

    print("  ✓ QLearner tests passed\n")


if __name__ == "__main__":
    test_environment()
    test_qlearner()
    print("All tests passed! ✓")
