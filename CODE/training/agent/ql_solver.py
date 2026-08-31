#!/usr/bin/env python3
"""Q-Learning solver for Active Feature Acquisition MDP.

Implements tabular Q-learning to learn an optimal acquisition policy over the
1024 possible acquisition subsets (10 bundles, 2^10 = 1024 states).

Key insight: the state space is discrete and fully enumerable (2^10 = 1024 masks),
so Q-learning can represent the entire state space in a lookup table indexed by
the bitmask. Episodes are short (at most 10 steps), so learning is fast.

The solver learns a policy π(mask) that maps each acquisition subset to the
best next action: either acquire one of the remaining bundles or predict & terminate.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from mdp_environment import MDPEnvironment


class QLearner:
    """Tabular Q-learning solver for the acquisition MDP."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        step_weight: float = 1.0,
        terminal_weight: float = 1.0,
        bundle_costs: Dict[int, float] = None,
        seed: int = 42,
    ):
        """Initialize Q-learner.

        Args:
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon: Exploration rate for epsilon-greedy policy (0-1)
            step_weight: α parameter for step rewards (confidence improvement)
            terminal_weight: β parameter for terminal reward (accuracy)
            bundle_costs: Dict mapping bundle index (0-9) to acquisition cost
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_weight = step_weight
        self.terminal_weight = terminal_weight
        self.bundle_costs = bundle_costs or {i: 0.0 for i in range(10)}
        self.seed = seed

        # Q-table: Q[mask][action] = expected discounted future reward
        # mask: 0-1023 (integer bitmask of acquired bundles)
        # action: 0 (predict) or 1-10 (acquire bundle 0-9)
        self.Q = defaultdict(lambda: defaultdict(float))

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []

        np.random.seed(seed)

    def _get_actions(self, mask: int) -> List[int]:
        """Get valid actions for a given mask.

        Action 0 (predict) is always available.
        Actions 1-10 (acquire bundle i-1) are available if bundle is not yet acquired.
        """
        actions = [0]  # Always can predict
        for i in range(10):
            if not (mask >> i & 1):
                actions.append(i + 1)
        return actions

    def _select_action(self, mask: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            mask: Current state (bitmask)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Action index (0-10)
        """
        actions = self._get_actions(mask)

        if training and np.random.rand() < self.epsilon:
            return np.random.choice(actions)

        # Greedy: pick action with highest Q-value
        q_values = [self.Q[mask][a] for a in actions]
        best_idx = np.argmax(q_values)
        return actions[best_idx]

    def train_episode(
        self,
        env: MDPEnvironment,
    ) -> Tuple[float, int]:
        """Run one training episode with Q-learning updates.

        Args:
            env: Environment instance

        Returns:
            (episode_return, episode_length)
        """
        env.reset()
        mask = env.current_mask
        episode_return = 0.0
        gamma_power = 1.0

        while not env.is_terminal:
            action = self._select_action(mask, training=True)
            state, reward, done = env.step(action)

            episode_return += gamma_power * reward
            gamma_power *= self.gamma

            # Q-learning update
            new_mask = state["mask"]
            if done:
                # Terminal state: no future value
                target = reward
            else:
                # Bootstrap from next state
                next_actions = self._get_actions(new_mask)
                next_q_values = [self.Q[new_mask][a] for a in next_actions]
                target = reward + self.gamma * max(next_q_values)

            self.Q[mask][action] += self.alpha * (target - self.Q[mask][action])
            mask = new_mask

        self.episode_rewards.append(episode_return)
        self.episode_lengths.append(len(env.acquisition_history))
        return episode_return, env.step_count

    def evaluate_episode(self, env: MDPEnvironment) -> Tuple[float, dict]:
        """Run one evaluation episode (greedy policy, no updates).

        Args:
            env: Environment instance

        Returns:
            (episode_return, info)
        """
        env.reset()
        mask = env.current_mask
        episode_return = 0.0
        gamma_power = 1.0

        while not env.is_terminal:
            action = self._select_action(mask, training=False)
            state, reward, done = env.step(action)

            episode_return += gamma_power * reward
            gamma_power *= self.gamma

            mask = state["mask"]

        info = env.get_optimal_path_info()
        return episode_return, info

    def train(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_episodes: int = 100,
        show_progress: bool = True,
    ) -> Dict:
        """Train Q-learner on test set.

        Each episode processes one collision's features (X_test row),
        learns an acquisition policy for that specific collision.

        Args:
            X_test: Feature matrix (rows are collisions)
            y_test: Ground truth labels
            n_episodes: Number of episodes (collisions) to train on
            show_progress: Show progress bar

        Returns:
            Dictionary with training statistics
        """
        n_samples = len(X_test)
        episode_indices = np.random.choice(n_samples, size=n_episodes, replace=True)

        pbar = tqdm(episode_indices, disable=not show_progress, desc="Training")
        for sample_idx in pbar:
            env = MDPEnvironment(
                X_test.iloc[sample_idx],
                y_test.iloc[sample_idx],
            )
            reward, length = self.train_episode(env)
            pbar.set_postfix({"return": f"{reward:.3f}", "steps": length})

        return {
            "n_episodes": n_episodes,
            "mean_return": np.mean(self.episode_rewards[-100:]),
            "mean_length": np.mean(self.episode_lengths[-100:]),
            "q_table_size": len(self.Q),
        }

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_episodes: int = None,
        show_progress: bool = True,
    ) -> Dict:
        """Evaluate learned policy on test set.

        Args:
            X_test: Feature matrix
            y_test: Ground truth labels
            n_episodes: Number of episodes; if None, use all samples
            show_progress: Show progress bar

        Returns:
            Dictionary with evaluation metrics
        """
        if n_episodes is None:
            n_episodes = len(X_test)

        n_samples = len(X_test)
        episode_indices = np.random.choice(n_samples, size=n_episodes, replace=False)

        returns = []
        infos = []

        pbar = tqdm(episode_indices, disable=not show_progress, desc="Evaluating")
        for sample_idx in pbar:
            env = MDPEnvironment(
                X_test.iloc[sample_idx],
                y_test.iloc[sample_idx],
            )
            reward, info = self.evaluate_episode(env)
            returns.append(reward)
            infos.append(info)
            pbar.set_postfix({"return": f"{reward:.3f}", "correct": info["is_correct"]})

        df_results = pd.DataFrame(infos)
        accuracy = df_results["is_correct"].mean()
        mean_steps = df_results["n_steps"].mean()
        mean_confidence = df_results["final_confidence"].mean()

        return {
            "n_episodes": n_episodes,
            "accuracy": accuracy,
            "mean_return": np.mean(returns),
            "mean_steps": mean_steps,
            "mean_confidence": mean_confidence,
            "results": df_results,
        }

    def get_policy(self) -> Dict[int, int]:
        """Extract greedy policy from Q-table.

        Returns:
            Dictionary mapping mask -> best action
        """
        policy = {}
        for mask in range(1 << 10):  # All 1024 possible masks
            actions = self._get_actions(mask)
            q_values = [self.Q[mask][a] for a in actions]
            best_action = actions[np.argmax(q_values)]
            policy[mask] = best_action
        return policy

    def save(self, path: Path):
        """Save Q-table and metadata to JSON.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "hyperparams": {
                "alpha": float(self.alpha),
                "gamma": float(self.gamma),
                "epsilon": float(self.epsilon),
                "step_weight": float(self.step_weight),
                "terminal_weight": float(self.terminal_weight),
            },
            "bundle_costs": {str(k): float(v) for k, v in self.bundle_costs.items()},
            "q_table": {
                str(mask): {str(action): float(q) for action, q in actions.items()}
                for mask, actions in self.Q.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load Q-table and metadata from JSON.

        Args:
            path: Path to load from
        """
        with open(path) as f:
            data = json.load(f)

        self.alpha = data["hyperparams"]["alpha"]
        self.gamma = data["hyperparams"]["gamma"]
        self.epsilon = data["hyperparams"]["epsilon"]
        self.step_weight = data["hyperparams"]["step_weight"]
        self.terminal_weight = data["hyperparams"]["terminal_weight"]

        self.Q.clear()
        for mask_str, actions in data["q_table"].items():
            mask = int(mask_str)
            for action_str, q in actions.items():
                self.Q[mask][int(action_str)] = q
