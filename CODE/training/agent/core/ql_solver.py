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
        n_conf_bins: int = 3,
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
            n_conf_bins: Confidence bins in the state (p_t discretised). 3 was
                best in the 12k-episode sweep; more bins split the table into
                states too rarely visited to learn from. 1 disables the
                confidence term entirely (mask-only ablation).
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_weight = step_weight
        self.terminal_weight = terminal_weight
        self.bundle_costs = bundle_costs or {i: 0.0 for i in range(10)}
        self.n_conf_bins = n_conf_bins
        self.seed = seed

        # Q-table: Q[mask][action] = expected discounted future reward
        # mask: 0-1023 (integer bitmask of acquired bundles)
        # action: 0 (predict) or 1-10 (acquire bundle 0-9)
        self.Q = defaultdict(lambda: defaultdict(float))

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []

        np.random.seed(seed)

    def warm_start(self, q_table: Dict[tuple, Dict[int, float]]):
        """Seed the Q-table with precomputed values (plan-then-learn).

        This method only loads values -- it contains no logic for producing
        them. The seed lives in seeding.py (one_step_seed), so
        the learner has no planner inside it and any source of initial values
        works here equally well.

        Seeding matters because cold-start epsilon-greedy essentially never
        discovers a specific multi-bundle acquisition sequence from a 0/1
        reward delivered only at the end of the episode; it plateaus after a
        single bundle. With a seed, training refines rather than discovers.

        Args:
            q_table: {(mask, confidence_bin): {action: q}}.
        """
        for key, actions in q_table.items():
            for action, q in actions.items():
                self.Q[key][action] = q

    def conf_bin(self, confidence: float) -> int:
        """Discretise the classifier's current confidence into a bin index.

        The MDP formulation defines the state as (x_free, x_acquired, m_t, p_t),
        but a tabular value function needs discrete states and p_t is a
        continuous distribution. Binning max(p_t) is the tabular-compatible way
        to keep it: it is what lets two collisions holding the SAME bundles be
        told apart, which is the whole point of acquiring actively rather than
        following a fixed shopping list.

        n_conf_bins=1 collapses this to nothing and recovers the mask-only
        policy, which is the ablation baseline.
        """
        if self.n_conf_bins <= 1:
            return 0
        # Binary max-probability lives in [0.5, 1.0]; three-class in [1/3, 1].
        lo = 0.5
        frac = (confidence - lo) / (1.0 - lo)
        return int(np.clip(frac * self.n_conf_bins, 0, self.n_conf_bins - 1))

    def _key(self, mask: int, confidence: float):
        """State key for the Q-table: the acquisition mask plus a confidence bin."""
        return (mask, self.conf_bin(confidence))

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

    def _select_action(self, key, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            key: (mask, confidence_bin) state key
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Action index (0-10)
        """
        actions = self._get_actions(key[0])

        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(actions))

        q_values = [self.Q[key][a] for a in actions]
        return actions[int(np.argmax(q_values))]

    def act(self, mask: int, confidence: float) -> int:
        """Greedy action for a live state. The accessor other scripts should
        use, so they never need to know how the Q-table is keyed."""
        return self._select_action(self._key(mask, confidence), training=False)

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
        key = self._key(env.current_mask, env.current_proba.max())
        episode_return = 0.0
        gamma_power = 1.0

        while not env.is_terminal:
            action = self._select_action(key, training=True)
            state, reward, done = env.step(action)

            episode_return += gamma_power * reward
            gamma_power *= self.gamma

            # Q-learning update
            new_key = self._key(state["mask"], state["max_proba"])
            if done:
                # Terminal state: no future value
                target = reward
            else:
                # Bootstrap from next state
                next_actions = self._get_actions(new_key[0])
                next_q_values = [self.Q[new_key][a] for a in next_actions]
                target = reward + self.gamma * max(next_q_values)

            self.Q[key][action] += self.alpha * (target - self.Q[key][action])
            key = new_key

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
        key = self._key(env.current_mask, env.current_proba.max())
        episode_return = 0.0
        gamma_power = 1.0

        while not env.is_terminal:
            action = self._select_action(key, training=False)
            state, reward, done = env.step(action)

            episode_return += gamma_power * reward
            gamma_power *= self.gamma

            key = self._key(state["mask"], state["max_proba"])

        info = env.get_optimal_path_info()
        return episode_return, info

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_episodes: int = 100,
        show_progress: bool = True,
    ) -> Dict:
        """Train Q-learner on training set.

        Each episode processes one collision's features (X_train row) with
        epsilon-greedy exploration and updates the Q-table.

        Args:
            X_train: Feature matrix (rows are collisions)
            y_train: Ground truth labels
            n_episodes: Number of training episodes (collisions sampled with replacement)
            show_progress: Show progress bar

        Returns:
            Dictionary with training statistics
        """
        n_samples = len(X_train)
        episode_indices = np.random.choice(n_samples, size=n_episodes, replace=True)

        pbar = tqdm(episode_indices, disable=not show_progress, desc="Training")
        for sample_idx in pbar:
            env = MDPEnvironment(
                X_train.iloc[sample_idx],
                y_train.iloc[sample_idx],
                step_weight=self.step_weight,
                terminal_weight=self.terminal_weight,
                bundle_costs=self.bundle_costs,
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
                step_weight=self.step_weight,
                terminal_weight=self.terminal_weight,
                bundle_costs=self.bundle_costs,
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

    def get_policy(self) -> Dict[tuple, int]:
        """Extract the greedy policy from the Q-table.

        Returns:
            {(mask, confidence_bin): best action}. Prefer act(mask, confidence)
            for live use -- it does the binning for you.
        """
        policy = {}
        for mask in range(1 << 10):
            actions = self._get_actions(mask)
            for b in range(max(self.n_conf_bins, 1)):
                q_values = [self.Q[(mask, b)][a] for a in actions]
                policy[(mask, b)] = actions[int(np.argmax(q_values))]
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
                "n_conf_bins": int(self.n_conf_bins),
            },
            "bundle_costs": {str(k): float(v) for k, v in self.bundle_costs.items()},
            # Keys are "mask:bin" -- JSON has no tuple keys.
            "q_table": {
                f"{mask}:{b}": {str(a): float(q) for a, q in actions.items()}
                for (mask, b), actions in self.Q.items()
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
        self.n_conf_bins = data["hyperparams"].get("n_conf_bins", 1)

        self.Q.clear()
        for key_str, actions in data["q_table"].items():
            mask, _, b = key_str.partition(":")
            key = (int(mask), int(b) if b else 0)
            for action_str, q in actions.items():
                self.Q[key][int(action_str)] = q
