#!/usr/bin/env python3
"""MDP Environment for Active Feature Acquisition.

Implements the Markov Decision Process environment where an RL agent decides
which features to acquire before making a severity prediction. The state space
includes the free features (call timestamp), acquired bundles (as a bitmask),
and the predicted probability distribution from the matching cached classifier.

The action space is discrete:
  - Action 0 (A_0): Predict & Terminate (end episode, return prediction)
  - Actions 1-10 (A_1 to A_10): Acquire one of the 10 bundles

State transitions are deterministic: acquiring bundle i sets bit i in the mask
and loads the pre-trained classifier for the new mask.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import (
    BINARY_LABELS, BUNDLE_COLS, BUNDLE_NAMES, CATEGORICAL_COLS,
    MODEL_DIR, N_BUNDLES, bundle_feature_columns, mask_bundles,
)


class MDPEnvironment:
    """Markov Decision Process environment for feature acquisition.

    Manages state transitions, model loading, and reward computation.
    Each instance wraps one collision and its features across all possible
    acquisition subsets.
    """

    def __init__(
        self,
        X_row: pd.Series,
        y_true: int,
        variant: str = "normalized",
        model_cache_dir: Path = None,
    ):
        """Initialize the environment for one collision.

        Args:
            X_row: Feature row from a data split (all columns present, no NaNs)
            y_true: Ground truth severity label (0 or 1 for binary Slight vs. Severe)
            variant: Data variant (only "normalized" is supported)
            model_cache_dir: Path to models/subsets/normalized_binary/
        """
        self.X_row = X_row.copy()
        self.y_true = y_true
        self.variant = variant

        # Set model cache directory (always normalized_binary)
        if model_cache_dir is None:
            model_cache_dir = MODEL_DIR / "subsets" / "normalized_binary"
        self.model_cache_dir = Path(model_cache_dir)

        # Initialize state: no bundles acquired yet (mask=0)
        self.current_mask = 0
        self.current_model = None
        self.current_proba = None
        self._load_model(self.current_mask)
        self._update_proba()

        # Episode bookkeeping
        self.step_count = 0
        self.acquisition_history = []
        self.is_terminal = False

    def _load_model(self, mask: int):
        """Load XGBoost model for a given acquisition subset mask."""
        model_path = self.model_cache_dir / f"{mask:03d}.json"
        self.current_model = xgb.XGBClassifier()
        self.current_model.load_model(str(model_path))

    def _get_model_features(self, mask: int) -> list[str]:
        """Feature columns needed for a given mask."""
        bundles = mask_bundles(mask)
        return bundle_feature_columns(bundles, self.variant)

    def reset(self):
        """Reset to initial state (free features only, no acquisitions)."""
        self.current_mask = 0
        self.current_model = None
        self.current_proba = None
        self._load_model(0)
        self.step_count = 0
        self.acquisition_history = []
        self.is_terminal = False
        self._update_proba()

    def _update_proba(self):
        """Update predicted probabilities with current model and features."""
        features = self._get_model_features(self.current_mask)
        # Create a DataFrame with one row, preserving categorical dtypes
        X_df = pd.DataFrame([self.X_row[features]])

        # Ensure categorical columns maintain their proper dtypes
        for col in CATEGORICAL_COLS:
            if col in features and hasattr(self.X_row[col], 'dtype'):
                # If original series had categorical dtype, preserve it
                if hasattr(self.X_row[col].dtype, 'categories'):
                    X_df[col] = X_df[col].astype(self.X_row[col].dtype)

        self.current_proba = self.current_model.predict_proba(X_df)[0]

    def get_state(self) -> dict:
        """Return current state representation.

        Returns:
            Dictionary with:
              - mask: current acquisition bitmask
              - n_acquired: number of bundles acquired
              - acquired_bundles: list of acquired bundle names
              - proba: predicted class probabilities
              - max_proba: highest predicted probability (confidence)
        """
        acquired = mask_bundles(self.current_mask)
        return {
            "mask": self.current_mask,
            "n_acquired": len(acquired),
            "acquired_bundles": acquired,
            "proba": self.current_proba.copy(),
            "max_proba": self.current_proba.max(),
        }

    def get_acquirable_actions(self) -> list[int]:
        """Return list of acquirable bundle indices (actions 1-10).

        An action i is acquirable if bundle i-1 is not yet acquired
        (bit i-1 of mask is 0).
        """
        actions = []
        for i in range(N_BUNDLES):
            if not (self.current_mask >> i & 1):
                actions.append(i + 1)  # Actions 1-10 correspond to bundles 0-9
        return actions

    def step(self, action: int) -> Tuple[dict, float, bool]:
        """Execute one action.

        Args:
            action: 0 to predict & terminate, or 1-10 to acquire bundle i-1

        Returns:
            (new_state, reward_step, is_done)
              - new_state: dict from get_state()
              - reward_step: per-step reward (0 if non-terminal, or final reward if terminal)
              - is_done: True if action was predict & terminate
        """
        if self.is_terminal:
            raise RuntimeError("Episode already terminal; call reset() to start a new episode")

        if action == 0:
            # Predict & Terminate
            self.is_terminal = True
            reward = self._compute_terminal_reward()
            return self.get_state(), reward, True

        elif 1 <= action <= N_BUNDLES:
            # Acquire bundle action-1
            bundle_idx = action - 1

            # Check if bundle is already acquired
            if self.current_mask >> bundle_idx & 1:
                raise ValueError(f"Bundle {BUNDLE_NAMES[bundle_idx]} already acquired")

            # Record old confidence
            old_confidence = self.current_proba.max()

            # Update mask and load new model
            self.current_mask |= (1 << bundle_idx)
            self._load_model(self.current_mask)
            self._update_proba()

            # Compute step reward
            new_confidence = self.current_proba.max()
            delta_confidence = new_confidence - old_confidence
            reward = self._compute_step_reward(
                delta_confidence=delta_confidence,
                bundle_idx=bundle_idx,
            )

            self.step_count += 1
            self.acquisition_history.append(bundle_idx)

            return self.get_state(), reward, False

        else:
            raise ValueError(f"Invalid action {action}; must be 0-{N_BUNDLES}")

    def _compute_step_reward(
        self,
        delta_confidence: float,
        bundle_idx: int,
        alpha: float = 1.0,
        costs: dict = None,
    ) -> float:
        """Compute per-step reward (before terminal action).

        Reward = α·ΔConfidence - C_i

        Args:
            delta_confidence: Change in max predicted probability
            bundle_idx: Index of acquired bundle (0-9)
            alpha: Weight for confidence improvement
            costs: Dict mapping bundle index to cost; if None, all costs are 0

        Returns:
            Step reward value
        """
        if costs is None:
            costs = {i: 0.0 for i in range(N_BUNDLES)}

        cost = costs.get(bundle_idx, 0.0)
        reward = alpha * delta_confidence - cost
        return reward

    def _compute_terminal_reward(
        self,
        beta: float = 1.0,
    ) -> float:
        """Compute terminal reward (predict & terminate action).

        Reward = β·Accuracy(y_true, ŷ)

        Accuracy is 1 if predicted class matches true class, 0 otherwise.

        Args:
            beta: Weight for accuracy

        Returns:
            Terminal reward value
        """
        y_pred = self.current_proba.argmax()
        accuracy = float(y_pred == self.y_true)
        return beta * accuracy

    def get_optimal_path_info(self) -> dict:
        """Return information about the episode once it's terminal.

        Returns:
            Dictionary with:
              - is_correct: whether final prediction was correct
              - n_steps: number of acquisition actions taken
              - acquired_bundles: list of bundle names acquired
              - final_confidence: max predicted probability at termination
              - y_pred: predicted class label
              - y_true: ground truth label
        """
        if not self.is_terminal:
            raise RuntimeError("Episode not terminal yet")

        y_pred = int(self.current_proba.argmax())
        return {
            "is_correct": bool(y_pred == self.y_true),
            "n_steps": int(self.step_count),
            "acquired_bundles": [BUNDLE_NAMES[i] for i in self.acquisition_history],
            "final_confidence": float(self.current_proba.max()),
            "y_pred": y_pred,
            "y_true": int(self.y_true),
        }
