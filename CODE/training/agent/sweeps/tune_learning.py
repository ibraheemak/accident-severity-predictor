#!/usr/bin/env python3
"""Sweep the learning-side hyperparameters of the acquisition agent.

Distinct from tune_rewards.py, which sweeps lambda -- that changes WHICH policy
is optimal. This sweeps what the learner needs in order to FIND a good policy,
now that the state includes p_t:

  --conf-bins   how finely max(p_t) is discretised. This is the real knob. Too
                coarse and the agent cannot tell an easy call from a hard one
                (1 bin reproduces the mask-only agent exactly); too fine and the
                1024*bins states get too few visits each to learn anything.
  --n-train     episode budget, for checking convergence rather than tuning.
  --epsilon     exploration rate.

Reports, per configuration, both halves of what a good policy means here:
accuracy and cost (mean bundles, mean seconds), plus two adaptivity measures
that a fixed-subset policy cannot move --

  distinct_sets   how many different acquisition sets the policy used across
                  the eval episodes. A fixed shopping list scores 1.
  varying_masks   masks where the chosen action depends on the confidence bin.

Usage (from the repo root):
    python CODE/training/agent/tune_learning.py
    python CODE/training/agent/tune_learning.py --conf-bins 1 5 --n-train 2000 8000
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import (
    load_split, prepare_xy, bundle_feature_columns, BUNDLE_NAMES,
    unify_categorical_dtypes,
)
from ql_solver import QLearner
from seeding import one_step_seed
from bundle_costs import get_costs_dict

SWEEP_CSV = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")


def episode_seconds(cell, seconds) -> float:
    from ast import literal_eval
    if isinstance(cell, str):
        try:
            cell = literal_eval(cell)
        except (ValueError, SyntaxError):
            return 0.0
    return sum(seconds[BUNDLE_NAMES.index(b)] for b in cell if b in BUNDLE_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf-bins", type=int, nargs="+", default=[1, 3, 5, 8, 12])
    ap.add_argument("--n-train", type=int, nargs="+", default=[8000])
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.15])
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1, help="Q-learning rate")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="1.0 matches the planner's undiscounted objective; "
                         "episodes are capped at 10 steps so nothing diverges")
    ap.add_argument("--cost-scale", type=float, default=0.01, help="lambda")
    ap.add_argument("--step-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/rl_agents/learning_sweep"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)

    seconds = get_costs_dict()
    costs = {i: s * args.cost_scale for i, s in seconds.items()}
    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()

    configs = list(itertools.product(args.conf_bins, args.n_train, args.epsilon))
    print(f"\n{len(configs)} configurations x {args.n_eval} eval episodes "
          f"(lambda={args.cost_scale}, gamma={args.gamma})\n")

    rows = []
    for bins, n_train, eps in tqdm(configs, desc="sweep"):
        learner = QLearner(alpha=args.lr, gamma=args.gamma, epsilon=eps,
                           step_weight=args.step_weight, terminal_weight=1.0,
                           bundle_costs=costs, n_conf_bins=bins, seed=args.seed)
        learner.warm_start(one_step_seed(valid_acc, costs, n_conf_bins=bins))
        learner.train(X_train, y_train, n_episodes=n_train, show_progress=False)
        ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                              show_progress=False)

        res = ev["results"]
        secs = res["acquired_bundles"].map(lambda c: episode_seconds(c, seconds))
        pol = learner.get_policy()
        varying = sum(1 for m in range(1 << 10)
                      if len({pol[(m, b)] for b in range(max(bins, 1))}) > 1)

        learner.save(args.output_dir / f"ql_bins{bins}_n{n_train}_e{eps}.json")
        rows.append({
            "conf_bins": bins, "n_train": n_train, "epsilon": eps,
            "accuracy": round(ev["accuracy"], 4),
            "mean_bundles": round(ev["mean_steps"], 2),
            "mean_seconds": round(secs.mean(), 2),
            # In memory this column holds real lists (unhashable); it is only a
            # string once it has round-tripped through CSV.
            "distinct_sets": len({tuple(sorted(c))
                                  for c in res["acquired_bundles"]}),
            "varying_masks": varying,
            "states": len(learner.Q),
        })

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print("\n" + "=" * 104)
    print("LEARNING SWEEP")
    print("=" * 104)
    print(table.to_string(index=False))

    out = args.output_dir / "learning_sweep.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    best = table.iloc[0]
    print(f"\nBest by accuracy: conf_bins={int(best.conf_bins)} "
          f"n_train={int(best.n_train)} epsilon={best.epsilon} -> "
          f"{best.accuracy:.4f} at {best.mean_bundles} bundles "
          f"({best.mean_seconds}s), {int(best.distinct_sets)} distinct sets")
    print("\ndistinct_sets = 1 means the policy collapsed to a fixed shopping "
          "list and is not using p_t.")


if __name__ == "__main__":
    main()
