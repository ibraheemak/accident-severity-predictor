#!/usr/bin/env python3
"""Sweep lambda (the seconds->score exchange rate) and check the trained agent
against the exact planner.

Lambda is the only reward parameter that changes what the optimal policy IS:
it prices one second of acquisition time in accuracy points. beta just rescales
the whole reward (policy-invariant), and alpha is exploration shaping whose
per-episode contribution telescopes to alpha*(final - initial confidence). So
tuning means picking lambda, and lambda is a deployment decision (how much is
call-handler time worth?), not something to maximize.

What this script is FOR: verifying that the trained Q-learner actually reaches
the score the exact DP planner says is achievable at each lambda. A large gap
means the learner is under-trained or badly warm-started, not that lambda is
wrong. Compare the "agent" column against "planner" -- they should be within
sampling error of each other.

Usage:
    python tune_rewards.py
    python tune_rewards.py --lambdas 0.002 0.01 0.05 --n-train 1000
"""

import argparse
from pathlib import Path

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
from plan_optimal_policy import solve, trace_path, SWEEP_CSV


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.002, 0.01, 0.05, 0.2],
                    help="score-points-per-second exchange rates to test")
    ap.add_argument("--step-weight", type=float, default=0.05,
                    help="alpha (confidence shaping); kept small on purpose")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                    help="train from a zero Q-table instead of the DP solution")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/rl_agents/lambda_sweep"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and bundle sweep...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)

    sweep = pd.read_csv(SWEEP_CSV).set_index("mask")
    valid_acc = sweep["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    rows = []
    for lam in tqdm(args.lambdas, desc="lambda sweep"):
        costs = {i: s * lam for i, s in seconds.items()}

        # Exact planner: what is achievable at this lambda.
        _, A = solve(valid_acc, lam, seconds)
        planner_mask, planner_order = trace_path(A)

        # Trained agent.
        learner = QLearner(alpha=0.1, gamma=0.99, epsilon=0.1,
                           step_weight=args.step_weight, terminal_weight=1.0,
                           bundle_costs=costs)
        if args.warm_start:
            learner.warm_start(valid_acc)
        learner.train(X_train, y_train, n_episodes=args.n_train,
                      show_progress=False)
        ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                              show_progress=False)

        learner.save(args.output_dir / f"ql_lambda{lam}.json")

        rows.append({
            "lambda": lam,
            "planner_bundles": len(planner_order),
            "planner_seconds": round(sum(seconds[i] for i in planner_order), 3),
            "planner_test_acc": round(sweep.loc[planner_mask, "test_accuracy"], 4),
            "agent_bundles": round(ev["mean_steps"], 2),
            "agent_test_acc": round(ev["accuracy"], 4),
            "gap": round(ev["accuracy"] - sweep.loc[planner_mask, "test_accuracy"], 4),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("LAMBDA SWEEP -- trained agent vs exact planner")
    print("=" * 90)
    print(table.to_string(index=False))

    out = args.output_dir / "lambda_sweep.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print("\n'gap' near 0 means the learner reached the planner's optimum. "
          "Large negative gaps mean under-training, not a bad lambda.")


if __name__ == "__main__":
    main()
