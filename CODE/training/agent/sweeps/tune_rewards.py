#!/usr/bin/env python3
"""Sweep lambda -- the seconds-to-score exchange rate -- and report what the
trained agent achieves at each setting.

Lambda is the only reward parameter that changes what the optimal policy IS:
it prices one second of acquisition time in accuracy points. beta just rescales
the whole reward (policy-invariant, leave at 1.0), and alpha is exploration
shaping whose per-episode contribution telescopes to alpha*(final - initial
confidence). So tuning means picking lambda -- and lambda is a deployment
decision (how much is call-handler time worth?), not a quantity to maximise.
Every row below is a legitimate operating point; you choose one by deciding
what acquisition time your dispatch scenario can absorb, not by taking the max.

To judge whether a row's accuracy is GOOD you need the best fixed subset at the
same acquisition budget -- a filter and an argmax over the 1,024 already-scored
subsets in the bundle sweep. That baseline is not computed here yet.

Usage:
    python tune_rewards.py
    python tune_rewards.py --lambdas 0.002 0.01 0.05 --n-train 8000
"""

import argparse
from ast import literal_eval

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
    if isinstance(cell, str):
        try:
            cell = literal_eval(cell)
        except (ValueError, SyntaxError):
            return 0.0
    return sum(seconds[BUNDLE_NAMES.index(b)] for b in cell if b in BUNDLE_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.002, 0.01, 0.05, 0.2],
                    help="score-points-per-second exchange rates to test")
    ap.add_argument("--step-weight", type=float, default=0.05,
                    help="alpha (confidence shaping); kept small on purpose")
    ap.add_argument("--conf-bins", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--no-seed", dest="seed_table", action="store_false",
                    help="train from a zero Q-table; expect it to plateau after "
                         "one bundle (see seeding.py)")
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

    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    rows = []
    for lam in tqdm(args.lambdas, desc="lambda sweep"):
        costs = {i: s * lam for i, s in seconds.items()}
        learner = QLearner(alpha=0.1, gamma=args.gamma, epsilon=args.epsilon,
                           step_weight=args.step_weight, terminal_weight=1.0,
                           bundle_costs=costs, n_conf_bins=args.conf_bins)
        if args.seed_table:
            learner.warm_start(one_step_seed(valid_acc, costs,
                                             n_conf_bins=args.conf_bins))
        learner.train(X_train, y_train, n_episodes=args.n_train,
                      show_progress=False)
        ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                              show_progress=False)

        res = ev["results"]
        secs = res["acquired_bundles"].map(lambda c: episode_seconds(c, seconds))
        learner.save(args.output_dir / f"ql_lambda{lam}.json")
        rows.append({
            "lambda": lam,
            "accuracy": round(ev["accuracy"], 4),
            "mean_bundles": round(ev["mean_steps"], 2),
            "mean_seconds": round(secs.mean(), 2),
            "mean_confidence": round(ev["mean_confidence"], 4),
            "distinct_sets": len({tuple(sorted(c))
                                  for c in res["acquired_bundles"]}),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("LAMBDA SWEEP -- what the agent does at each price of time")
    print("=" * 84)
    print(table.to_string(index=False))

    out = args.output_dir / "lambda_sweep.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print("\nHigher lambda should buy fewer seconds at lower accuracy. "
          "distinct_sets = 1 means the policy collapsed to a fixed shopping "
          "list and is ignoring p_t.")


if __name__ == "__main__":
    main()
