#!/usr/bin/env python3
"""Tune the reward function's weights: alpha, beta and lambda.

    R_step(t)   = alpha * dConfidence_t  -  C_i,   C_i = lambda * seconds_i
    R_terminal  = beta  * Accuracy(yhat, y)

Sweeps a grid over the three and reports accuracy alongside cost, because a
reward setting that buys accuracy by asking every question is not a better
setting -- it is a different operating point.

ONE THING TO KNOW BEFORE READING THE RESULTS. These three weights have only
TWO degrees of freedom. Multiplying alpha, beta and lambda all by the same
constant leaves every Q-value scaled by that constant, and the policy picks
actions by comparing Q-values against each other -- so the policy is
unchanged. Only the RATIOS alpha/beta and lambda/beta matter.

Two consequences:

  * Sweeping beta alone is not a third axis. Doubling beta with alpha and
    lambda held fixed is the same experiment as halving alpha and lambda with
    beta fixed. It will appear to "matter" only because it silently rescales
    the other two.
  * The convention is therefore to FIX beta = 1 and vary alpha and lambda.
    Any (alpha, beta, lambda) is reachable that way.

This script sweeps beta anyway, because the redundancy is easier to believe
when measured than when asserted: the summary at the end groups configurations
by their (alpha/beta, lambda/beta) ratios and shows that configurations sharing
a ratio pair behave identically up to sampling noise. That is a result worth
having in writing -- it turns "beta was fixed at 1" from an unexplained choice
into a demonstrated property of the formulation.

Usage (from the repo root):
    python CODE/training/agent/sweeps/tune_rewards.py
    python CODE/training/agent/sweeps/tune_rewards.py --alphas 0 0.05 --betas 1
"""

import argparse
import itertools
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
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.2, 1.0],
                    help="step_weight: weight on the dConfidence shaping term")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="terminal_weight: weight on terminal accuracy")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01],
                    help="cost scale: score points per second of acquisition")
    ap.add_argument("--conf-bins", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--scale-check", action="store_true",
                    help="ignore the grid; instead run k*(alpha,beta,lambda) for "
                         "k in {0.5,1,2}. All three have identical ratios, so if "
                         "the scale-invariance argument holds they must produce "
                         "the same policy.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/rl_agents/reward_sweep"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)

    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    if args.scale_check:
        a0, b0, l0 = args.alphas[0], args.betas[0], args.lambdas[0]
        configs = [(a0 * k, b0 * k, l0 * k) for k in (0.5, 1.0, 2.0)]
    else:
        configs = list(itertools.product(args.alphas, args.betas, args.lambdas))
    print(f"\n{len(configs)} configurations x {args.n_train} train / "
          f"{args.n_eval} eval episodes\n")

    rows = []
    for alpha, beta, lam in tqdm(configs, desc="reward sweep"):
        costs = {i: s * lam for i, s in seconds.items()}
        learner = QLearner(lr=args.lr, gamma=args.gamma, epsilon=args.epsilon,
                           step_weight=alpha, terminal_weight=beta,
                           bundle_costs=costs, n_conf_bins=args.conf_bins,
                           seed=args.seed)
        learner.warm_start(one_step_seed(valid_acc, costs,
                                         n_conf_bins=args.conf_bins,
                                         terminal_weight=beta))
        learner.train(X_train, y_train, n_episodes=args.n_train,
                      show_progress=False)
        ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                              show_progress=False)

        res = ev["results"]
        secs = res["acquired_bundles"].map(lambda c: episode_seconds(c, seconds))
        rows.append({
            "alpha": alpha, "beta": beta, "lambda": lam,
            "a/b": round(alpha / beta, 4), "l/b": round(lam / beta, 5),
            "accuracy": round(ev["accuracy"], 4),
            "bundles": round(ev["mean_steps"], 2),
            "seconds": round(secs.mean(), 2),
            "distinct_sets": len({tuple(sorted(c))
                                  for c in res["acquired_bundles"]}),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("REWARD-WEIGHT SWEEP")
    print("=" * 92)
    print(table.sort_values(["alpha", "beta"]).to_string(index=False))

    out = args.output_dir / "reward_sweep.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # --- the redundancy check ---------------------------------------------
    print("\n" + "=" * 92)
    print("IS BETA A REAL PARAMETER?  (grouping by the ratios alpha/beta, lambda/beta)")
    print("=" * 92)
    grp = table.groupby(["a/b", "l/b"])
    shared = [(k, g) for k, g in grp if len(g) > 1]
    if not shared:
        print("\n  No two configurations shared a ratio pair -- widen --betas to test this.")
    for (ab, lb), g in shared:
        spread = g["accuracy"].max() - g["accuracy"].min()
        print(f"\n  alpha/beta={ab}, lambda/beta={lb}  -> should be the SAME policy")
        for _, r in g.iterrows():
            print(f"     alpha={r.alpha:<6} beta={r.beta:<5} lambda={r['lambda']:<7} "
                  f"accuracy {r.accuracy:.4f}  bundles {r.bundles:.2f}")
        print(f"     accuracy spread: {spread:.4f}"
              f"{'  <- within noise, as predicted' if spread < 0.02 else '  <- LARGER THAN EXPECTED'}")

    print("\nalpha effect at beta=1 (the row that matters for the report):")
    sub = table[(table.beta == 1.0)].sort_values("alpha")
    if len(sub):
        print(sub[["alpha", "accuracy", "bundles", "seconds", "distinct_sets"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
