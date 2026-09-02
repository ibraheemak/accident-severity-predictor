#!/usr/bin/env python3
"""Solve the mask-level acquisition MDP *exactly* by dynamic programming.

The mask-level MDP is fully known: transitions are deterministic (acquiring
bundle i sets bit i), and the expected terminal reward of stopping at mask m
is that subset classifier's accuracy -- already measured on the VALID split by
bundle_sweep.py for all 1024 masks. So instead of estimating Q-values from
sampled episodes, we can compute the optimal policy in closed form:

    V(m) = max(  valid_score(m),                        # stop and predict
                 max_i not in m [ -lam * seconds_i + V(m | 1<<i) ] )

solved over the subset lattice in decreasing-popcount order. lam is the
exchange rate between acquisition time and score ("1 second of acquisition
time is worth lam score points"). Sweeping lam traces the full cost/accuracy
Pareto frontier, which is the value-of-information sweep used to calibrate
per-bundle costs -- calibration from measured numbers, not guesses.

Two absences from that recursion are deliberate:

* No alpha * dConfidence term. dConfidence is per-collision, while this works
  with per-mask aggregates; and it would not change the answer anyway, since
  summed over an episode the shaping telescopes to alpha*(Conf(m_T) - Conf(m_0))
  with Conf(m_0) constant across policies -- potential-based shaping, which is
  policy-invariant. alpha guides exploration during *learning*; this enumerates.

* No discount factor, unlike QLearner.warm_start() which uses gamma. Only the
  final mask determines the score and acquisition order does not change it, so
  discounting would penalise longer paths for no reason; the undiscounted value
  is the true frontier. warm_start needs gamma only so its seeded Q-values sit
  on the same scale the Q-learner bootstraps with.

Planning uses VALID metrics (choosing a subset on test would leak); the table
reports TEST metrics for the chosen subsets.

Usage:
    python plan_optimal_policy.py
    python plan_optimal_policy.py --metric valid_accuracy
    python plan_optimal_policy.py --lambdas 0 0.01 0.05
"""

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from train_classifier import BUNDLE_NAMES, N_BUNDLES

from bundle_costs import BUNDLE_COSTS

SWEEP_CSV = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")


def solve(valid_score: dict, lam: float, seconds: dict):
    """Exact DP over the subset lattice. Returns (V, best_action) where
    best_action[m] is 0 (stop) or i+1 (acquire bundle i)."""
    n_masks = 1 << N_BUNDLES
    V = [0.0] * n_masks
    A = [0] * n_masks

    # Decreasing popcount: every successor of m has more bits than m,
    # so it is already solved when we reach m.
    for m in sorted(range(n_masks), key=lambda x: bin(x).count("1"), reverse=True):
        best_v, best_a = valid_score[m], 0
        for i in range(N_BUNDLES):
            if not m >> i & 1:
                v = -lam * seconds[i] + V[m | (1 << i)]
                if v > best_v:
                    best_v, best_a = v, i + 1
        V[m], A[m] = best_v, best_a
    return V, A


def trace_path(A):
    """Follow the optimal policy from mask 0; return (final_mask, order)."""
    m, order = 0, []
    while A[m] != 0:
        i = A[m] - 1
        order.append(i)
        m |= 1 << i
    return m, order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="valid_macro_f1",
                    choices=["valid_macro_f1", "valid_accuracy"],
                    help="score used as the terminal reward during planning")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                    help="score-points-per-second exchange rates to sweep")
    args = ap.parse_args()

    df = pd.read_csv(SWEEP_CSV).set_index("mask")
    if len(df) != (1 << N_BUNDLES):
        print(f"Warning: sweep has {len(df)}/{1 << N_BUNDLES} masks; "
              "run bundle_sweep.py to completion first.")

    valid_score = df[args.metric].to_dict()
    seconds = dict(BUNDLE_COSTS)

    # Context: the unconstrained ceiling.
    best_mask_uncon = df[args.metric].idxmax()
    print("=" * 100)
    print(f"EXACT MDP SOLUTION (planning metric: {args.metric})")
    print("=" * 100)
    print(f"\nBaseline (mask 0, timestamp only): "
          f"valid={valid_score[0]:.4f}  test_f1={df.loc[0, 'test_macro_f1']:.4f}")
    print(f"Unconstrained best subset (ignoring cost): mask={best_mask_uncon} "
          f"valid={df.loc[best_mask_uncon, args.metric]:.4f}  "
          f"test_f1={df.loc[best_mask_uncon, 'test_macro_f1']:.4f}  "
          f"bundles={df.loc[best_mask_uncon, 'subset']}")
    print(f"Full acquisition (mask 1023): "
          f"valid={valid_score[1023]:.4f}  test_f1={df.loc[1023, 'test_macro_f1']:.4f}\n")

    rows = []
    for lam in args.lambdas:
        V, A = solve(valid_score, lam, seconds)
        final_mask, order = trace_path(A)
        total_sec = sum(seconds[i] for i in order)
        rows.append({
            "lambda": lam,
            "mask": final_mask,
            "n_bundles": len(order),
            "seconds": round(total_sec, 3),
            "valid": round(valid_score[final_mask], 4),
            "test_f1": round(df.loc[final_mask, "test_macro_f1"], 4),
            "test_acc": round(df.loc[final_mask, "test_accuracy"], 4),
            "acquired (in order)": " -> ".join(BUNDLE_NAMES[i] for i in order) or "(none)",
        })

    table = pd.DataFrame(rows)
    print("PARETO FRONTIER (each row = exactly optimal policy for that lambda):")
    print("-" * 100)
    print(table.to_string(index=False))

    out = Path("docs/figures/policy_analysis")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "exact_policy_frontier.csv", index=False)
    print(f"\nSaved to {out / 'exact_policy_frontier.csv'}")

    print("\nHow to read this: pick the row whose time budget you accept; its "
          "test_f1 is the score the RL agent should reach, and lambda*seconds "
          "are the calibrated bundle costs (--cost-scale lambda in "
          "train_rl_agent.py).")


if __name__ == "__main__":
    main()
