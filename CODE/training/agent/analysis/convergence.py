#!/usr/bin/env python3
"""How many training episodes does the agent actually need?

The episode budget used everywhere else (12,000) was chosen for runtime, not
because anything showed the agent had converged by then. This measures it:
accuracy against episode count, from well under one pass over the training
split to two full passes.

Training samples whole shuffled passes, so 36,389 episodes -- the size of the
split -- means every collision seen exactly once, and 72,778 means twice.

If accuracy flattens well before one pass, the 12,000 default is defensible and
sweeps can stay cheap. If it is still climbing at one or two passes, every
headline number in the report is under-trained and should be re-run at the
larger budget.

Usage (from the repo root):
    python CODE/training/agent/analysis/convergence.py
    python CODE/training/agent/analysis/convergence.py --seeds 3
"""

import argparse
import json

import numpy as np
import pandas as pd

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
BEST_PARAMS = Path("models/rl_agents/optuna/best_params.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--cost-scale", type=float, nargs="+", default=[0.01],
                    help="lambda(s). The plateau may sit elsewhere at a "
                         "different operating point: at high lambda the "
                         "agent acquires less, so episodes are shorter and "
                         "each visits fewer states.")
    ap.add_argument("--best-params", type=Path, default=BEST_PARAMS)
    ap.add_argument("--out-dir", type=Path, default=Path("models/rl_agents/convergence"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(args.best_params))["params"]
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)
    n_split = len(X_train)

    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    budgets = [3000, 6000, 12000, 24000, n_split, 2 * n_split]
    notes = {12000: "current default", n_split: "1 full pass",
             2 * n_split: "2 full passes"}

    print(f"training split: {n_split:,} collisions\n")
    rows = []
    for lam in args.cost_scale:
      costs = {i: s * lam for i, s in seconds.items()}
      print(f"--- lambda = {lam} ---")
      print(f"{'episodes':>9}  {'passes':>7}  {'accuracy':>18}  {'bundles':>8}  note",
            flush=True)
      for n in budgets:
          accs, bundles = [], []
          for s in range(args.seeds):
              learner = QLearner(lr=cfg["lr"], gamma=1.0, epsilon=cfg["epsilon"],
                                 step_weight=cfg["step_weight"], terminal_weight=1.0,
                                 bundle_costs=costs, n_conf_bins=cfg["conf_bins"],
                                 seed=42 + s)
              learner.warm_start(one_step_seed(valid_acc, costs,
                                               n_conf_bins=cfg["conf_bins"]))
              learner.train(X_train, y_train, n_episodes=n, show_progress=False)
              ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                                    show_progress=False)
              accs.append(ev["accuracy"]); bundles.append(ev["mean_steps"])
          sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan")
          rows.append(dict(lam=lam, episodes=n, passes=round(n / n_split, 2),
                           accuracy=float(np.mean(accs)), accuracy_sd=sd,
                           bundles=float(np.mean(bundles))))
          print(f"{n:>9}  {n/n_split:>7.2f}  {np.mean(accs):>10.4f} ± {sd:.4f}  "
                f"{np.mean(bundles):>8.2f}  {notes.get(n,'')}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "convergence.csv", index=False)

    for lam, g in df.groupby("lam"):
        base = g.loc[g.episodes == 12000, "accuracy"].iloc[0]
        best = g.accuracy.max()
        n_best = int(g.loc[g.accuracy.idxmax(), "episodes"])
        pooled = g.accuracy_sd.max()
        verdict = ("12,000 is at the plateau" if best - base < 2 * pooled
                   else f"PEAK IS ELSEWHERE ({n_best:,} episodes)")
        print(f"\nlambda={lam}: best {best:.4f} at {n_best:,}; 12,000 gives "
              f"{base:.4f} (diff {best-base:+.4f}, sd {pooled:.4f}) -> {verdict}")

    print(f"\nSaved to {args.out_dir/'convergence.csv'}")


if __name__ == "__main__":
    main()
