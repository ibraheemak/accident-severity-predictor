#!/usr/bin/env python3
"""Sweep lambda and record how the agent's behaviour changes.

lambda prices one second of acquisition time in accuracy points. It is not a
parameter to be optimised -- it is a deployment decision, and every value is a
valid operating point. So the result here is not "the best lambda" but a MENU:
what accuracy is available at what acquisition cost, and which questions the
agent stops asking as time gets more expensive.

The grid is uniform: lambda = 0, 0.01, ..., 0.10. Uniform spacing is chosen so
the sweep can be stated in one line and cannot be suspected of having had its
points placed after seeing the results. Worth knowing where the action is: each
bundle stops paying for itself at lambda = (its marginal value)/(its seconds),
and the three expensive ask-the-caller bundles break even at 0.010, 0.019 and
0.032 -- so the transitions all fall in the first four points, and past ~0.04
only the millisecond lookups remain and the curve flattens. lambda = 0 is
included as the "time is free" anchor: the accuracy ceiling when the agent has
no reason ever to stop acquiring.

Hyperparameters are FIXED across all lambda rather than re-tuned per point.
That is a measured decision, not laziness: Optuna's top five trials spread
0.0060 while the winning trial's own seed-to-seed noise was 0.0086, and at full
budget the tuned configuration and the pre-tuning default differed by 0.0048
against a pooled sd of 0.0198. Hyperparameters are inside the noise here; SEEDS
are not (the same configuration gave 0.7392 and 0.7040 on two seeds). The
compute is therefore spent on repetitions, which buy error bars, rather than on
re-tuning, which buys nothing measurable.

Outputs (docs/figures/lambda_curve/):
    lambda_curve.csv       per lambda: accuracy, macro F1, seconds, bundles,
                           distinct acquisition sets -- mean +/- sd over seeds
    bundle_frequency.csv   per lambda: how often each bundle is acquired
    lambda_curve.png       the operating curve, and which bundles survive

Usage (from the repo root):
    python CODE/training/agent/analysis/lambda_curve.py
    python CODE/training/agent/analysis/lambda_curve.py --seeds 5 --n-train 20000
"""

import argparse
import json
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
BEST_PARAMS = Path("models/rl_agents/optuna/best_params.json")
LAMBDAS = [round(0.01 * i, 2) for i in range(11)]   # 0, 0.01, ..., 0.10


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=12000,
                    help="default 12,000 episodes (~1/3 of a pass over the 36,389-collision training split). Not arbitrary: analysis/convergence.py sweeps 3k to two full passes and finds accuracy plateaus by 12,000 -- a full pass gives 0.7184 vs 0.7196 here, and two passes is slightly worse")
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--best-params", type=Path, default=BEST_PARAMS)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/lambda_curve"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(args.best_params))["params"]
    print(f"Fixed hyperparameters: lr={cfg['lr']:.4f} eps={cfg['epsilon']:.3f} "
          f"bins={cfg['conf_bins']} sw={cfg['step_weight']:.3f}")

    print("Loading data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)

    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    def as_list(cell):
        return literal_eval(cell) if isinstance(cell, str) else cell

    print(f"\n{len(args.lambdas)} lambdas x {args.seeds} seeds x {args.n_train} "
          f"episodes = {len(args.lambdas)*args.seeds} runs\n")

    rows, freq_rows = [], []
    for lam in args.lambdas:
        costs = {i: s * lam for i, s in seconds.items()}
        per_seed, counts = [], Counter()
        for s in range(args.seeds):
            learner = QLearner(lr=cfg["lr"], gamma=1.0, epsilon=cfg["epsilon"],
                               step_weight=cfg["step_weight"], terminal_weight=1.0,
                               bundle_costs=costs, n_conf_bins=cfg["conf_bins"],
                               seed=42 + s)
            learner.warm_start(one_step_seed(valid_acc, costs,
                                             n_conf_bins=cfg["conf_bins"]))
            learner.train(X_train, y_train, n_episodes=args.n_train,
                          show_progress=False)
            ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                                  show_progress=False)
            res = ev["results"]
            sets = res["acquired_bundles"].map(as_list)
            for bl in sets:
                counts.update(bl)
            per_seed.append(dict(
                accuracy=ev["accuracy"],
                macro_f1=f1_score(res["y_true"], res["y_pred"], average="macro"),
                bundles=ev["mean_steps"],
                secs=sets.map(lambda b: sum(seconds[BUNDLE_NAMES.index(x)]
                                            for x in b if x in BUNDLE_NAMES)).mean(),
                distinct=len({tuple(sorted(b)) for b in sets}),
            ))
        d = pd.DataFrame(per_seed)
        n_ep = args.seeds * args.n_eval
        rows.append(dict(lam=lam,
                         accuracy=d.accuracy.mean(), accuracy_sd=d.accuracy.std(),
                         macro_f1=d.macro_f1.mean(), macro_f1_sd=d.macro_f1.std(),
                         seconds=d.secs.mean(), seconds_sd=d.secs.std(),
                         bundles=d.bundles.mean(), distinct_sets=d.distinct.mean()))
        freq_rows.append(dict(lam=lam,
                              **{b: counts[b] / n_ep for b in BUNDLE_NAMES}))
        print(f"  λ={lam:<6} acc {d.accuracy.mean():.4f} ± {d.accuracy.std():.4f}   "
              f"{d.secs.mean():5.2f}s   {d.bundles.mean():.2f} bundles   "
              f"{d.distinct.mean():.0f} sets", flush=True)

    curve = pd.DataFrame(rows)
    freq = pd.DataFrame(freq_rows)
    curve.to_csv(args.out_dir / "lambda_curve.csv", index=False)
    freq.to_csv(args.out_dir / "bundle_frequency.csv", index=False)

    print("\n" + "=" * 96)
    print("OPERATING CURVE")
    print("=" * 96)
    print(curve[["lam", "accuracy", "accuracy_sd", "macro_f1", "seconds",
                 "bundles", "distinct_sets"]].round(4).to_string(index=False))

    print("\n" + "=" * 96)
    print("HOW OFTEN EACH BUNDLE IS ACQUIRED  (1.00 = every call)")
    print("=" * 96)
    show = freq.set_index("lam")[BUNDLE_NAMES]
    print(show.round(2).to_string())
    print("\nRead down a column to see the lambda at which the agent gives up on "
          "that bundle.\nThe three ask-the-caller bundles are the expensive ones "
          "and should fall away first.")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
