#!/usr/bin/env python3
"""Does Q-learning improve on its own seed, or just churn it?

The agent is initialised from one_step_seed and then trained. Everything
reported so far measures the combination. This isolates the two:

  seed only   the seeded table evaluated directly, zero training episodes.
              A greedy one-step heuristic, no learning at all.
  trained     the same seed after the usual training budget.

The question matters for what can be claimed. If training does not beat its
seed, the honest description is "a one-step heuristic that Q-learning failed to
improve", not "a trained RL agent" -- and that is a very different sentence in
a thesis.

There is reason to expect little: the terminal reward is 0/1, and at 12,000
episodes over 1024*bins states each state is visited on the order of 16 times,
against the ~8,000 needed to resolve the 0.01-scale differences that separate
good subsets.

Run at several lambda so the answer is not an artefact of one operating point.

Usage (from the repo root):
    python CODE/training/agent/analysis/seed_vs_trained.py
    python CODE/training/agent/analysis/seed_vs_trained.py --lambdas 0.01 --seeds 5
"""

import argparse
import json
from ast import literal_eval

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
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/seed_vs_trained"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(BEST_PARAMS))["params"]
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)
    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()

    def ep_secs(cell):
        if isinstance(cell, str):
            cell = literal_eval(cell)
        return sum(seconds[BUNDLE_NAMES.index(b)] for b in cell if b in BUNDLE_NAMES)

    def build(lam, seed):
        costs = {i: s * lam for i, s in seconds.items()}
        L = QLearner(lr=cfg["lr"], gamma=1.0, epsilon=cfg["epsilon"],
                     step_weight=cfg["step_weight"], terminal_weight=1.0,
                     bundle_costs=costs, n_conf_bins=cfg["conf_bins"], seed=seed)
        L.warm_start(one_step_seed(valid_acc, costs, n_conf_bins=cfg["conf_bins"]))
        return L

    def score(L):
        ev = L.evaluate(X_test, y_test, n_episodes=args.n_eval, show_progress=False)
        res = ev["results"]
        return (ev["accuracy"], res["acquired_bundles"].map(ep_secs).mean(),
                len({tuple(sorted(c)) for c in res["acquired_bundles"]}))

    print(f"{args.seeds} seeds, {args.n_train} training episodes\n")
    print(f"{'lambda':>7}{'seed only':>22}{'trained':>22}{'training gains':>17}")
    print("-" * 70)
    rows = []
    for lam in args.lambdas:
        s_acc, t_acc, s_sec, t_sec, s_sets, t_sets = [], [], [], [], [], []
        for k in range(args.seeds):
            L = build(lam, 42 + k)
            a, sec, sets = score(L)                      # zero training
            s_acc.append(a); s_sec.append(sec); s_sets.append(sets)
            L.train(X_train, y_train, n_episodes=args.n_train, show_progress=False)
            a, sec, sets = score(L)                      # after training
            t_acc.append(a); t_sec.append(sec); t_sets.append(sets)
        sm, ss = np.mean(s_acc), np.std(s_acc, ddof=1)
        tm, ts = np.mean(t_acc), np.std(t_acc, ddof=1)
        gap, pooled = tm - sm, float(np.hypot(ss, ts))
        rows.append(dict(lam=lam, seed_acc=sm, seed_sd=ss, seed_seconds=np.mean(s_sec),
                         seed_sets=np.mean(s_sets), trained_acc=tm, trained_sd=ts,
                         trained_seconds=np.mean(t_sec), trained_sets=np.mean(t_sets),
                         gap=gap, pooled_sd=pooled))
        print(f"{lam:>7}{sm:>14.4f} ± {ss:.4f}{tm:>14.4f} ± {ts:.4f}"
              f"{gap:>+12.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "seed_vs_trained.csv", index=False)

    print("\nper-lambda verdict (2x pooled sd):")
    for _, r in df.iterrows():
        v = ("training helps" if r.gap > 2 * r.pooled_sd else
             "training HURTS" if r.gap < -2 * r.pooled_sd else
             "no measurable effect")
        print(f"  lambda={r.lam:<6} {r.gap:+.4f}  (pooled sd {r.pooled_sd:.4f})  -> {v}")
    print(f"\nacquisition time:  seed {df.seed_seconds.mean():.2f}s  ->  "
          f"trained {df.trained_seconds.mean():.2f}s")
    print(f"distinct sets:     seed {df.seed_sets.mean():.1f}  ->  "
          f"trained {df.trained_sets.mean():.1f}")
    print(f"\nSaved to {args.out_dir/'seed_vs_trained.csv'}")


if __name__ == "__main__":
    main()
