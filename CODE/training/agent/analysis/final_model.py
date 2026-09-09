#!/usr/bin/env python3
"""Train the headline models at full episode budget, with error bars.

The sweeps in sweeps/ run a reduced episode budget so that dozens of
configurations fit in a few hours. That is fine for ranking configurations but
wrong for reporting: an agent trained on 5,000 episodes scores below the same
agent trained on 12,000, so a "tuned" number taken straight from a sweep can
look worse than an untuned number measured properly. This script is the last
step -- full budget, several seeds, mean +/- std.

It trains three configurations rather than one, because a single number cannot
answer the questions the report needs answered:

  tuned        the best configuration from sweeps/optuna_tune.py
  default      the hand-picked settings used before any tuning. If tuned does
               not beat this, the tuning bought nothing -- worth knowing, and
               not something a single run of `tuned` alone would reveal.
  mask-only    identical to tuned except conf_bins=1, which removes p_t from
               the state. This is THE ablation: it isolates whether
               per-collision adaptivity does anything, as opposed to the agent
               simply having found a good fixed subset.

The `distinct_sets` column is the one to read for that last question. A policy
that ignores p_t necessarily acquires the same bundles for every collision and
scores 1; anything above 1 means the agent is genuinely adapting per call.

Usage (from the repo root):
    python CODE/training/agent/analysis/final_model.py
    python CODE/training/agent/analysis/final_model.py --seeds 5 --n-train 20000
"""

import argparse
import json
from ast import literal_eval

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

# Settings used before any tuning existed -- the "did tuning help?" control.
DEFAULT_CFG = dict(lr=0.1, epsilon=0.15, conf_bins=3, step_weight=0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--cost-scale", type=float, default=0.01, help="lambda")
    ap.add_argument("--best-params", type=Path, default=BEST_PARAMS)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/rl_agents/final"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tuned = json.load(open(args.best_params))["params"]
    configs = {
        "tuned":     dict(tuned),
        "default":   dict(DEFAULT_CFG),
        "mask-only": dict(tuned, conf_bins=1),
    }

    print("Loading data...")
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X_train, y_train = prepare_xy(
        load_split("train", cols, variant="normalized"), cols, binary=True)
    X_test, y_test = prepare_xy(
        load_split("test", cols, variant="normalized"), cols, binary=True)
    unify_categorical_dtypes([X_train, X_test], cols)

    valid_acc = pd.read_csv(SWEEP_CSV).set_index("mask")["valid_accuracy"].to_dict()
    seconds = get_costs_dict()
    costs = {i: s * args.cost_scale for i, s in seconds.items()}

    def ep_secs(cell):
        if isinstance(cell, str):
            cell = literal_eval(cell)
        return sum(seconds[BUNDLE_NAMES.index(b)] for b in cell if b in BUNDLE_NAMES)

    print(f"\n{len(configs)} configurations x {args.seeds} seeds x "
          f"{args.n_train} episodes (lambda={args.cost_scale})\n")
    for name, c in configs.items():
        print(f"  {name:<10} lr={c['lr']:.4f} eps={c['epsilon']:.3f} "
              f"bins={c['conf_bins']} sw={c['step_weight']:.3f}")
    print()

    rows, best = [], (-1, None, None)
    for name, cfg in configs.items():
        per_seed = []
        for s in tqdm(range(args.seeds), desc=f"{name:<10}", leave=False):
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
            rec = dict(
                accuracy=ev["accuracy"],
                macro_f1=f1_score(res["y_true"], res["y_pred"], average="macro"),
                bundles=ev["mean_steps"],
                seconds=res["acquired_bundles"].map(ep_secs).mean(),
                distinct_sets=len({tuple(sorted(c)) for c in res["acquired_bundles"]}),
            )
            per_seed.append(rec)
            if rec["accuracy"] > best[0]:
                best = (rec["accuracy"], name, learner)
            print(f"    {name:<10} seed {42+s}  acc {rec['accuracy']:.4f}  "
                  f"F1 {rec['macro_f1']:.4f}  {rec['seconds']:.2f}s  "
                  f"sets {rec['distinct_sets']}", flush=True)

        d = pd.DataFrame(per_seed)
        rows.append({
            "config": name,
            "accuracy": f"{d.accuracy.mean():.4f} ± {d.accuracy.std():.4f}",
            "macro_f1": f"{d.macro_f1.mean():.4f} ± {d.macro_f1.std():.4f}",
            "bundles": f"{d.bundles.mean():.2f}",
            "seconds": f"{d.seconds.mean():.2f}",
            "distinct_sets": f"{d.distinct_sets.mean():.1f}",
            "_acc": d.accuracy.mean(), "_std": d.accuracy.std(),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"FINAL MODELS  ({args.seeds} seeds x {args.n_train} episodes, "
          f"lambda={args.cost_scale})")
    print("=" * 96)
    print(table.drop(columns=["_acc", "_std"]).to_string(index=False))

    t = table.set_index("config")
    gap = t.loc["tuned", "_acc"] - t.loc["default", "_acc"]
    pooled = np.hypot(t.loc["tuned", "_std"], t.loc["default", "_std"])
    print(f"\ntuned - default = {gap:+.4f}   (pooled seed std {pooled:.4f})")
    print("  -> " + ("tuning bought a real improvement."
                     if abs(gap) > 2 * pooled else
                     "INSIDE NOISE: tuning did not measurably help. Report the "
                     "agent as insensitive to these hyperparameters."))

    gap2 = t.loc["tuned", "_acc"] - t.loc["mask-only", "_acc"]
    pooled2 = np.hypot(t.loc["tuned", "_std"], t.loc["mask-only", "_std"])
    print(f"\ntuned - mask-only = {gap2:+.4f}   (pooled seed std {pooled2:.4f})")
    print("  -> " + ("adaptivity (p_t in the state) is a real effect."
                     if abs(gap2) > 2 * pooled2 else
                     "INSIDE NOISE: cannot claim adaptivity helps from this run."))

    table.drop(columns=["_acc", "_std"]).to_csv(
        args.output_dir / "final_models.csv", index=False)
    if best[2] is not None:
        best[2].save(args.output_dir / "best_agent.json")
    print(f"\nSaved to {args.output_dir}/  (best single run: {best[1]}, "
          f"acc {best[0]:.4f})")


if __name__ == "__main__":
    main()
