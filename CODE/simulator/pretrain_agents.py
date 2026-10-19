#!/usr/bin/env python3
"""Train the simulator's agents once and commit them, so it never trains again.

The simulator needs a trained policy per cost setting. Training one takes a
couple of minutes, which is a poor first impression for a demo and pointless
repeated work: training is deterministic in (lambda, episodes, seed), so the
same three always produce the same table.

This writes them to CODE/simulator/agents/, which -- unlike models/ -- is not
gitignored, so they travel with the repo and anyone who clones it can run the
simulator immediately with no training step and no access to the 3.5 GB model
cache for the policy itself (the classifiers are still needed to predict).

Agents already in models/rl_agents/ are reused when their hyperparameters match
the current Optuna config and their stored costs imply the right lambda. At the
time of writing only final_lam1.0 qualifies: final/ and final_lam0.1/ predate
the move to 5 confidence bins and would run a different policy from the one the
reported results describe.

Usage (from the repo root):
    python CODE/simulator/pretrain_agents.py
    python CODE/simulator/pretrain_agents.py --lambdas 0.05 --force
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import sys
_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
sys.path[:0] = [str(_HERE), str(_CODE / "training" / "agent" / "core"),
                str(_CODE / "training" / "train")]
from train_classifier import (BUNDLE_NAMES, bundle_feature_columns, load_split,
                              prepare_xy, unify_categorical_dtypes)
from ql_solver import QLearner
from seeding import one_step_seed
from bundle_costs import get_costs_dict
from dispatch import PARAMS, SWEEP, AGENTS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="retrain even if the file already exists")
    args = ap.parse_args()

    AGENTS.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(PARAMS))["params"]
    secs = get_costs_dict()
    valid = pd.read_csv(SWEEP).set_index("mask")["valid_accuracy"].to_dict()

    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    X, y = prepare_xy(load_split("train", cols, variant="normalized"),
                      cols, binary=True)
    unify_categorical_dtypes([X], cols)

    want = (round(cfg["lr"], 6), round(cfg["epsilon"], 6),
            cfg["conf_bins"], round(cfg["step_weight"], 6))

    def existing(lam):
        """An agent already in models/rl_agents/ for this lambda and config."""
        for cand in sorted(Path("models/rl_agents").rglob("*.json")):
            if cand.name == "best_params.json":
                continue
            try:
                d = json.loads(cand.read_text())
                hp, bc = d["hyperparams"], d["bundle_costs"]
            except Exception:
                continue
            got = (round(hp.get("lr", hp.get("alpha", -1)), 6),
                   round(hp.get("epsilon", -1), 6), hp.get("n_conf_bins"),
                   round(hp.get("step_weight", -1), 6))
            implied = float(bc.get("8", 0)) / secs[8]
            if got == want and abs(implied - lam) < 1e-9:
                return cand
        return None

    print(f"config: lr={cfg['lr']:.4f} eps={cfg['epsilon']:.3f} "
          f"bins={cfg['conf_bins']} sw={cfg['step_weight']:.3f}\n")
    for lam in args.lambdas:
        out = AGENTS / f"lam{lam:g}.json"
        if out.exists() and not args.force:
            print(f"  lambda={lam:<5g} already present, skipping")
            continue
        src = existing(lam)
        if src:
            shutil.copy(src, out)
            print(f"  lambda={lam:<5g} copied from {src}")
            continue
        costs = {i: s * lam for i, s in secs.items()}
        a = QLearner(lr=cfg["lr"], gamma=1.0, epsilon=cfg["epsilon"],
                     step_weight=cfg["step_weight"], terminal_weight=1.0,
                     bundle_costs=costs, n_conf_bins=cfg["conf_bins"],
                     seed=args.seed)
        a.warm_start(one_step_seed(valid, costs, n_conf_bins=cfg["conf_bins"]))
        a.train(X, y, n_episodes=args.n_train, show_progress=False)
        a.save(out)
        print(f"  lambda={lam:<5g} trained ({args.n_train:,} episodes) -> {out.name}")

    total = sum(f.stat().st_size for f in AGENTS.glob("*.json")) / 1024 / 1024
    print(f"\n{len(list(AGENTS.glob('*.json')))} agents in {AGENTS}  ({total:.1f} MB)")


if __name__ == "__main__":
    main()
