#!/usr/bin/env python3
"""Optuna search over the agent's learning hyperparameters.

Why this rather than the grids in tune_learning.py / tune_rewards.py: a grid
tests a handful of hand-chosen points and cannot see between them. A TPE
sampler searches continuous ranges, concentrates trials where the objective
looks promising, and handles interactions between parameters (a learning rate
that is right at 3 bins may be wrong at 10). This mirrors how the classifier
itself was tuned in CODE/training/tune/, so the two halves of the project use
the same method.

WHAT IS SEARCHED
    lr           Q-learning rate, log-uniform. Controls how far each experience
                 moves an estimate. The terminal reward is 0/1 -- maximally
                 noisy -- so this trades responsiveness against smoothing.
    epsilon      exploration rate for epsilon-greedy.
    conf_bins    how finely max(p_t) is discretised.
    step_weight  the report's alpha: weight on the dConfidence shaping term.

WHAT IS HELD FIXED, AND WHY
    lambda       the price of acquisition time. NOT a tuning parameter -- it is
                 a deployment decision (how much is call-handler time worth?),
                 and every value gives a different, equally valid operating
                 point. Tuning it would just discover that acquiring nothing is
                 cheapest. Fixed here; sweep it separately with tune_rewards.py.
    beta         provably redundant: scaling alpha, beta and lambda together
                 leaves the policy identical (verified to 4 decimals by
                 tune_rewards.py --scale-check). Fixed at 1.0.
    gamma        1.0. Episodes cap at 10 steps so nothing diverges, and only
                 the final mask determines the score, so discounting would
                 penalise longer acquisition paths for no reason.

THE OBJECTIVE is  accuracy  -  lambda * mean acquisition seconds,  averaged
over several seeds. Three deliberate choices in that:

  * ACCURACY, matching the agent's own terminal reward, so hyperparameters are
    selected against the same quantity the agent is trained to maximise. Macro
    F1 is reported alongside every trial (it is the metric the rest of the
    project quotes) but is not what is optimised. On this near-balanced binary
    task the choice is nearly free -- mean |F1 - accuracy| = 0.0001 across the
    1,024 subsets. It would NOT be free on the 3-class task: Fatal is rare, so
    accuracy can be maximised by neglecting it while macro F1 cannot. If this
    moves to 3 classes, revisit both this objective and the reward itself (a
    class-weighted terminal reward is the per-episode analogue of macro F1).
  * MINUS the acquisition cost, because F1 rises monotonically with more
    acquisitions. Tuning on F1 alone would select hyperparameters that make the
    agent ignore lambda and acquire everything -- i.e. would optimise something
    different from what the agent is trained on.
  * The alpha*dConfidence shaping term is EXCLUDED even though it is part of
    the agent's reward, because step_weight is one of the tuned parameters:
    including it would hand free objective to high-alpha trials and make trials
    incomparable. Shaping is a training device, not a deployment goal.

The
averaging is not optional: one training run's accuracy varies by more than the
differences we are trying to resolve, and a sampler handed a noisy objective
will confidently report whichever trial got the luckiest draw. The final
summary prints the seed-to-seed spread of the best trial so you can see whether
its lead is bigger than its own noise.

Usage (from the repo root):
    python CODE/training/agent/sweeps/optuna_tune.py --n-trials 40
    python CODE/training/agent/sweeps/optuna_tune.py --n-trials 60 --seeds 3
"""

import argparse
import json

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2,
                    help="training seeds averaged per trial; more = less noise, "
                         "linearly more time")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1200)
    ap.add_argument("--cost-scale", type=float, default=0.01, help="lambda, held fixed")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/rl_agents/optuna"))
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
    costs = {i: s * args.cost_scale for i, s in seconds.items()}

    def episode_seconds(cell):
        return sum(seconds[BUNDLE_NAMES.index(b)] for b in cell
                   if b in BUNDLE_NAMES)

    def run_once(lr, eps, bins, sw, seed):
        learner = QLearner(lr=lr, gamma=1.0, epsilon=eps, step_weight=sw,
                           terminal_weight=1.0, bundle_costs=costs,
                           n_conf_bins=bins, seed=seed)
        learner.warm_start(one_step_seed(valid_acc, costs, n_conf_bins=bins))
        learner.train(X_train, y_train, n_episodes=args.n_train,
                      show_progress=False)
        ev = learner.evaluate(X_test, y_test, n_episodes=args.n_eval,
                              show_progress=False)
        res = ev["results"]
        macro_f1 = f1_score(res["y_true"], res["y_pred"], average="macro")
        secs = res["acquired_bundles"].map(episode_seconds).mean()
        return ev["accuracy"], macro_f1, secs, ev["mean_steps"]

    def objective(trial):
        lr = trial.suggest_float("lr", 0.005, 0.5, log=True)
        eps = trial.suggest_float("epsilon", 0.02, 0.40)
        bins = trial.suggest_int("conf_bins", 2, 12)
        sw = trial.suggest_float("step_weight", 0.0, 1.0)

        accs, f1s, secs, steps = [], [], [], []
        for s in range(args.seeds):
            ac, f1, sc, st = run_once(lr, eps, bins, sw, seed=42 + s)
            accs.append(ac); f1s.append(f1); secs.append(sc); steps.append(st)
        # score = what we value at this lambda: accuracy net of time spent
        scores = [a - args.cost_scale * c for a, c in zip(accs, secs)]
        trial.set_user_attr("acc_std", float(np.std(scores)))
        trial.set_user_attr("accuracy", float(np.mean(accs)))
        trial.set_user_attr("macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("seconds", float(np.mean(secs)))
        trial.set_user_attr("mean_bundles", float(np.mean(steps)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # SQLite storage so each trial is durable as it completes. A run of this
    # size takes hours; without it, one interruption (or a laptop that sleeps
    # and gets closed) throws away every completed trial. load_if_exists means
    # re-running the same command resumes instead of starting over.
    storage = f"sqlite:///{args.output_dir}/optuna.db"
    study = optuna.create_study(direction="maximize", study_name="rl_agent",
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0))
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        print(f"Resuming: {done} trials already in {storage}")
    remaining = max(args.n_trials - done, 0)
    print(f"\n{args.n_trials} trials x {args.seeds} seeds x {args.n_train} episodes "
          f"(lambda={args.cost_scale} fixed, beta=1, gamma=1)\n")
    # One line per finished trial, flushed immediately, so a long run can be
    # watched with `tail -f` instead of being a black box until it ends.
    def report(study_, trial_):
        best = study_.best_value
        print(f"trial {trial_.number + 1:>3}/{args.n_trials}  "
              f"score {trial_.value:.4f} (+/-{trial_.user_attrs['acc_std']:.4f})  "
              f"acc {trial_.user_attrs['accuracy']:.4f}  "
              f"F1 {trial_.user_attrs['macro_f1']:.4f}  "
              f"{trial_.user_attrs['seconds']:.2f}s  "
              f"bundles {trial_.user_attrs['mean_bundles']:.2f}  |  "
              f"lr {trial_.params['lr']:.4f}  eps {trial_.params['epsilon']:.3f}  "
              f"bins {trial_.params['conf_bins']:>2}  sw {trial_.params['step_weight']:.3f}"
              f"   [best so far {best:.4f}]", flush=True)

    study.optimize(objective, n_trials=remaining, callbacks=[report],
                   show_progress_bar=False)

    df = study.trials_dataframe()
    df.to_csv(args.output_dir / "optuna_trials.csv", index=False)

    b = study.best_trial
    print("\n" + "=" * 84)
    print("BEST TRIAL")
    print("=" * 84)
    for k, v in b.params.items():
        print(f"  {k:<14}{v:.4f}" if isinstance(v, float) else f"  {k:<14}{v}")
    print(f"\n  score (acc - lambda*s) {b.value:.4f}  "
          f"(+/- {b.user_attrs['acc_std']:.4f} across seeds)")
    print(f"  accuracy               {b.user_attrs['accuracy']:.4f}   <- optimised")
    print(f"  macro F1               {b.user_attrs['macro_f1']:.4f}   <- reported")
    print(f"  acquisition time       {b.user_attrs['seconds']:.2f} s "
          f"({b.user_attrs['mean_bundles']:.2f} bundles)")

    # Is the winner actually ahead of the field, or inside its own noise?
    vals = np.array([t.value for t in study.trials if t.value is not None])
    print(f"\n  all trials: best {vals.max():.4f}  median {np.median(vals):.4f}  "
          f"worst {vals.min():.4f}")
    print(f"  spread across trials: {vals.max() - vals.min():.4f}")
    print(f"  best trial's own seed noise: {b.user_attrs['acc_std']:.4f}")
    if b.user_attrs["acc_std"] > (vals.max() - np.median(vals)):
        print("  -> the winner's lead over the median is SMALLER than its own "
              "seed-to-seed noise.\n     Treat these hyperparameters as "
              "'anything reasonable works', not as a tuned optimum.")

    try:
        imp = optuna.importance.get_param_importances(study)
        print("\n  parameter importance (share of objective variance explained):")
        for k, v in imp.items():
            print(f"    {k:<14}{v:.3f}")
    except Exception as e:
        print(f"\n  (importance unavailable: {e})")

    with open(args.output_dir / "best_params.json", "w") as f:
        json.dump({"params": b.params, "score": b.value,
                   "accuracy": b.user_attrs["accuracy"],
                   "macro_f1": b.user_attrs["macro_f1"],
                   "seconds": b.user_attrs["seconds"],
                   "acc_std": b.user_attrs["acc_std"],
                   "mean_bundles": b.user_attrs["mean_bundles"],
                   "cost_scale": args.cost_scale,
                   "n_train": args.n_train, "seeds": args.seeds}, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
