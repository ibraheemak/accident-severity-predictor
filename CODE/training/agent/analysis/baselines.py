#!/usr/bin/env python3
"""The baselines the learned agent has to beat.

Four non-learned policies, spanning the two axes that matter -- whether a
policy adapts to the individual collision, and whether it was learned:

                        adaptive?   learned?
  no-acquisition           no          no     predict from the timestamp alone
  full-acquisition         no          no     take all ten bundles
  best fixed subset        no          no     best subset fitting a time budget
  random acquisition       no          no     random bundles until the budget
                                              runs out
  static priority          no          no     fixed order, stop when the budget
                                              runs out
  (the RL agent)          YES         YES

All four baselines are non-adaptive, so every comparison here is the agent
against a fixed policy at a matched acquisition time. Note what that means for
scope: these say whether the agent picks a good acquisition SET, not whether
adapting per collision is worth anything. Isolating adaptivity needs a baseline
that walks the same order and differs only in when it stops; a confidence
threshold did that and was removed on 4 October 2026 at the author's request.

EVERYTHING IS SELECTED ON VALIDATION AND REPORTED ON TEST. That applies to the
subset choice and -- easy to miss -- the acquisition ORDER static priority
walks, which is derived from marginal values computed here on validation rather
than read from the value-of-information sweep (whose default metric is a test
column; using it would hand the baseline a test-informed ordering the agent
never had).

Usage (from the repo root):
    python CODE/training/agent/analysis/baselines.py
    python CODE/training/agent/analysis/baselines.py --budgets 0.5 2 5
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import (
    BUNDLE_NAMES, N_BUNDLES, load_split, prepare_xy, bundle_feature_columns,
    unify_categorical_dtypes, mask_bundles,
)
from bundle_costs import get_costs_dict

BUNDLE_COSTS = get_costs_dict()

SWEEP_CSV = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")
CURVE_CSV = Path("docs/figures/lambda_curve/lambda_curve_combined.csv")
MODEL_DIR = Path("models/subsets/normalized_binary")


def mask_seconds(m: int) -> float:
    return sum(BUNDLE_COSTS[i] for i in range(N_BUNDLES) if m >> i & 1)


def value_per_second_order(scores) -> list:
    """Bundles ranked by marginal value per second, best first.

    Marginal value is what a bundle ADDS on average over all subsets lacking
    it; dividing by its acquisition time gives the greedy knapsack ordering.

    All ten bundles are ranked, including any whose marginal value is negative
    (meteo is, at -0.00076). A negative ratio sorts last, so such a bundle is
    only ever reached at a budget large enough to have taken everything else
    first -- the ordering demotes it rather than forbidding it. Dropping it
    instead would make the baseline a policy that already knows which bundles
    are worthless, which is knowledge the agent has to acquire for itself.

    Computed from whatever `scores` is keyed on, which the caller passes as a
    VALIDATION column.
    """
    out = {}
    for i, name in enumerate(BUNDLE_NAMES):
        bit = 1 << i
        out[name] = np.mean([scores[m | bit] - scores[m]
                             for m in range(1 << N_BUNDLES) if not m & bit])
    return sorted(out, key=lambda b: out[b] / BUNDLE_COSTS[BUNDLE_NAMES.index(b)],
                  reverse=True)


def path_predictions(order, splits):
    """Confidence and correctness at each step along the fixed acquisition path.

    The path is the same for every collision, so only len(order)+1 classifiers
    are ever queried -- cheap, unlike the per-collision case.
    """
    masks, secs = [0], [0.0]
    for b in order:
        i = BUNDLE_NAMES.index(b)
        masks.append(masks[-1] | 1 << i)
        secs.append(secs[-1] + BUNDLE_COSTS[i])
    out = {}
    for name, (X, y) in splits.items():
        conf = np.zeros((len(masks), len(y)))
        ok = np.zeros((len(masks), len(y)), dtype=bool)
        for j, m in enumerate(masks):
            mdl = xgb.XGBClassifier()
            mdl.load_model(str(MODEL_DIR / f"{m:03d}.json"))
            p = mdl.predict_proba(X[bundle_feature_columns(mask_bundles(m), "normalized")])
            conf[j], ok[j] = p.max(1), (p.argmax(1) == y)
        out[name] = (conf, ok)
    return out, np.array(secs)


def random_acquisition(test_acc, budget, n_samples=4000, rng=None):
    """Spend the budget on randomly chosen bundles.

    Shuffle the ten bundles and take them in that order while the next one
    still fits, then predict. Averaged over many shuffles, this is what a
    policy gets for spending the SAME time without choosing well -- so the gap
    between it and any real policy is the value of the choice itself, separated
    from the value of the acquisition.

    The counterpart baselines answer different questions: no-acquisition is the
    floor for spending nothing, and static priority spends the budget in a
    deliberate order. This one spends it blindly.

    Cheap, because every subset's accuracy is already in the sweep -- only the
    mask has to be simulated, not the classifier.
    """
    rng = rng or np.random.default_rng(0)
    out = np.empty(n_samples)
    for k in range(n_samples):
        mask, spent = 0, 0.0
        for i in rng.permutation(N_BUNDLES):
            c = BUNDLE_COSTS[i]
            if spent + c <= budget + 1e-9:
                mask |= 1 << int(i)
                spent += c
        out[k] = test_acc[mask]
    return float(out.mean())


def static_priority(ok, secs, budget):
    """Walk the fixed order until the next bundle would break the budget.

    The stopping point is the same for every collision, so this is one fixed
    subset -- always a prefix of the priority order, which is why it cannot beat
    the best fixed subset (that one searches all 1024).
    """
    j = int(np.searchsorted(secs, budget + 1e-9, side="right") - 1)
    return ok[j].mean(), secs[j]


def time_to_reach(menu, targets):
    """Cheapest way each policy has of reaching each accuracy target.

    `menu` is the policy's whole (seconds, accuracy) frontier -- every subset,
    every lambda. For each target we take the least time among the operating
    points that reach it, so the answer is "how long this policy needs to be
    this accurate", NaN if it never gets there.

    This is the budget-matched table read the other way round. It is the more
    stable direction for static priority, whose prefixes cost 0.23 s, then
    1.23 s, then 3.23 s: matched on TIME its accuracy is a step function of
    which prefix the agent's mean time happens to clear.
    """
    secs, accs = np.asarray(menu[0], float), np.asarray(menu[1], float)
    out = []
    for a in targets:
        hit = secs[accs >= a - 1e-12]
        out.append(hit.min() if len(hit) else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=None,
                    help="restrict to these lambdas; default is every lambda "
                         "present in the curve, minus --skip-lambdas")
    ap.add_argument("--skip-lambdas", type=float, nargs="+",
                    default=[0.2, 0.3, 0.5],
                    help="lambdas excluded from the comparison entirely. The "
                         "agent's results at these settings remain in "
                         "docs/figures/lambda_curve/lambda_curve_combined.csv, "
                         "which keeps all 19; only the baseline comparison "
                         "omits them. Pass --skip-lambdas none for the "
                         "complete table.")
    ap.add_argument("--curve-csv", type=Path, default=CURVE_CSV,
                    help="the agent's lambda curve to compare against")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/baselines"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(SWEEP_CSV)
    d["seconds"] = [mask_seconds(m) for m in d["mask"]]
    idx = d.set_index("mask")
    valid_scores = idx["valid_accuracy"].to_dict()
    test_scores = idx["test_accuracy"].to_dict()
    rng = np.random.default_rng(0)

    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    splits = {}
    for s in ("valid", "test"):
        X, y = prepare_xy(load_split(s, cols, variant="normalized"), cols, binary=True)
        unify_categorical_dtypes([X], cols)
        splits[s] = (X, y.values)

    order = value_per_second_order(valid_scores)
    print("fixed acquisition order (marginal value per second, from validation):")
    print("  " + " > ".join(order) + "\n")
    paths, path_secs = path_predictions(order, splits)

    agent = None
    if args.budgets is None and args.curve_csv.exists():
        agent = pd.read_csv(args.curve_csv).sort_values("lam").reset_index(drop=True)
        if args.lambdas:
            agent = agent[agent.lam.isin(args.lambdas)].reset_index(drop=True)
        if args.skip_lambdas and args.skip_lambdas != ["none"]:
            skip = [float(v) for v in args.skip_lambdas]
            gone = sorted(v for v in agent.lam if any(abs(v - k) < 1e-12 for k in skip))
            agent = agent[~agent.lam.isin(gone)].reset_index(drop=True)
            if gone:
                print("excluded lambdas: "
                      + ", ".join(f"{v:g}" for v in gone) + "\n")
        budgets = list(agent.seconds)
    else:
        budgets = args.budgets or [6.26, 4.83, 3.17, 1.69, 0.75, 0.09, 0.003]

    rows = []
    hdr = ((f"{'lambda':>7}" if agent is not None else "") +
           f"{'budget':>8}{'best fixed':>12}{'static':>9}{'random':>9}"
           + (f"{'agent':>9}{'agent-stat':>12}" if agent is not None else ""))
    print(hdr); print("-" * len(hdr))
    for i, b in enumerate(budgets):
        fits = d[d.seconds <= b + 1e-9]
        best = fits.loc[fits.valid_accuracy.idxmax()]

        _, to = paths["test"]
        s_acc, s_sec = static_priority(to, path_secs, b)
        r_acc = random_acquisition(test_scores, b, rng=rng)

        # Static priority matched on ACCURACY as well as on time. Matching on
        # time alone is fragile for it: its prefixes cost 0.23 s, then 1.23 s,
        # then 3.23 s, so which prefix the agent's mean time clears is a step
        # function, and a 0.28 s wobble in the agent between runs moved the
        # target it had to beat by 0.026 -- larger than any effect here.
        s_match = float("nan")
        if agent is not None:
            want = agent.iloc[i].accuracy
            s_reach = [path_secs[j] for j in range(len(path_secs))
                       if to[j].mean() >= want]
            if s_reach:
                s_match = min(s_reach)

        row = dict(budget=round(b, 3), best_fixed=round(best.test_accuracy, 4),
                   best_fixed_seconds=round(best.seconds, 3),
                   best_fixed_subset=best.subset,
                   static=round(s_acc, 4), static_seconds=round(s_sec, 3),
                   random=round(r_acc, 4),
                   )
        line = f"{b:>8.3f}{best.test_accuracy:>12.4f}{s_acc:>9.4f}{r_acc:>9.4f}"
        if agent is not None:
            a = agent.iloc[i]
            line = f"{a.lam:>7g}" + line
            row.update(lam=a.lam, agent=round(a.accuracy, 4), agent_seconds=round(a.seconds, 3),
                       agent_minus_best=round(a.accuracy - best.test_accuracy, 4),
                       agent_minus_static=round(a.accuracy - s_acc, 4),
                       agent_minus_random=round(a.accuracy - r_acc, 4),
                       static_matched_seconds=round(s_match, 3),
                       agent_static_time_ratio=(round(a.seconds / s_match, 3)
                                                if s_match else np.nan))
            line += f"{a.accuracy:>9.4f}{a.accuracy - s_acc:>+12.4f}"
        rows.append(row); print(line, flush=True)

    t = pd.DataFrame(rows)
    floor, full = idx.loc[0], idx.loc[(1 << N_BUNDLES) - 1]
    t["no_acquisition"] = round(floor.test_accuracy, 4)
    t["full_acquisition"] = round(full.test_accuracy, 4)
    t["full_acquisition_seconds"] = round(mask_seconds((1 << N_BUNDLES) - 1), 3)
    t.to_csv(args.out_dir / "baselines.csv", index=False)
    print(f"\n  no-acquisition   {floor.test_accuracy:.4f}  0.00 s")
    print(f"  full-acquisition {full.test_accuracy:.4f}  {mask_seconds(1023):.2f} s")
    if agent is not None:
        def summarise(label, col):
            print(f"  {label:<26}{col.mean():+.4f}   "
                  f"{(col > 0).sum()}/{len(t)} operating points ahead")
        print("\naccuracy comparison -- budget-knobbed policies, matched on time:")
        summarise("agent - best fixed", t.agent_minus_best)
        summarise("agent - static priority", t.agent_minus_static)
        summarise("agent - random acquisition", t.agent_minus_random)

        print("\ntime comparison -- static priority matched to the agent's accuracy:")
        print(f"{'lambda':>7}{'agent acc':>11}{'agent s':>9}{'static s':>10}{'ratio':>7}")
        print("-" * 44)
        for _, r in t.iterrows():
            print(f"{r.lam:>7g}{r.agent:>11.4f}{r.agent_seconds:>9.3f}"
                  f"{r.static_matched_seconds:>10.3f}"
                  f"{r.agent_static_time_ratio:>7.2f}")
        print("  ratio < 1 means the agent reaches that accuracy in less time")

    # ---- the same policies, matched on accuracy instead of on time ----------
    _, to = paths["test"]
    # every menu is (seconds, accuracy) -- keep that order, it is easy to flip
    static_menu = (list(path_secs), [to[j].mean() for j in range(len(path_secs))])
    ag = pd.read_csv(CURVE_CSV) if CURVE_CSV.exists() else None
    menus = {"best fixed": (d.seconds, d.test_accuracy),
             "static priority": static_menu,
             "agent": (ag.seconds, ag.accuracy) if ag is not None else None}
    menus = {k: v for k, v in menus.items() if v is not None}

    targets = np.round(np.arange(0.56, 0.7351, 0.005), 3)
    tt = pd.DataFrame({k: time_to_reach(v, targets) for k, v in menus.items()},
                      index=targets).rename_axis("target_accuracy")
    tt.round(4).to_csv(args.out_dir / "time_to_accuracy.csv")

    print("\nseconds of acquisition needed to reach a given test accuracy")
    print("(each policy read off its own full frontier; '-' = never reaches it)\n")
    print(f"{'accuracy':>9}" + "".join(f"{k:>17}" for k in menus))
    print("-" * (9 + 17 * len(menus)))
    for a in targets[::2]:
        cells = "".join(
            f"{'-':>17}" if np.isnan(tt.loc[a, k]) else f"{tt.loc[a, k]:>17.3f}"
            for k in menus)
        print(f"{a:>9.3f}{cells}", flush=True)

    # A single mean ratio over all targets pools two opposite regimes and comes
    # out near 1.0, which says nothing. Report the ceiling and the crossover.
    print("\nhighest accuracy each policy reaches, and where it is the fastest:")
    for k in menus:
        col = tt[k]
        reached = col.dropna()
        ceiling = reached.index.max() if len(reached) else float("nan")
        wins = (tt.idxmin(axis=1) == k).sum()
        print(f"  {k:<17}ceiling {ceiling:.3f}   fastest at "
              f"{wins}/{len(tt)} targets")
    if "agent" in menus:
        cmp = tt.dropna(subset=["agent", "static priority"])
        faster = cmp.index[cmp.agent < cmp["static priority"]]
        if len(faster):
            print(f"\nthe agent gets there in less time than static priority up "
                  f"to accuracy {faster.max():.3f}; past "
                  f"{tt.agent.dropna().index.max():.3f} it cannot reach the "
                  f"target at any price.")
    print(f"\nSaved to {args.out_dir/'time_to_accuracy.csv'}")
    print(f"\nSaved to {args.out_dir/'baselines.csv'}")


if __name__ == "__main__":
    main()
