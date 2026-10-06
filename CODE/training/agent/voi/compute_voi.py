#!/usr/bin/env python3
"""Value-of-information analysis over the 1,024 acquisition subsets.

Reads the bundle sweep produced by CODE/training/train/bundle_sweep.py and asks
what each acquisition bundle is actually worth. Three tables come out:

  bundle_value.csv     per bundle: mean marginal value across all 512 subsets
                       that lack it, its value acquired alone, how often it
                       helps at all, and its value per second of acquisition time
  saturation.csv       per subset size k: the best, mean and worst score
                       achievable with exactly k bundles
  tier_value.csv       the same marginal value rolled up by acquisition kind

"Marginal value of bundle b" is the honest version of the question: not what b
scores on its own, but what it ADDS on average once you already hold some other
arbitrary combination -- averaged over all 2^9 = 512 subsets that do not contain
it. A bundle can look strong alone and still be near-worthless marginally, if
whatever signal it carries is already supplied by something cheaper.

Usage (from the repo root):
    python CODE/training/agent/voi/compute_voi.py
    python CODE/training/agent/voi/compute_voi.py --metric valid_macro_f1
"""

import argparse

import numpy as np
import pandas as pd

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import BUNDLE_NAMES, BUNDLE_TIERS, N_BUNDLES
from bundle_costs import get_costs_dict

BUNDLE_COSTS = get_costs_dict()

SWEEP_CSV = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")
OUT_DIR = Path("docs/figures/voi_sweep")


def per_bundle_value(score: dict, metric: str) -> pd.DataFrame:
    """Marginal value of each bundle, averaged over every subset lacking it."""
    baseline = score[0]
    rows = []
    for i, name in enumerate(BUNDLE_NAMES):
        bit = 1 << i
        deltas = [score[m | bit] - score[m]
                  for m in range(1 << N_BUNDLES) if not m & bit]
        deltas = np.array(deltas)
        seconds = BUNDLE_COSTS[i]
        rows.append({
            "bundle": name,
            "tier": BUNDLE_TIERS[name],
            "mean_marginal": deltas.mean(),
            "std_marginal": deltas.std(),
            "min_marginal": deltas.min(),
            "max_marginal": deltas.max(),
            "alone": score[bit] - baseline,
            "helps_frac": float((deltas > 0).mean()),
            "n_subsets": len(deltas),
            "seconds": seconds,
            "value_per_second": deltas.mean() / seconds,
        })
    return (pd.DataFrame(rows)
            .sort_values("mean_marginal", ascending=False)
            .reset_index(drop=True))


def saturation(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best / mean / worst score achievable at each number of bundles held."""
    g = df.groupby("n_bundles")[metric]
    out = pd.DataFrame({
        "n_bundles": g.max().index,
        "best": g.max().values,
        "mean": g.mean().values,
        "worst": g.min().values,
        "n_subsets": g.size().values,
    })
    best_mask = df.loc[df.groupby("n_bundles")[metric].idxmax()]
    out["best_subset"] = best_mask["subset"].values
    out["gain_over_prev"] = out["best"].diff().fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="test_macro_f1",
                    choices=["test_macro_f1", "valid_macro_f1",
                             "test_accuracy", "valid_accuracy"])
    ap.add_argument("--sweep-csv", type=Path, default=SWEEP_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.sweep_csv)
    if len(df) != (1 << N_BUNDLES):
        print(f"WARNING: sweep has {len(df)}/{1 << N_BUNDLES} masks — "
              "run bundle_sweep.py to completion for a complete analysis.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score = df.set_index("mask")[args.metric].to_dict()

    print("=" * 88)
    print(f"VALUE-OF-INFORMATION SWEEP   (metric: {args.metric})")
    print("=" * 88)
    print(f"\nFree-information floor (no bundles):  {score[0]:.4f}")
    print(f"Full acquisition (all ten bundles):   {score[(1 << N_BUNDLES) - 1]:.4f}")
    # Selecting a "best subset" must be done on validation -- picking the
    # argmax of the test column would be selecting on the test set, and the
    # number it reports would be meaningless. Choose on valid, report test.
    sel_metric = args.metric.replace("test_", "valid_")
    idx = df.set_index("mask")
    best_mask = int(idx[sel_metric].idxmax())
    best_row = idx.loc[best_mask]
    print(f"Best subset (chosen on {sel_metric}, mask {best_mask}): "
          f"{score[best_mask]:.4f}"
          f"  [{best_row['n_bundles']} bundles: {best_row['subset']}]")
    print(f"Whole action space is worth:          "
          f"+{score[(1 << N_BUNDLES) - 1] - score[0]:.4f}\n")

    bv = per_bundle_value(score, args.metric)
    print("MARGINAL VALUE PER BUNDLE (averaged over all 512 subsets lacking it)")
    print("-" * 88)
    print(f"{'bundle':<23}{'kind':<14}{'marginal':>10}{'alone':>9}"
          f"{'helps':>8}{'secs':>8}{'val/sec':>10}")
    for _, r in bv.iterrows():
        print(f"{r.bundle:<23}{r.tier:<14}{r.mean_marginal:>+10.4f}"
              f"{r.alone:>+9.4f}{r.helps_frac:>7.0%}{r.seconds:>8.3f}"
              f"{r.value_per_second:>10.3f}")

    sat = saturation(df, args.metric)
    print("\nSATURATION — best achievable score at each number of bundles")
    print("-" * 88)
    for _, r in sat.iterrows():
        bar = "#" * int(round((r.best - sat.best.min()) /
                              max(sat.best.max() - sat.best.min(), 1e-9) * 34))
        print(f"  k={int(r.n_bundles):2d}  best={r.best:.4f}  "
              f"(+{r.gain_over_prev:.4f})  {bar}")

    tier = (bv.groupby("tier")
              .agg(mean_marginal=("mean_marginal", "mean"),
                   total_seconds=("seconds", "sum"),
                   n_bundles=("bundle", "count"))
              .sort_values("mean_marginal", ascending=False)
              .reset_index())
    print("\nBY ACQUISITION KIND")
    print("-" * 88)
    for _, r in tier.iterrows():
        print(f"  {r.tier:<14} mean marginal {r.mean_marginal:+.4f}   "
              f"{int(r.n_bundles)} bundles   {r.total_seconds:.3f}s total")

    for name, table in (("bundle_value", bv), ("saturation", sat),
                        ("tier_value", tier)):
        path = args.out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
