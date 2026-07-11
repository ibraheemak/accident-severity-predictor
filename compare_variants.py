#!/usr/bin/env python3
"""compare_variants.py

Trains and evaluates every combination of:
  - source subset: all 8 subsets of {aadf, osm, meteo}
  - data variant: masked, normalized, raw (see train_classifier.py docstring)
  - target: 3-class severity, binary (Slight vs Severe)

i.e. 8 * 3 * 2 = 48 XGBoost runs, using the exact same feature-selection,
loading, and training logic as train_classifier.py (imported, not
duplicated). Prints a full results table plus a variant-vs-variant pivot
(one row per source subset, one column per variant) so it's easy to see
whether the "masked" variant systematically trails "normalized"/"raw" for
matched subsets -- i.e. whether partial-observability masking is actually
hurting accuracy, as opposed to the effect just being noise.

--cross-eval-masked adds 8 * 2 = 16 more runs: train on "masked" (as
usual) but evaluate valid/test on "normalized" (clean, one-row-per-
collision, no deliberate NaNs) instead of on "masked"'s own valid/test.
The masked test set is a mix of many missingness levels (copy_id -1..9,
mask_rate 0..1), so a masked-trained model's macro F1 on it reflects
average performance across that whole missingness distribution -- not
what it scores when given full information at inference time. This
isolates that: same trained model, scored against the clean target the
"normalized"/"raw" columns in the main table already use, so it's a fair
apples-to-apples comparison of the masked-trained model against the
masked-free ones. Labeled "masked_evalon_normalized" in the results.

No models or per-run metrics files are saved (unlike train_classifier.py);
only the aggregate comparison table, written to --output.

Usage:
    python compare_variants.py                             # full grid, full training
    python compare_variants.py --n-estimators 50            # fast smoke test
    python compare_variants.py --cross-eval-masked          # + masked-train/normalized-eval runs
"""

import argparse
import itertools
import time

import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from train_classifier import (
    BINARY_LABELS, DATA_VARIANT_DIRS, SEVERITY_LABELS, SOURCE_COLS, evaluate,
    feature_columns, load_split, prepare_xy, subset_name,
    unify_categorical_dtypes,
)


def all_source_subsets() -> list[list[str]]:
    sources = list(SOURCE_COLS)
    subsets = []
    for r in range(len(sources) + 1):
        subsets.extend(list(c) for c in itertools.combinations(sources, r))
    return subsets


def run_one(sources: list[str], variant: str, binary: bool,
            n_estimators: int, max_depth: int, learning_rate: float,
            early_stopping_rounds: int, seed: int, eval_variant: str = None) -> dict:
    eval_variant = eval_variant or variant
    cols = feature_columns(sources, variant)
    train_df = load_split("train", cols, variant)
    valid_df = load_split("valid", cols, eval_variant)
    test_df = load_split("test", cols, eval_variant)

    X_train, y_train = prepare_xy(train_df, cols, binary)
    X_valid, y_valid = prepare_xy(valid_df, cols, binary)
    X_test, y_test = prepare_xy(test_df, cols, binary)
    unify_categorical_dtypes([X_train, X_valid, X_test], cols)

    sample_weight = compute_sample_weight("balanced", y_train)

    if binary:
        objective_kwargs = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        objective_kwargs = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"}

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
        **objective_kwargs,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    fit_time = time.time() - t0

    labels = BINARY_LABELS if binary else SEVERITY_LABELS
    valid_metrics = evaluate(model, X_valid, y_valid, "valid", labels, verbose=False)
    test_metrics = evaluate(model, X_test, y_test, "test", labels, verbose=False)

    label = variant if eval_variant == variant else f"{variant}_evalon_{eval_variant}"
    return {
        "sources": subset_name(sources),
        "variant": label,
        "train_variant": variant,
        "eval_variant": eval_variant,
        "binary": binary,
        "n_train": len(train_df),
        "valid_macro_f1": valid_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_iteration": int(model.best_iteration),
        "fit_time_s": fit_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/variant_comparison.csv")
    parser.add_argument("--cross-eval-masked", action="store_true",
                         help="Also train on 'masked' but evaluate valid/test on 'normalized' "
                              "(clean, unmasked) instead of masked's own valid/test -- adds "
                              "8 subsets * 2 targets = 16 more runs, labeled "
                              "'masked_evalon_normalized'. Isolates whether masking during "
                              "training helps/hurts full-information inference, instead of "
                              "being diluted across the masked test set's missingness levels.")
    args = parser.parse_args()

    subsets = all_source_subsets()
    variants = list(DATA_VARIANT_DIRS)
    binaries = [False, True]

    # (sources, train_variant, eval_variant, binary) for every run.
    run_plan = [
        (sources, variant, variant, binary)
        for binary in binaries for variant in variants for sources in subsets
    ]
    if args.cross_eval_masked:
        run_plan += [
            (sources, "masked", "normalized", binary)
            for binary in binaries for sources in subsets
        ]
    total = len(run_plan)

    results = []
    for run_i, (sources, variant, eval_variant, binary) in enumerate(run_plan, start=1):
        name = subset_name(sources)
        print(f"[{run_i}/{total}] sources={name:20s} variant={variant:10s} "
              f"eval_variant={eval_variant:10s} binary={binary}", end="  ", flush=True)
        r = run_one(
            sources, variant, binary,
            args.n_estimators, args.max_depth, args.learning_rate,
            args.early_stopping_rounds, args.seed, eval_variant,
        )
        results.append(r)
        print(f"valid_f1={r['valid_macro_f1']:.4f}  test_f1={r['test_macro_f1']:.4f}  "
              f"best_iter={r['best_iteration']}  fit={r['fit_time_s']:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nFull results written to {args.output}")

    for binary in binaries:
        label = "binary (Slight vs Severe)" if binary else "3-class"
        sub = df[df["binary"] == binary].sort_values(["variant", "test_macro_f1"], ascending=[True, False])
        print(f"\n{'=' * 70}\nResults ({label}), sorted by variant then test macro F1\n{'=' * 70}")
        print(sub[["variant", "sources", "n_train", "valid_macro_f1", "test_macro_f1", "best_iteration", "fit_time_s"]]
              .to_string(index=False))

        pivot = sub.pivot(index="sources", columns="variant", values="test_macro_f1")
        col_order = [v for v in variants if v in pivot.columns] + \
                    [c for c in pivot.columns if c not in variants]
        pivot = pivot[col_order]
        print(f"\n{'-' * 70}\ntest macro F1 by variant ({label})\n{'-' * 70}")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
