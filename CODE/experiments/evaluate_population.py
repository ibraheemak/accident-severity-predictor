#!/usr/bin/env python3
"""evaluate_population.py

Reports a trained classifier's accuracy on the NATURAL (real-world) class
distribution, alongside the oversampled-test-set numbers we normally report.

Why this exists: stats19_download.R keeps all Fatal collisions but caps
Serious/Slight per year, so the train/valid/test splits are severity-enriched
(~44% Severe) rather than population-representative (~21% Severe). Plain
accuracy on that oversampled test set (~70%) is accuracy on a harder-than-
reality mix. Papers that report ~80% accuracy are almost always measuring on
the natural distribution, where the Slight majority alone gets you ~79%.

This script recovers the population number for OUR model by weighting each
test row by the inverse of its sampling fraction:

    weight_i = 1 / f(year_i, severity_i)
      f(year, Fatal)   = 1.0                       (Fatal kept in full)
      f(year, Serious) = 2800 / serious_total(year)
      f(year, Slight)  = 5600 / slight_total(year)

    natural_accuracy = sum(weight_i * correct_i) / sum(weight_i)

The per-year pre-cap totals are parsed from the run's *_meta.txt. Population
accuracy is ALWAYS printed next to the minority-class recall it depends on --
a high field-accuracy that comes purely from predicting the Slight majority
(and missing the Severe cases the system exists to catch) is not a good model.

Usage:
    python evaluate_population.py --binary
    python evaluate_population.py                      # 3-class
    python evaluate_population.py --data-variant masked --binary
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "train"))
from train_classifier import (
    BINARY_LABELS, DATA_VARIANT_DIRS, MODEL_DIR, SEVERITY_LABELS,
    feature_columns, load_split, prepare_xy, subset_name, unify_categorical_dtypes,
)

# Severity caps applied at download (stats19_download.R). Fatal is uncapped.
SERIOUS_CAP = 2800
SLIGHT_CAP = 5600
META_LINE = re.compile(
    r"(\d{4}):\s*Fatal=\d+.*?Serious=\d+ \(of (\d+)\)\s+Slight=\d+ \(of (\d+)\)"
)


def parse_sampling_fractions(meta_path: str) -> dict:
    """year -> {1: f_fatal, 2: f_serious, 3: f_slight}, the fraction of each
    severity's population that survived the download cap that year."""
    fracs = {}
    with open(meta_path) as f:
        for line in f:
            m = META_LINE.search(line)
            if m:
                year, serious_total, slight_total = int(m[1]), int(m[2]), int(m[3])
                fracs[year] = {
                    1: 1.0,
                    2: SERIOUS_CAP / serious_total,
                    3: SLIGHT_CAP / slight_total,
                }
    if not fracs:
        raise ValueError(f"No per-year sampling lines parsed from {meta_path}")
    return fracs


def population_weights(sev3: np.ndarray, year: np.ndarray, fracs: dict) -> np.ndarray:
    """Inverse-sampling weight per row, reconstructing the natural population."""
    w = np.empty(len(sev3), dtype=float)
    for i in range(len(sev3)):
        w[i] = 1.0 / fracs[int(year[i])][int(sev3[i])]
    return w


def weighted_accuracy(y_true, y_pred, w) -> float:
    return float(np.average((np.asarray(y_pred) == np.asarray(y_true)), weights=w))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", choices=["aadf", "osm", "meteo"],
                         default=["aadf", "osm", "meteo"])
    parser.add_argument("--data-variant", choices=list(DATA_VARIANT_DIRS), default="normalized")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--meta", default=None,
                         help="Path to a stats19 *_meta.txt (default: latest under data/raw/stats19/).")
    args = parser.parse_args()

    labels = BINARY_LABELS if args.binary else SEVERITY_LABELS
    name = subset_name(args.sources)
    file_stem = name if args.data_variant == "masked" else f"{name}_{args.data_variant}"
    if args.binary:
        file_stem += "_binary"

    meta_path = args.meta or sorted(glob.glob("data/raw/stats19/stats19_*/*_meta.txt"))[-1]
    fracs = parse_sampling_fractions(meta_path)
    print(f"Sampling fractions parsed from {meta_path} ({len(fracs)} years)")

    cols = feature_columns(args.sources, args.data_variant)
    test = load_split("test", cols + ["collision_index", "collision_year"], args.data_variant)
    X, y = prepare_xy(test, cols, args.binary)
    unify_categorical_dtypes([X], cols)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"classifier_{file_stem}.json")
    y_pred = model.predict(X)

    sev3 = test["collision_severity"].astype(int).to_numpy()
    year = test["collision_year"].astype(int).to_numpy()
    w = population_weights(sev3, year, fracs)

    # Class mix, oversampled vs natural
    y = np.asarray(y)
    print("\n=== Class mix ===")
    for c in range(len(labels)):
        os_frac = float((y == c).mean())
        pop_frac = float(w[y == c].sum() / w.sum())
        print(f"  {labels[c]:<24} oversampled test {100*os_frac:5.1f}%   natural {100*pop_frac:5.1f}%")

    # Metrics: as-measured (oversampled) vs natural (population-weighted)
    print("\n=== Accuracy: oversampled test set  vs  natural population ===")
    os_acc = accuracy_score(y, y_pred)
    pop_acc = weighted_accuracy(y, y_pred, w)
    os_bal = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    print(f"  plain accuracy          oversampled {100*os_acc:5.1f}%   natural {100*pop_acc:5.1f}%")
    print(f"  balanced accuracy       {100*os_bal:5.1f}%   (distribution-independent)")
    print(f"  macro F1                {macro_f1:.4f}   (distribution-independent)")

    # The recall(s) the natural accuracy leans on -- always shown alongside.
    print("\n=== Minority-class recall the natural accuracy depends on ===")
    if args.binary:
        sev_recall = recall_score(y, y_pred, pos_label=1)
        trivial = float(w[y == 0].sum() / w.sum())
        print(f"  Severe recall (accidents actually caught) = {100*sev_recall:.1f}%")
        print(f"  'always predict Slight' natural accuracy  = {100*trivial:.1f}%  <- the bar to beat")
        print(f"\n  Read together: {100*pop_acc:.1f}% field-accuracy while catching "
              f"{100*sev_recall:.1f}% of severe accidents.")

        # The two operating points, so the accuracy/recall trade is explicit:
        # the model's default (balanced) threshold vs the threshold that
        # maximises NATURAL accuracy (which drifts toward always-Slight).
        p_sev = model.predict_proba(X)[:, 1]
        ts = np.linspace(0.1, 0.95, 171)
        accs = [weighted_accuracy(y, (p_sev >= t).astype(int), w) for t in ts]
        t_best = float(ts[int(np.argmax(accs))])
        pred_best = (p_sev >= t_best).astype(int)
        print("\n=== Same model, two operating points on the natural distribution ===")
        print(f"  {'operating point':<34}{'nat. accuracy':>14}{'Severe recall':>15}")
        print(f"  {'default (catches severe cases)':<34}{100*pop_acc:>13.1f}%{100*sev_recall:>14.1f}%")
        print(f"  {f'accuracy-max threshold={t_best:.2f}':<34}"
              f"{100*np.max(accs):>13.1f}%{100*recall_score(y, pred_best, pos_label=1):>14.1f}%")
        print("  ^ the second row is the '~80% accuracy' figure -- bought by missing "
              "most severe accidents.")
    else:
        for c in range(len(labels)):
            rec = recall_score(y, y_pred, labels=[c], average="macro")
            print(f"  {labels[c]:<24} recall = {100*rec:.1f}%")


if __name__ == "__main__":
    main()
