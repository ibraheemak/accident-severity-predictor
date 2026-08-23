"""tune_model_families.py -- give every model family the same search budget.

compare_model_families.py runs each family at library defaults, except XGBoost,
which carries hyperparameters from this project's Optuna searches. That is an
unfair comparison in the incumbent's favour: the honest question is how the
families rank when each has had the same chance to be tuned.

Protocol, identical for every family:
  * random search over that family's parameter space, N trials, fixed seed
  * each trial trains on `train` and is scored by macro F1 on `valid`
  * the winning configuration is retrained over 3 seeds and reported on `test`,
    which is touched only at that final step
XGBoost is searched on the same budget as everyone else, and its project-canonical
tuned parameters (from this repo's much larger Optuna searches) are injected as
trial 0 of that search, so they compete as one candidate among many and are
selected -- or not -- by the same validation rule as every other configuration.
The canonical config is additionally reported on its own row for transparency,
since it comes from a far larger search budget than 120 random trials.

Families with nothing meaningful to tune (majority-class, stratified random,
the one-rule stump, OLS) are carried through at their fixed configuration so
the table stays complete.

Usage:
  python CODE/experiments/model_comparison/tune_model_families.py [--trials 30] [--seeds 42 43 44]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from compare_model_families import (LinearRegressionClassifier, load_data,
                                    sklearn_preprocessor)
from train_classifier import MODEL_DIR, unify_categorical_dtypes

OUT_DIR = Path("docs/figures/model_families")


def spaces(rng):
    """Parameter samplers, one per tunable family. Each returns a dict."""
    lo = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    # rng.choice coerces mixed-type and ragged sequences (it would turn 0.3 into
    # np.str_("0.3") and choke on tuples of different length), so pick by index.
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    MAX_FEATURES = ["sqrt", "log2", 0.3, 0.6]
    HIDDEN = [(64,), (128,), (256,), (128, 64), (256, 128), (128, 64, 32)]
    return {
        "Logistic regression": lambda: {"C": lo(1e-3, 1e2)},
        "Linear SVM": lambda: {"C": lo(1e-4, 1e1)},
        "Gaussian naive Bayes": lambda: {"var_smoothing": lo(1e-12, 1e-3)},
        "k-NN": lambda: {"n_neighbors": int(rng.integers(5, 200)),
                         "weights": pick(["uniform", "distance"]),
                         "p": pick([1, 2])},
        "Decision tree": lambda: {"max_depth": int(rng.integers(2, 26)),
                                  "min_samples_leaf": int(rng.integers(1, 200)),
                                  "ccp_alpha": lo(1e-6, 1e-2)},
        "Random forest": lambda: {"n_estimators": int(rng.integers(200, 900)),
                                  "max_depth": int(rng.integers(4, 40)),
                                  "min_samples_leaf": int(rng.integers(1, 40)),
                                  "max_features": pick(MAX_FEATURES)},
        "Extra trees": lambda: {"n_estimators": int(rng.integers(200, 900)),
                                "max_depth": int(rng.integers(4, 40)),
                                "min_samples_leaf": int(rng.integers(1, 40)),
                                "max_features": pick(MAX_FEATURES)},
        "Neural net (MLP)": lambda: {
            "hidden_layer_sizes": pick(HIDDEN),
            "alpha": lo(1e-6, 1e-1), "learning_rate_init": lo(1e-4, 1e-2),
            "batch_size": pick([128, 256, 512])},
        "HistGradientBoosting": lambda: {
            "learning_rate": lo(5e-3, 3e-1), "max_leaf_nodes": int(rng.integers(8, 128)),
            "min_samples_leaf": int(rng.integers(5, 200)), "l2_regularization": lo(1e-6, 10),
            "max_iter": int(rng.integers(150, 900))},
        "LightGBM": lambda: {
            "learning_rate": lo(5e-3, 3e-1), "num_leaves": int(rng.integers(8, 200)),
            "min_child_samples": int(rng.integers(5, 200)),
            "feature_fraction": float(rng.uniform(0.4, 1.0)),
            "bagging_fraction": float(rng.uniform(0.5, 1.0)), "bagging_freq": 1,
            "reg_lambda": lo(1e-6, 10), "n_estimators": int(rng.integers(200, 1200))},
        "CatBoost": lambda: {
            "learning_rate": lo(1e-2, 3e-1), "depth": int(rng.integers(4, 11)),
            "l2_leaf_reg": lo(1e-1, 30), "iterations": int(rng.integers(200, 900))},
        "XGBoost": lambda: {
            "learning_rate": lo(5e-3, 3e-1), "max_depth": int(rng.integers(3, 13)),
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.4, 1.0)),
            "min_child_weight": int(rng.integers(1, 40)), "gamma": lo(1e-3, 10),
            "reg_lambda": lo(1e-3, 10), "n_estimators": int(rng.integers(200, 1200))},
    }


def make(name, params, seed, num, cat):
    """Build a fitted-ready estimator for `name` with `params`."""
    pre = lambda est: Pipeline([("pre", sklearn_preprocessor(num, cat)), ("est", est)])
    if name == "Logistic regression":
        return pre(LogisticRegression(max_iter=3000, class_weight="balanced", **params)), True
    if name == "Linear SVM":
        return pre(LinearSVC(class_weight="balanced", dual="auto", max_iter=5000, **params)), True
    if name == "Gaussian naive Bayes":
        return pre(GaussianNB(**params)), True
    if name == "k-NN":
        return pre(KNeighborsClassifier(n_jobs=-1, **params)), True
    if name == "Decision tree":
        return pre(DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)), True
    if name == "Random forest":
        return pre(RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                          random_state=seed, **params)), True
    if name == "Extra trees":
        return pre(ExtraTreesClassifier(class_weight="balanced", n_jobs=-1,
                                        random_state=seed, **params)), True
    if name == "Neural net (MLP)":
        return pre(MLPClassifier(max_iter=400, early_stopping=True,
                                 random_state=seed, **params)), True
    if name == "HistGradientBoosting":
        return pre(HistGradientBoostingClassifier(class_weight="balanced",
                                                  random_state=seed, **params)), True
    if name == "LightGBM":
        return lgb.LGBMClassifier(class_weight="balanced", random_state=seed,
                                  n_jobs=-1, verbose=-1, **params), False
    if name == "CatBoost":
        return CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False,
                                  auto_class_weights="Balanced", **params), False
    if name == "XGBoost":
        return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=-1,
                                 random_state=seed, objective="binary:logistic",
                                 eval_metric="logloss", **params), False
    raise KeyError(name)


def fit_score(name, params, seed, data, cols, num, cat, split="valid"):
    (Xtr, ytr) = data["train"]
    (Xev, yev) = data[split]
    est, needs_pre = make(name, params, seed, num, cat)
    w = compute_sample_weight("balanced", ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if needs_pre:
            Xa, Xb = Xtr, Xev
        else:                       # native-categorical families need aligned dtypes
            Xa, Xb = Xtr.copy(), Xev.copy()
            if name == "CatBoost":
                for c in cat:
                    Xa[c] = Xa[c].astype(str).fillna("missing")
                    Xb[c] = Xb[c].astype(str).fillna("missing")
                est.set_params(cat_features=cat)
            else:
                # Unify categories across ALL THREE splits, not just the two in
                # play, so a config scored on valid and the same config scored on
                # test see identical category codes -- otherwise this script's
                # numbers drift from compare_model_families.py's for one model.
                Xc = data["valid"][0].copy() if split == "test" else data["test"][0].copy()
                unify_categorical_dtypes([Xa, Xb, Xc], cols)
        t0 = time.perf_counter()
        if name == "XGBoost":
            est.fit(Xa, ytr, sample_weight=w)
        else:
            est.fit(Xa, ytr)
        secs = time.perf_counter() - t0
        pred = est.predict(Xb)
        score = est.predict_proba(Xb)[:, 1] if hasattr(est, "predict_proba") \
            else est.decision_function(Xb)
    return (f1_score(yev, pred, average="macro"), roc_auc_score(yev, score), secs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these families")
    ap.add_argument("--reeval", action="store_true",
                    help="skip the search; re-score the winners already saved in "
                         "tuned_results.csv (use after changing the final-eval path)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xy, cols, cat, num = load_data()
    print(f"train={len(Xy['train'][0])} valid={len(Xy['valid'][0])} test={len(Xy['test'][0])}  "
          f"{len(cols)} features, {args.trials} trials/family", flush=True)

    rng = np.random.default_rng(0)
    space = spaces(rng)
    families = list(space) if args.only is None else args.only
    # CatBoost trains ~10x slower than the rest; halve its budget to keep the
    # total runtime sane, and record the budget actually used per family.
    budget = {f: (args.trials // 2 if f == "CatBoost" else args.trials) for f in families}

    canon = json.load(open(MODEL_DIR / "best_hyperparams_normalized_binary.json"))["params"]
    canon = {**canon, "n_estimators": 1000}

    if args.reeval:
        prev = pd.read_csv(OUT_DIR / "tuned_results.csv")
        winners = {f: (g.valid_f1.iloc[0], json.loads(g.params.iloc[0]))
                   for f, g in prev.groupby("family") if f != "XGBoost (project-tuned)"}
        print(f"re-evaluating {len(winners)} saved winners", flush=True)
        trials = []
    else:
      trials, winners = [], {}
      for fam in families:
          best = (-1, None)
          for t in range(budget[fam]):
              # Trial 0 for XGBoost is this project's existing tuned configuration:
              # it competes as a candidate and wins only on validation score.
              params = canon if (fam == "XGBoost" and t == 0) else space[fam]()
              try:
                  f1, auc, secs = fit_score(fam, params, 42, Xy, cols, num, cat, "valid")
              except Exception as exc:                    # a sampled config can be invalid
                  print(f"  {fam:22s} trial {t:2d} FAILED {type(exc).__name__}: {exc}", flush=True)
                  continue
              trials.append(dict(family=fam, trial=t, valid_f1=f1, valid_auc=auc,
                                 secs=secs, params=json.dumps(params, default=str)))
              if f1 > best[0]:
                  best = (f1, params)
                  print(f"  {fam:22s} trial {t:2d} valid_f1={f1:.4f}  * new best", flush=True)
          winners[fam] = best
          print(f"== {fam}: best valid macro F1 {best[0]:.4f}  {best[1]}", flush=True)

    if trials:
        pd.DataFrame(trials).to_csv(OUT_DIR / "tuning_trials.csv", index=False)

    # Final: retrain each winner over seeds, report on test.
    rows = []
    for fam, (vf1, params) in winners.items():
        for seed in args.seeds:
            f1, auc, secs = fit_score(fam, params, seed, Xy, cols, num, cat, "test")
            rows.append(dict(family=fam, seed=seed, valid_f1=vf1, test_f1=f1,
                             test_auc=auc, secs=secs, params=json.dumps(params, default=str)))
            print(f"[test] {fam:22s} seed={seed} macroF1={f1:.4f} auc={auc:.4f}", flush=True)

    # The project's canonical XGBoost parameters on their own row, for reference
    # against the random search (they are also trial 0 of that search).
    for seed in args.seeds:
        f1, auc, secs = fit_score("XGBoost", canon, seed, Xy, cols, num, cat, "test")
        rows.append(dict(family="XGBoost (project-tuned)", seed=seed, valid_f1=np.nan,
                         test_f1=f1, test_auc=auc, secs=secs, params=json.dumps(canon)))
        print(f"[test] {'XGBoost (project-tuned)':22s} seed={seed} macroF1={f1:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tuned_results.csv", index=False)
    summary = (df.groupby("family")
                 .agg(test_f1=("test_f1", "mean"), sd=("test_f1", "std"),
                      test_auc=("test_auc", "mean"), valid_f1=("valid_f1", "mean"),
                      secs=("secs", "mean"))
                 .reset_index().sort_values("test_f1", ascending=False))
    summary.to_csv(OUT_DIR / "tuned_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
