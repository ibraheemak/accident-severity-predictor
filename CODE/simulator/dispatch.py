#!/usr/bin/env python3
"""Answer the agent's questions as they come in, and watch it decide.

You are the call handler. The agent starts with nothing but the time of the
call, asks for one bundle at a time, and after each answer decides whether it
has heard enough or wants something more. You type what you know; it decides
what to ask next.

This runs the REAL classifiers and the REAL trained policy -- the same 1,024
cached models and the same Q-table the results in the report come from. Nothing
is approximated, which is why this is a script and not a web page: the models
are 3.5 GB and cannot go in a browser.

Typed answers are put through exactly the transforms the training data went
through -- cyclic encoding for the timestamp, log1p where the pipeline applied
it, then the training split's own mean and standard deviation, all read from
data/splits/normalized/normalization_stats.csv. A value entered here is
therefore treated identically to the same value in the training data.

Usage (from the repo root):
    python CODE/simulator/dispatch.py                 # lambda 0.01
    python CODE/simulator/dispatch.py --cost-scale 1  # time is expensive
    python CODE/simulator/dispatch.py --example       # fill from a real collision
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
_CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_CODE / "training" / "agent" / "core"),
                str(_CODE / "training" / "train")]
from train_classifier import (BUNDLE_COLS, BUNDLE_NAMES, CATEGORICAL_COLS,
                              bundle_feature_columns, load_split, mask_bundles,
                              prepare_xy, unify_categorical_dtypes)
from ql_solver import QLearner
from seeding import one_step_seed
from bundle_costs import get_costs_dict

SWEEP = Path("docs/figures/bundle_sweep/bundle_sweep_normalized_binary.csv")
PARAMS = Path("models/rl_agents/optuna/best_params.json")
NORM = Path("data/splits/normalized/normalization_stats.csv")
MODELS = Path("models/subsets/normalized_binary")
AGENTS = Path(__file__).resolve().parent / "agents"
LABELS = json.loads((Path(__file__).resolve().parent / "code_labels.json").read_text())

DIM, B, R, G, Y, O = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


# ---------------------------------------------------------------- encoding --
class Encoder:
    """Turns typed values into model features, the same way the pipeline did."""

    def __init__(self):
        s = pd.read_csv(NORM)
        self.mean = dict(zip(s.feature, s["mean"]))
        self.sd = dict(zip(s.feature, s.sd))
        self.log1p = set(s.loc[s.log1p_first, "feature"])

    def num(self, col, v):
        if col not in self.mean:                 # not a z-scored column
            return float(v)
        x = math.log1p(float(v)) if col in self.log1p else float(v)
        return (x - self.mean[col]) / self.sd[col]

    @staticmethod
    def clock(day_of_week, hour):
        """day_of_week 1-7 (1 = Sunday, as STATS19 codes it), hour 0-23."""
        a, b = 2 * math.pi * (day_of_week - 1) / 7, 2 * math.pi * hour / 24
        return {"day_of_week_sin": math.sin(a), "day_of_week_cos": math.cos(a),
                "hour_sin": math.sin(b), "hour_cos": math.cos(b)}


def get_agent(lam, n_train, seed, Xtr, ytr, quiet=False):
    """The trained policy for this lambda, trained once and kept.

    Training is deterministic given (lambda, seed, n_train), so a run with the
    same three is the same table every time -- there is nothing to gain by
    redoing it. The cache key carries all three, so changing any of them
    trains a new one rather than silently reusing the wrong policy.
    """
    cfg = json.load(open(PARAMS))["params"]
    secs = get_costs_dict()
    costs = {i: s * lam for i, s in secs.items()}
    agent = QLearner(lr=cfg["lr"], gamma=1.0, epsilon=cfg["epsilon"],
                     step_weight=cfg["step_weight"], terminal_weight=1.0,
                     bundle_costs=costs, n_conf_bins=cfg["conf_bins"], seed=seed)

    shipped = AGENTS / f"lam{lam:g}.json"
    if shipped.exists():
        agent.load(shipped)
        agent.bundle_costs = costs      # the caller's lambda, not the saved one
        if not quiet:
            print(f"{DIM}loaded {shipped.relative_to(Path.cwd())}{O}")
        return agent

    # An agent already in the repo is preferred to training a new one -- but
    # only if its hyperparameters match the current config AND its saved costs
    # imply this lambda. models/rl_agents/final/ and final_lam0.1/ predate the
    # move to 5 confidence bins, so loading them would run a policy that is not
    # the one the reported results describe.
    want = (round(cfg["lr"], 6), round(cfg["epsilon"], 6),
            cfg["conf_bins"], round(cfg["step_weight"], 6))
    for cand in sorted(Path("models/rl_agents").rglob("*.json")):
        if cand.name == "best_params.json" or AGENTS in cand.parents:
            continue
        try:
            d = json.loads(cand.read_text())
            hp, bc = d["hyperparams"], d["bundle_costs"]
        except Exception:
            continue
        got = (round(hp.get("lr", hp.get("alpha", -1)), 6),
               round(hp.get("epsilon", -1), 6), hp.get("n_conf_bins"),
               round(hp.get("step_weight", -1), 6))
        # bundle 8 is osm at 0.010 s, so its stored cost is lambda * 0.010
        implied = float(bc.get("8", 0)) / secs[8] if secs[8] else 0.0
        if got == want and abs(implied - lam) < 1e-9:
            agent.load(cand)
            agent.bundle_costs = costs
            if not quiet:
                print(f"{DIM}using the agent already in the repo: {cand}{O}")
            return agent

    path = AGENTS / f"lam{lam:g}_n{n_train}_s{seed}.json"   # unlisted lambda
    if path.exists():
        agent.load(path)
        # load() restores the saved costs; the caller's lambda must still win
        agent.bundle_costs = costs
        if not quiet:
            print(f"{DIM}loaded {path}{O}")
        return agent

    if not quiet:
        print(f"{DIM}training the agent at lambda={lam:g} "
              f"({n_train:,} episodes); it is cached after this...{O}", flush=True)
    valid = pd.read_csv(SWEEP).set_index("mask")["valid_accuracy"].to_dict()
    agent.warm_start(one_step_seed(valid, costs, n_conf_bins=cfg["conf_bins"]))
    agent.train(Xtr, ytr, n_episodes=n_train, show_progress=False)
    agent.save(path)
    if not quiet:
        print(f"{DIM}saved {path}{O}")
    return agent


# Open-Meteo WMO weather codes. Not a STATS19 field, so no labels file covers it.
WMO = {0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
       45:"Fog", 48:"Depositing rime fog", 51:"Light drizzle",
       53:"Moderate drizzle", 55:"Dense drizzle", 61:"Slight rain",
       63:"Moderate rain", 65:"Heavy rain", 71:"Slight snowfall",
       73:"Moderate snowfall", 75:"Heavy snowfall", 80:"Rain showers"}

UNITS = {"speed_limit":"{:g} mph", "aadf_distance_m":"{:,.0f} m",
         "dist_to_nearest_hist_acc_m":"{:,.0f} m", "aadf_traffic_volume":"{:,.0f} veh/day",
         "temperature_2m_c":"{:.1f} °C", "precipitation_mm":"{:.1f} mm",
         "wind_speed_10m_kmh":"{:.0f} km/h", "snowfall_cm":"{:.1f} cm",
         "snow_depth_m":"{:.2f} m", "youngest_casualty_age":"{:g} yrs",
         "oldest_casualty_age":"{:g} yrs", "lane_count":"{:g}"}


def readable(field, v):
    """A value a person can read: labels for coded fields, Yes/No for flags,
    units where they exist. Falls back to the raw value rather than inventing
    a label, so an unmapped code shows as itself instead of silently wrong."""
    if pd.isna(v):
        return None
    if field in LABELS:
        key = str(int(v)) if float(v).is_integer() else str(v)
        return LABELS[field].get(key, f"code {key}")
    if field.startswith("has_"):
        return "Yes" if float(v) else "No"
    if field == "weather_code":
        return WMO.get(int(v), f"code {int(v)}")
    if field in UNITS:
        return UNITS[field].format(float(v))
    if field in ("first_road_number", "second_road_number"):
        return "none" if float(v) == 0 else f"{int(v)}"
    if isinstance(v, (np.integer,)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v):,}"
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.4g}"
    return str(v)


# ------------------------------------------------------------------ asking --
def ask(col, cats):
    """One field. Coded fields list their options; blank means not known."""
    if col in LABELS:
        opts = LABELS[col]
        print(f"\n  {B}{col.replace('_',' ')}{O}")
        for k, v in opts.items():
            print(f"    {DIM}{k:>3}{O}  {v}")
        while True:
            r = input(f"    code (blank = unknown): ").strip()
            if r == "":
                return None
            if r in opts:
                return float(r)
            print(f"    {R}not one of the codes above{O}")
    while True:
        r = input(f"  {col.replace('_',' ')}: ").strip()
        if r == "":
            return None
        try:
            return float(r)
        except ValueError:
            print(f"  {R}needs to be a number{O}")


def predict(row, mask, cats, cache):
    """p(severe) from the cached classifier for exactly this acquisition set."""
    cols = bundle_feature_columns(mask_bundles(mask), "normalized")
    if mask not in cache:
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS / f"{mask:03d}.json"))
        cache[mask] = m
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    for c in cols:
        if c in CATEGORICAL_COLS:
            df[c] = pd.Categorical(df[c], categories=cats[c])
    return float(cache[mask].predict_proba(df)[0][1])


def bar(p, width=34):
    n = int(round(p * width))
    col = R if p >= .5 else G
    return f"{col}{'#' * n}{O}{DIM}{'.' * (width - n)}{O}"


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-scale", type=float, default=0.01, metavar="LAMBDA",
                    help="accuracy points charged per second. Low means the "
                         "agent asks a lot; high means it commits early.")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--example", action="store_true",
                    help="prefill answers from a real test collision instead "
                         "of typing them, to see the flow quickly")
    args = ap.parse_args()

    enc = Encoder()
    cfg = json.load(open(PARAMS))["params"]
    secs = get_costs_dict()
    costs = {i: s * args.cost_scale for i, s in secs.items()}

    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    Xtr, ytr = prepare_xy(load_split("train", cols, variant="normalized"),
                          cols, binary=True)
    unify_categorical_dtypes([Xtr], cols)
    cats = {c: list(Xtr[c].cat.categories) for c in CATEGORICAL_COLS if c in Xtr}

    agent = get_agent(args.cost_scale, args.n_train, args.seed, Xtr, ytr)

    example = None
    if args.example:
        raw = pd.read_csv("data/splits/test.csv")
        example = raw.sample(1, random_state=args.seed).iloc[0]
        print(f"{DIM}answers prefilled from collision "
              f"{example['collision_index']}{O}")

    print(f"\n{B}=== 999 call ==={O}")
    if example is not None:
        dow, hr = int(example.get("day_of_week", 4)), 14
        print(f"  day of week {dow}, hour {hr}")
    else:
        dow = int(input("  day of week (1=Sun .. 7=Sat): ") or 4)
        hr = int(input("  hour of day (0-23): ") or 12)

    row, mask, spent, held = dict(enc.clock(dow, hr)), 0, 0.0, []
    cache = {}
    p = predict(row, mask, cats, cache)
    print(f"\n  {bar(p)}  {B}{p*100:5.1f}%{O} severe   {DIM}(timestamp only){O}")

    while True:
        a = agent.act(mask, max(p, 1 - p))
        if a == 0:
            break
        b = BUNDLE_NAMES[a - 1]
        print(f"\n{B}>> the agent asks for: {b.replace('_',' ')}{O}"
              f"   {DIM}({secs[a-1]:.2f}s){O}")
        for c in BUNDLE_COLS[b]:
            v = (enc.num(c, example[c]) if example is not None
                 and c in example and not pd.isna(example[c])
                 else None) if example is not None else ask(c, cats)
            if example is None and v is not None:
                v = enc.num(c, v)
            if v is not None:
                row[c] = v
        mask |= 1 << (a - 1)
        spent += secs[a - 1]
        held.append(b)
        before = p
        p = predict(row, mask, cats, cache)
        d = (p - before) * 100
        arrow = f"{R}+{d:.1f}{O}" if d >= 0 else f"{G}{d:.1f}{O}"
        print(f"\n  {bar(p)}  {B}{p*100:5.1f}%{O} severe   "
              f"{arrow} pp   {Y}{spent:.2f}s{O} spent")

    verdict = "SEVERE" if p >= .5 else "SLIGHT"
    print(f"\n{B}=== the agent stops here ==={O}")
    print(f"  calls it     {B}{R if p>=.5 else G}{verdict}{O}  "
          f"at {p*100:.1f}% confidence" if p >= .5 else
          f"  calls it     {B}{G}{verdict}{O}  at {(1-p)*100:.1f}% confidence")
    print(f"  asked for    {len(held)} of 10 bundles"
          + (f" ({', '.join(b.replace('_',' ') for b in held)})" if held else ""))
    print(f"  spent        {spent:.2f}s of a possible {sum(secs.values()):.2f}s")
    if example is not None:
        truth = "SLIGHT" if int(example["collision_severity"]) == 3 else "SEVERE"
        ok = truth == verdict
        print(f"  actually was {B}{truth}{O}  -> "
              f"{G if ok else R}{'correct' if ok else 'wrong'}{O}")


if __name__ == "__main__":
    main()
