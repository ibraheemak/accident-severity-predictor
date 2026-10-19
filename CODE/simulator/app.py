#!/usr/bin/env python3
"""The dispatch simulator with a browser interface.

Two tabs. "Take a call" is the loop from dispatch.py -- the agent asks for one
bundle, you answer, it decides what to ask next -- rendered as a page instead
of a terminal. "Replay a real collision" steps through a held-out collision and
scores the agent's call against the recorded severity.

Runs the REAL classifiers and the REAL trained policy. The 1,024 models total
3.5 GB, which is why this is a local server rather than a static site: the
page talks to Python, and Python holds the models.

State lives in a module-level dict keyed by a session id. That is fine for a
tool one person runs on their own machine and wrong for anything shared, which
this is not: it binds to localhost only.

Usage (from the repo root):
    python CODE/simulator/app.py
    python CODE/simulator/app.py --cost-scale 1 --n-train 3000
then open http://127.0.0.1:8000
"""

import argparse
import json
import math
import secrets
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, redirect, request, session, url_for

import sys
_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
sys.path[:0] = [str(_HERE), str(_CODE / "training" / "agent" / "core"),
                str(_CODE / "training" / "train")]
from train_classifier import (BUNDLE_COLS, BUNDLE_NAMES, CATEGORICAL_COLS,
                              bundle_feature_columns, load_split, mask_bundles,
                              prepare_xy, unify_categorical_dtypes)
from ql_solver import QLearner
from seeding import one_step_seed
from bundle_costs import get_costs_dict
from dispatch import (Encoder, LABELS, MODELS, PARAMS, SWEEP,
                      get_agent, readable)

TIER = {b: t for t, bs in {
    "caller": ("vehicles", "casualties", "casualty_demographics"),
    "map": ("road_class", "road_environment", "junction", "crash_history"),
    "api": ("aadf", "osm", "meteo")}.items() for b in bs}
TIERNAME = {"caller": "Ask the caller", "map": "Map lookup",
            "api": "External API"}
# STATS19 codes the week from Sunday; taken from the labels/ copy of the
# download rather than assumed, since a Sunday/Monday slip would be invisible
DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
        5: "Thursday", 6: "Friday", 7: "Saturday"}

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
RUNS = {}                      # session id -> {row, mask, spent, log}
REPLAY = {}                    # session id -> {cid, k} for the replay tab
ENV = {}                       # agent, encoder, categories, model cache


# ------------------------------------------------------------------ engine --
def predict(row, mask):
    cols = bundle_feature_columns(mask_bundles(mask), "normalized")
    cache = ENV["cache"]
    if mask not in cache:
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS / f"{mask:03d}.json"))
        cache[mask] = m
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    for c in cols:
        if c in CATEGORICAL_COLS:
            df[c] = pd.Categorical(df[c], categories=ENV["cats"][c])
    return float(cache[mask].predict_proba(df)[0][1])


def state():
    """The run for this browser session, created on first use."""
    sid = session.get("sid")
    if not sid or sid not in RUNS:
        sid = secrets.token_hex(8)
        session["sid"] = sid
        RUNS[sid] = None
    return RUNS, sid


def current_lam():
    """The chosen time budget: the live run's, else the session's, else the
    value the server started with."""
    runs, sid = state()
    run = runs.get(sid)
    if run:
        return run["lam"]
    lam = session.get("lam", ENV["lam"])
    return lam if lam in ENV["agents"] else ENV["lam"]


def next_action(run):
    """What the policy says to do from here: a bundle index, or None to stop.

    The agent is looked up by the run's own lambda, so two browser tabs can be
    partway through calls at different cost settings without interfering.
    """
    a = ENV["agents"][run["lam"]].act(run["mask"], max(run["p"], 1 - run["p"]))
    return None if a == 0 else a - 1


# -------------------------------------------------------------------- view --
CSS = """
:root{--ground:#f2f0ec;--panel:#fbfaf8;--sunk:#e7e4de;--line:#d8d4cc;
--line-soft:#e6e2db;--ink:#1a1c20;--ink-2:#55585e;--ink-3:#86898f;
--caller:#c2572b;--map:#1f7a68;--api:#5566b8;--caller-wash:#f6e6dd;
--map-wash:#dcece8;--api-wash:#e2e5f4;--severe:#b0342c;--slight:#36699e;
--clock:#a8761a;--anchor:#7a5bb5}
@media(prefers-color-scheme:dark){:root{--ground:#16181c;--panel:#1e2127;
--sunk:#111317;--line:#2f343c;--line-soft:#262a31;--ink:#e9e7e2;--ink-2:#a3a7ae;
--ink-3:#747981;--caller:#e9834f;--map:#4bb49c;--api:#8e9ce0;
--caller-wash:#2e2019;--map-wash:#152a26;--api-wash:#1d2133;--severe:#e4645a;
--slight:#6ba3d8;--clock:#d9a63f;--anchor:#9b7fd4}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55
"Public Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:34px 22px 72px}
h1{font-family:"Barlow Condensed",system-ui,sans-serif;font-weight:700;
font-size:clamp(30px,5vw,46px);line-height:.98;margin:0;text-transform:uppercase}
.masthead{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:20px;
display:flex;justify-content:space-between;align-items:flex-end;gap:16px}
.lampick{display:flex;flex-direction:column;gap:4px;align-items:flex-end;
min-width:230px}
.lam{font-family:"Barlow Condensed",sans-serif;font-weight:600;
text-transform:uppercase;letter-spacing:.09em;font-size:12px;color:var(--ink-3)}
.lampick select{font-size:12.5px;padding:5px 8px}
.tabs{display:flex;gap:2px;margin:-8px 0 18px}
.tabs a{font-family:"Barlow Condensed",sans-serif;font-weight:600;font-size:14px;
text-transform:uppercase;letter-spacing:.07em;text-decoration:none;
color:var(--ink-3);padding:7px 15px;border:1px solid var(--line);
border-radius:2px;background:var(--panel)}
.tabs a.on{background:var(--ink);color:var(--ground);border-color:var(--ink)}
/* Each play tick is a real page load, so the controls would scroll away and
   the view would jump to the top. Sticky keeps them reachable while playing. */
.controls{position:sticky;top:0;z-index:5;display:flex;gap:8px;
margin-bottom:12px;padding:10px 0;background:var(--ground);
border-bottom:1px solid var(--line-soft)}
.stepcard{border-left:3px solid var(--line);padding:11px 0 11px 14px;
margin-bottom:9px}
.stepcard.caller{border-left-color:var(--caller)}
.stepcard.map{border-left-color:var(--map)}
.stepcard.api{border-left-color:var(--api)}
.sfields{display:grid;gap:3px;margin-top:7px;
font-family:"IBM Plex Mono",monospace;font-size:11.5px}
.sfields div{display:flex;justify-content:space-between;gap:12px}
.sfields span:first-child{color:var(--ink-3)}
.sfields span:last-child{text-align:right}
.truth{display:grid;gap:6px;font-size:13.5px;margin-top:4px}
.truth div{display:flex;justify-content:space-between;gap:10px}
.truth b{font-family:"IBM Plex Mono",monospace}
.tag{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;
letter-spacing:.07em;font-size:12.5px;padding:1px 9px;border:1px solid
currentColor;border-radius:2px}
.tag.ok{color:var(--map)}.tag.no{color:var(--severe)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:3px;
padding:17px 19px;margin-bottom:12px}
.lbl{font-family:"Barlow Condensed",sans-serif;font-weight:600;
text-transform:uppercase;letter-spacing:.09em;font-size:12.5px;color:var(--ink-3)}
.bar{position:relative;height:11px;background:var(--sunk);border:1px solid
var(--line);border-radius:2px;overflow:hidden;margin:4px 0 8px}
.fill{height:100%;background:var(--slight)}.fill.hi{background:var(--severe)}
.mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--ink-3)}
.pnum{font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:600;
font-variant-numeric:tabular-nums}
.meta{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-3);
font-variant-numeric:tabular-nums}
.ask{border-left:3px solid var(--line);padding-left:14px}
.ask.caller{border-left-color:var(--caller)}.ask.map{border-left-color:var(--map)}
.ask.api{border-left-color:var(--api)}
.askname{font-family:"Barlow Condensed",sans-serif;font-weight:600;font-size:23px}
.asktier{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;
letter-spacing:.08em;font-size:11px}
.ask.caller .asktier{color:var(--caller)}.ask.map .asktier{color:var(--map)}
.ask.api .asktier{color:var(--api)}
.f{display:grid;gap:5px;margin:13px 0}
.f label{font-size:13.5px;color:var(--ink-2)}
input,select{font-family:"IBM Plex Mono",monospace;font-size:13px;width:100%;
background:var(--ground);color:var(--ink);border:1px solid var(--line);
border-radius:2px;padding:7px 9px}
.btn{font-family:"Barlow Condensed",sans-serif;font-weight:600;font-size:16px;
text-transform:uppercase;letter-spacing:.06em;background:var(--ink);
color:var(--ground);border:1px solid var(--ink);border-radius:2px;
padding:9px 22px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--ink)}
input:focus-visible,select:focus-visible,.btn:focus-visible{outline:2px solid
var(--clock);outline-offset:2px}
ol{margin:0;padding:0;list-style:none;display:grid;gap:4px;
font-family:"IBM Plex Mono",monospace;font-size:12px}
ol li{display:flex;justify-content:space-between;gap:10px}
.up{color:var(--severe)}.down{color:var(--slight)}
.verdict{font-family:"Barlow Condensed",sans-serif;font-weight:700;
font-size:40px;text-transform:uppercase;line-height:1}
.verdict.sev{color:var(--severe)}.verdict.sli{color:var(--slight)}
"""

HEAD = ('<meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        'initial-scale=1"><title>Dispatch Simulator</title>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600'
        '&family=Public+Sans:wght@400;500;600&display=swap"><style>' + CSS + '</style>')


def page(body, lam, here='call'):
    """lambda sits in the masthead, not in the call form: it is a standing
    decision about how much time the service will spend, not something the
    caller supplies. Changing it takes effect on the agent's next decision,
    mid-call included."""
    opts = "".join(
        f"<option value='{l:g}'{' selected' if l == lam else ''}>"
        f"&lambda; = {l:g}{' — ' + d if d else ''}</option>"
        for l, d in ENV["lamdesc"])
    picker = (f"<form method='post' action='/lam' class='lampick'>"
              f"<label class='lam' for='lam'>Time budget</label>"
              f"<select id='lam' name='lam' onchange='this.form.submit()'>"
              f"{opts}</select></form>")
    return (f"<!doctype html><html><head>{HEAD}</head><body><div class='wrap'>"
            f"<div class='masthead'><h1>Dispatch Simulator</h1>{picker}</div>"
            f"<nav class='tabs'>"
            f"<a href='/' class='{'on' if here == 'call' else ''}'>Take a call</a>"
            f"<a href='/replay' class='{'on' if here == 'replay' else ''}'>"
            f"Replay a real collision</a></nav>"
            f"{body}</div></body></html>")


def meter(run):
    p = run["p"]
    return (f"<div class='panel'><div class='lbl'>Call received</div>"
            f"<div class='pnum' style='font-size:21px'>{run.get('when','—')}</div>"
            f"<div class='meta'>the only thing known for free</div></div>"
            f"<div class='panel'><div class='bar'>"
            f"<div class='fill{' hi' if p>=.5 else ''}' style='width:{p*100:.1f}%'>"
            f"</div><div class='mid'></div></div>"
            f"<div class='pnum'>{p*100:.1f}%</div>"
            f"<div class='meta'>probability severe &middot; "
            f"{run['spent']:.2f}s spent &middot; "
            f"{len(run['log'])} of 10 bundles</div></div>")


def logbox(run):
    if not run["log"]:
        return ""
    rows = "".join(
        f"<li><span>{i+1}. {b.replace('_',' ')}</span>"
        f"<span class='{'up' if d>=0 else 'down'}'>"
        f"{'+' if d>=0 else '−'}{abs(d):.1f}pp</span></li>"
        for i, (b, d) in enumerate(run["log"]))
    return f"<div class='panel'><div class='lbl'>Asked so far</div><ol>{rows}</ol></div>"


# ------------------------------------------------------------------ routes --
@app.get("/")
def home():
    runs, sid = state()
    if runs[sid] is None:
        return page(
            "<div class='panel'><div class='lbl'>The call comes in</div>"
            "<form method='post' action='/start'>"
            "<div class='f'><label for='d'>Day of week</label>"
            "<select id='d' name='dow'>" + "".join(
                f"<option value='{i}'{' selected' if i==4 else ''}>{n}</option>"
                for i, n in enumerate(["Sunday","Monday","Tuesday","Wednesday",
                                       "Thursday","Friday","Saturday"], start=1))
            + "</select></div>"
            "<div class='f'><label for='h'>Hour of day (0–23)</label>"
            "<input id='h' name='hour' type='number' min='0' max='23' value='17'>"
            "</div><button class='btn'>Take the call</button></form></div>",
            current_lam(), "call")
    return step(runs[sid])


@app.post("/start")
def start():
    runs, sid = state()
    dow = int(request.form.get("dow", 4))
    hour = int(request.form.get("hour", 12))
    lam = current_lam()
    row = dict(ENV["enc"].clock(dow, hour))
    runs[sid] = {"row": row, "mask": 0, "spent": 0.0, "log": [], "lam": lam,
                 "when": f"{DAYS.get(dow, '?')}, {hour:02d}:00",
                 "p": predict(row, 0)}
    return redirect(url_for("home"))


def step(run):
    """Either the next question, or the verdict once the policy stops."""
    b = next_action(run)
    if b is None:
        p, sev = run["p"], run["p"] >= .5
        conf = p if sev else 1 - p
        names = ", ".join(x.replace("_", " ") for x, _ in run["log"]) or "nothing"
        return page(
            meter(run)
            + "<div class='panel'><div class='lbl'>The agent stops here</div>"
            f"<div class='verdict {'sev' if sev else 'sli'}'>"
            f"{'Severe' if sev else 'Slight'}</div>"
            f"<div class='meta' style='margin-top:7px'>{conf*100:.1f}% confidence "
            f"&middot; {run['spent']:.2f}s of 6.26s &middot; asked for {names}"
            "</div></div>" + logbox(run)
            + "<form method='post' action='/reset'>"
            "<button class='btn ghost'>New call</button></form>",
            run["lam"], "call")

    name = BUNDLE_NAMES[b]
    tier = TIER[name]
    fields = ""
    for c in BUNDLE_COLS[name]:
        if c in LABELS:
            opts = "".join(f"<option value='{k}'>{k} — {v}</option>"
                           for k, v in LABELS[c].items())
            inp = (f"<select name='{c}'><option value=''>unknown</option>"
                   f"{opts}</select>")
        else:
            inp = f"<input name='{c}' type='number' step='any' placeholder='unknown'>"
        fields += (f"<div class='f'><label for='{c}'>{c.replace('_',' ')}</label>"
                   f"{inp}</div>")
    return page(
        meter(run)
        + f"<div class='panel'><div class='ask {tier}'>"
        f"<div class='asktier'>{TIERNAME[tier]} &middot; "
        f"{ENV['secs'][b]:.2f}s</div>"
        f"<div class='askname'>{name.replace('_',' ')}</div></div>"
        f"<form method='post' action='/answer'>"
        f"<input type='hidden' name='bundle' value='{b}'>{fields}"
        "<button class='btn'>Answer</button></form></div>" + logbox(run),
        run["lam"], "call")


@app.post("/answer")
def answer():
    runs, sid = state()
    run = runs[sid]
    if run is None:
        return redirect(url_for("home"))
    b = int(request.form["bundle"])
    name = BUNDLE_NAMES[b]
    for c in BUNDLE_COLS[name]:
        v = request.form.get(c, "").strip()
        if v != "":
            run["row"][c] = ENV["enc"].num(c, float(v))
    before = run["p"]
    run["mask"] |= 1 << b
    run["spent"] += ENV["secs"][b]
    run["p"] = predict(run["row"], run["mask"])
    run["log"].append((name, (run["p"] - before) * 100))
    return redirect(url_for("home"))


def trajectory(cid, lam):
    """Every step the agent takes on this collision, computed once and kept.

    Deterministic given (collision, lambda) -- the policy is a lookup and the
    classifiers are fixed -- so the whole run is built on first request and the
    step buttons just reveal more of it. Recomputing per click would reload
    classifiers for no reason.
    """
    key = (cid, lam)
    if key in ENV["traj"]:
        return ENV["traj"][key]

    i = ENV["index"][cid]
    row = {c: ENV["Xte"].iloc[i][c] for c in ENV["Xte"].columns}
    raw = ENV["raw"].iloc[i]
    free = {k: row[k] for k in bundle_feature_columns([], "normalized")}

    mask, spent, steps, held = 0, 0.0, [], {}
    p = predict(free, 0)
    start = p
    while len(steps) < 10:
        a = ENV["agents"][lam].act(mask, max(p, 1 - p))
        if a == 0:
            break
        b = a - 1
        name = BUNDLE_NAMES[b]
        mask |= 1 << b
        spent += ENV["secs"][b]
        for c in BUNDLE_COLS[name]:
            held[c] = row.get(c)
        before = p
        p = predict(held | free, mask)
        steps.append(dict(name=name, cost=ENV["secs"][b], p=p,
                          d=(p - before) * 100, spent=spent,
                          fields={c: readable(c, raw[c])
                                  for c in BUNDLE_COLS[name] if c in raw.index}))
    out = dict(steps=steps, start=start, final=p, spent=spent,
               truth=int(ENV["yte"].iloc[i]))
    ENV["traj"][key] = out
    return out


@app.get("/replay")
def replay():
    """Pick a held-out collision and step the agent through it.

    Nothing is precomputed as it is on the published page: the policy and the
    classifiers are both in memory, so this runs on request and the ground
    truth comes from the same row.
    """
    lam = current_lam()
    runs, sid = state()
    rp = REPLAY.get(sid) or {}
    chosen = request.args.get("c", rp.get("cid", ""))
    if "c" in request.args:                       # a new pick restarts the walk
        REPLAY[sid] = {"cid": chosen, "k": 0, "playing": False}
        rp = REPLAY[sid]

    opts = "".join(f"<option value='{c}'{' selected' if c == chosen else ''}>"
                   f"{c}</option>" for c in ENV["sample"])
    form = ("<div class='panel'><div class='lbl'>Held-out collision</div>"
            "<form method='get' action='/replay'>"
            f"<div class='f'><select name='c' onchange='this.form.submit()'>"
            f"<option value=''>choose one…</option>{opts}</select></div>"
            "</form></div>")
    if not chosen or chosen not in ENV["index"]:
        return page(form, lam, "replay")

    t = trajectory(chosen, lam)
    when = ENV["raw"].iloc[ENV["index"][chosen]]
    stamp = (f"{DAYS.get(int(when['day_of_week']), '?')}, "
             f"{str(when['time'])[:5]}")
    k = min(rp.get("k", 0), len(t["steps"]))
    done = k == len(t["steps"])
    p = t["steps"][k - 1]["p"] if k else t["start"]
    spent = t["steps"][k - 1]["spent"] if k else 0.0

    body = form + (
        f"<div class='panel'><div class='lbl'>Call received</div>"
        f"<div class='pnum' style='font-size:21px'>{stamp}</div>"
        f"<div class='meta'>the only thing known for free &middot; "
        f"collision {chosen}</div></div>"
        f"<div class='panel'><div class='bar'>"
        f"<div class='fill{' hi' if p>=.5 else ''}' style='width:{p*100:.1f}%'>"
        f"</div><div class='mid'></div></div>"
        f"<div class='pnum'>{p*100:.1f}%</div>"
        f"<div class='meta'>probability severe &middot; {spent:.2f}s spent "
        f"&middot; {k} of {len(t['steps'])} steps"
        + (" &middot; from the timestamp alone" if k == 0 else "")
        + "</div></div>")

    # Controls. Play is the same one-step-at-a-time walk, driven by a timer
    # instead of a click: the page re-posts to itself while the flag is set, so
    # each step is a real render rather than a client-side animation of data
    # fetched up front.
    playing = rp.get("playing") and not done
    body += ("<form method='post' action='/replay/step' class='controls'>"
             + ("" if done else
                "<button class='btn' id='autonext' name='go' value='next'>"
                "Next step</button>"
                + (f"<button class='btn ghost' name='go' value='pause'>Pause"
                   f"</button>" if playing else
                   f"<button class='btn ghost' name='go' value='play'>Play"
                   f"</button>"))
             + "<button class='btn ghost' name='go' value='reset'>Restart</button>"
             "</form>")
    if playing:
        body += ("<script>setTimeout(function(){"
                 "document.getElementById('autonext').click();},900);</script>")

    for n, st in enumerate(t["steps"][:k], 1):
        rows = "".join(f"<div><span>{c.replace('_',' ')}</span>"
                       f"<span>{v if v is not None else '—'}</span></div>"
                       for c, v in st["fields"].items())
        body += (f"<div class='panel'><div class='stepcard {TIER[st['name']]}'>"
                 f"<div class='asktier'>{n}. {TIERNAME[TIER[st['name']]]} "
                 f"&middot; {st['cost']:.2f}s</div>"
                 f"<div class='askname'>{st['name'].replace('_',' ')}</div>"
                 f"<div class='sfields'>{rows}</div></div>"
                 f"<div class='meta' style='margin-top:9px'>"
                 f"p(severe) {st['p']*100:.1f}%  "
                 f"<span class='{'up' if st['d']>=0 else 'down'}'>"
                 f"{'+' if st['d']>=0 else '−'}{abs(st['d']):.1f}pp</span>"
                 f"</div></div>")

    if done:
        if not t["steps"]:
            body += ("<div class='panel'><div class='meta'>the agent asked for "
                     "nothing and predicted from the timestamp alone</div></div>")
        sev = t["final"] >= .5
        called = "Severe" if sev else "Slight"
        actual = "Severe" if t["truth"] == 1 else "Slight"
        ok = called == actual
        body += (f"<div class='panel'><div class='lbl'>Outcome</div>"
                 f"<div class='verdict {'sev' if sev else 'sli'}'>{called}</div>"
                 f"<div class='truth' style='margin-top:11px'>"
                 f"<div><span>Actual severity</span><b>{actual}</b></div>"
                 f"<div><span>Result</span>"
                 f"<span class='tag {'ok' if ok else 'no'}'>"
                 f"{'correct' if ok else 'wrong'}</span></div>"
                 f"<div><span>Asked for</span><b>{len(t['steps'])} of 10</b></div>"
                 f"<div><span>Spent</span><b>{t['spent']:.2f}s</b></div>"
                 f"</div></div>")

    if k:
        # Must come AFTER the step cards: an inline script runs during parsing,
        # so placed any earlier it queries a document that has none of them yet.
        # 'auto' rather than 'smooth' -- the page is new, and an animated scroll
        # on load reads as a glitch.
        body += ("<script>var c=document.querySelectorAll('.stepcard');"
                 "if(c.length)c[c.length-1].scrollIntoView("
                 "{block:'center',behavior:'auto'});</script>")
    return page(body, lam, "replay")


@app.post("/replay/step")
def replay_step():
    runs, sid = state()
    rp = REPLAY.get(sid)
    if not rp:
        return redirect(url_for("replay"))
    go = request.form.get("go")
    t = trajectory(rp["cid"], current_lam())
    if go == "next":
        rp["k"] = min(rp["k"] + 1, len(t["steps"]))
    elif go == "play":
        rp["playing"] = True
        rp["k"] = min(rp["k"] + 1, len(t["steps"]))
    elif go == "pause":
        rp["playing"] = False
    else:
        rp["k"], rp["playing"] = 0, False
    if rp["k"] >= len(t["steps"]):
        rp["playing"] = False          # nothing left to advance to
    return redirect(url_for("replay"))


@app.post("/lam")
def setlam():
    runs, sid = state()
    try:
        lam = float(request.form["lam"])
    except (KeyError, ValueError):
        return redirect(url_for("home"))
    if lam in ENV["agents"]:
        session["lam"] = lam
        if runs.get(sid):
            runs[sid]["lam"] = lam       # applies from the next decision on
    return redirect(url_for("home"))


@app.post("/reset")
def reset():
    runs, sid = state()
    runs[sid] = None
    return redirect(url_for("home"))


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-scale", type=float, default=0.01, metavar="LAMBDA")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    # Not 5000: macOS Control Centre (AirPlay Receiver) listens on *:5000 and
    # answers anything that is not AirPlay with HTTP 403, which looks exactly
    # like the app refusing to serve.
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cfg = json.load(open(PARAMS))["params"]
    secs = get_costs_dict()
    costs = {i: s * args.cost_scale for i, s in secs.items()}
    cols = bundle_feature_columns(BUNDLE_NAMES, "normalized")
    Xtr, ytr = prepare_xy(load_split("train", cols, variant="normalized"),
                          cols, binary=True)
    unify_categorical_dtypes([Xtr], cols)

    # One agent per shipped lambda, so the setting is a choice in the page
    # rather than a flag fixed when the server started. They are small (~1 MB)
    # and already trained, so loading all of them costs a second.
    from dispatch import AGENTS
    lams = sorted(float(f.stem[3:]) for f in AGENTS.glob("lam*.json")
                  if f.stem[3:].replace(".", "", 1).isdigit())
    if args.cost_scale not in lams:
        lams.append(args.cost_scale)
    agents = {l: get_agent(l, args.n_train, args.seed, Xtr, ytr, quiet=True)
              for l in sorted(lams)}
    # the held-out split, for the replay tab: normalised for the models, raw
    # for what gets shown, and the truth column to score against
    norm = load_split("test", cols + ["collision_index"], variant="normalized")
    Xte, yte = prepare_xy(norm, cols, binary=True)
    unify_categorical_dtypes([Xtr, Xte], cols)
    raw = pd.read_csv("data/splits/test.csv")
    assert (raw["collision_index"].values == norm["collision_index"].values).all(), \
        "raw and normalized test splits are not in the same row order"
    ids = list(norm["collision_index"].astype(str))
    rng = np.random.default_rng(args.seed)
    sample = sorted(str(ids[i]) for i in
                    rng.choice(len(ids), size=min(150, len(ids)), replace=False))

    print(f"policies ready for lambda: "
          f"{', '.join(f'{l:g}' for l in sorted(lams))}", flush=True)

    DESC = {0: "time is free, asks for almost everything",
            0.01: "a second costs little", 0.1: "a second costs real accuracy",
            1: "seconds are expensive, commits early",
            10: "barely asks anything at all"}
    ENV.update(agents=agents, enc=Encoder(), secs=secs, lam=args.cost_scale,
               lamdesc=[(l, DESC.get(l, "")) for l in sorted(lams)],
               Xte=Xte, yte=yte, raw=raw, sample=sample,
               index={c: i for i, c in enumerate(ids)}, traj={},
               cats={c: list(Xtr[c].cat.categories)
                     for c in CATEGORICAL_COLS if c in Xtr},
               cache={})
    print(f"\n  open  http://127.0.0.1:{args.port}\n", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
