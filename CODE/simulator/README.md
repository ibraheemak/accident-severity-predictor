# Dispatch simulator

Inspect the trained agent on individual calls rather than in aggregate. It runs
the **real** classifiers and the **real** saved policies, so every decision shown
is the system itself, not a reconstruction.

```bash
python CODE/simulator/app.py        # browser  -> http://127.0.0.1:8000
python CODE/simulator/dispatch.py   # the same loop in the terminal
```

Run from the repo root — like everything else here, paths resolve against the
working directory, not the script's own location.

## Two modes

**Take a call.** You are the handler. Give the day and hour, and the agent asks
for one bundle at a time; you answer what is known and it decides after each
answer whether to ask for more or commit. Coded fields offer DfT's own wordings,
and blank means unknown — which XGBoost handles natively, the same way genuinely
missing data was handled in training, so "I don't know" is information rather
than a gap.

**Replay a real collision.** Pick one of 150 held-out collisions and step the
agent through it, or press Play to advance automatically. Each step shows the
bundle chosen, what it revealed, and how the predicted probability moved. The
outcome — the call, the recorded severity, correct or wrong — stays hidden until
the walk finishes.

λ sits in the masthead rather than in either form: it is a standing decision
about how much time the service will spend, not something a caller supplies.
Changing it applies from the agent's next decision, mid-call included.

## Files

```
app.py              the simulator (Flask)
dispatch.py         terminal version; also holds the shared encoding and
                    the STATS19 code-to-text formatting
pretrain_agents.py  one-off: regenerates agents/
agents/             five saved policies, one per shipped λ  (5.1 MB)
code_labels.json    STATS19 code -> DfT wording
```

## What it needs that the repo does not carry

`agents/` ships, so the policy travels with the code and nothing trains on
startup. The **1,024 classifiers do not** — `models/subsets/normalized_binary/`
is 3.5 GB and gitignored. A fresh clone can read the policy but cannot predict
with it until `CODE/training/train/bundle_sweep.py` has been run.

That size is also why this is a local server and not a static site: no browser
can hold the models, and no artifact CSP would let a page fetch them.

## Two things worth knowing

**Typed answers go through the pipeline's own transforms** — cyclic encoding for
the timestamp, `log1p` where the pipeline applied it, then the training split's
mean and standard deviation from `data/splits/normalized/normalization_stats.csv`.
Verified to 1e-6 against the normalised split, so a value entered here is treated
identically to the same value in the training data.

**The field labels come from the data, not from the spec.** The STATS19 download
has a `labels/` directory beside `codes/`; `code_labels.json` was derived by
joining them on `collision_index`. This matters — `junction_detail` uses codes
13/16/17/18/19 and `pedestrian_crossing` uses 11–17, neither of which matches the
published guide. Transcribing from memory would have made every junction and
crossing value confidently wrong. Regenerate it the same way if the download
changes; anything unmapped renders as `code <n>` rather than inventing a label.

## Regenerating the policies

```bash
python CODE/simulator/pretrain_agents.py                  # the five shipped λ
python CODE/simulator/pretrain_agents.py --lambdas 0.05   # add another
```

An agent already in `models/rl_agents/` is reused when its hyperparameters match
the current Optuna config **and** its stored costs imply the right λ. At the time
of writing only `final_lam1.0` qualifies: `final/` and `final_lam0.1/` predate the
move to 5 confidence bins and would run a different policy from the one the
reported results describe.

Any λ outside the shipped five still works — it trains once and caches.
