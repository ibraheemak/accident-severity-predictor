# Predicting Road Accident Severity in the UK
### A Multi-Source Machine Learning Approach

> **Course:** Project in Artificial Intelligence (02360502) — Spring 2026  
> **Students:** Ibraheem Akaree : ibraheema@campus.technion.ac.il , Sewar Abahry :sewar.abahry@campus.technion.ac.il  
> **Advisor:** Yotam Gavish

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
  - [Multiclass Classification](#multiclass-classification)
  - [Feature Engineering Pipeline](#feature-engineering-pipeline)
  - [Active Feature Acquisition (RL)](#active-feature-acquisition-rl)
- [Data Sources](#data-sources)
- [System Components](#system-components)
- [Evaluation & Baselines](#evaluation--baselines)
- [Research Extensions](#research-extensions)
- [Setup & Installation](#setup--installation)

---

## Overview

This project builds an intelligent severity prediction system for road traffic accidents in Great Britain. Given only the information available in the first moments after a 999 call, the system predicts the severity of the accident and decides — in real time — which additional data is worth acquiring before making a final prediction.

The system combines supervised classification with a Reinforcement Learning (RL) agent that dynamically trades off prediction accuracy against data acquisition cost and latency.

---

## Problem Statement

When an accident is reported to emergency services, dispatchers must decide within seconds what level of response to send. This project addresses the question:

> *Given only the information available at the moment of a 999 call, can we accurately predict accident  severity?*

An accurate severity predictor at dispatch centres could optimise emergency resource allocation, reduce response times, and ultimately save lives.

---

## Solution Architecture

### Classification

The target variable is the STATS19 `collision_severity` field (Fatal / Serious / Slight), also trained as a binary Slight-vs-Severe task. A supervised classifier is trained **per acquisition subset** — one for every combination of the acquisition bundles defined below — and frozen before RL training begins. These classifiers are the environment the RL agent acts on.

> **Design revision (August 2026).** This was originally *eight* classifiers, one per subset of the three external sources (OSM / AADF / Weather), each trained on randomly masked STATS19 fields to cope with partial inputs. Both halves of that design were replaced:
>
> - **Random masking was dropped.** Training on masked copies cost accuracy at every level of missingness relative to training a dedicated model per subset, and it made every headline metric a pooled average across missingness levels rather than a number about any real deployment scenario. The `masked` data variant and `generate_masked_splits.R` still exist for reference (see `docs/data_pipeline.md` §13) but are no longer used.
> - **The action space was widened past the three external sources.** Those three turned out to carry very little of the available signal, so the STATS19 fields became acquirable too. Since one action per field means 2^31 classifiers, fields are grouped into **10 bundles**, giving a tractable 2^10 = 1024 pre-trained classifiers.

### Feature Engineering Pipeline

Rather than relying solely on STATS19, each accident record is enriched with three external sources requiring non-trivial spatial or temporal joining logic:

| Source | Features Added | Join Method |
|--------|---------------|-------------|
| **OpenStreetMap** | Speed cameras, lane count, road dividers, street lighting | Spatial radius query |
| **DfT AADF Traffic** | Annual Average Daily Flow (road busyness) | Spatial nearest-neighbour (500m) |
| **Open-Meteo Historical** | Temperature, precipitation, wind speed, visibility, humidity, snowfall | API call by lat/lon + datetime |

### Active Feature Acquisition (RL)

The system is extended into an **Active Feature Acquisition (AFA)** framework modelled as a Markov Decision Process (MDP), solved with Reinforcement Learning.

#### The acquisition bundles

An action is not a single column but one **real-world acquisition act**: one question put to the caller, one lookup against the collision coordinate, or one external API call. Features are grouped accordingly, which is both what makes the cost model defensible and what cuts the classifier count from 2^31 to 2^10.

| Kind | Bundle | Features |
|---|---|---|
| **Ask the caller** | `vehicles` | `number_of_vehicles` + 7 vehicle-type flags |
| | `casualties` | How many hurt, and who: pedestrian / driver / passenger / motorcyclist / cyclist counts |
| | `casualty_demographics` | Youngest & oldest age, child / elderly / female counts |
| **Map lookup** | `road_class` | Road class & number (1st + 2nd), road type, trunk-road flag |
| | `road_environment` | Speed limit, urban/rural, light conditions, weather conditions |
| | `junction` | Junction detail & control, pedestrian crossing |
| | `crash_history` | Spatial risk density: prior-year accident rate, severe fraction and fatal rate within 500 m, distance to nearest prior accident |
| **External API** | `osm` | Traffic calming, signals, lane count, median, speed camera |
| | `aadf` | Annual Average Daily Flow, distance to count point |
| | `meteo` | Temperature, precipitation, wind, weather code, snowfall, snow depth |

**Each bundle is an independent action with its own independently-settable acquisition cost `C_i`.** The "Kind" column is a descriptive label for analysis only — it does not group actions into one step, and bundles of the same kind are not required to cost the same. The cost values themselves are defined as part of the RL work, not yet set.

**Free (no acquisition needed):** the call timestamp only — day-of-week and hour, cyclically encoded. At the moment the phone rings, nothing else is known without asking or looking it up. Every bundle is strictly *additive*: acquiring one can only add information, never remove it.

**State:** `s_t = (x_free, x_acquired, m_t, p_t)`
- `x_free` — the call timestamp, available automatically
- `x_acquired` — bundles acquired so far during the episode
- `m_t` — 10-bit mask of which bundles have been obtained (also the cache key for the matching frozen classifier)
- `p_t` — current predicted probability distribution over severity classes

**Actions:**
- `A0` — **Predict & Terminate** (use current `p_t` as final output)
- `A1…A10` — **Acquire bundle *i***, at that bundle's tier cost

**Reward Function:**

```
Reward = R_terminal + Σ R_step(t)

R_step(t) = α · ΔConfidence_t − C_i
R_terminal = β · Accuracy(ŷ, y)
```

- `ΔConfidence_t` — reduction in prediction entropy after acquiring feature `i`
- `C_i` — acquisition cost (higher for external APIs than STATS19 fields)
- `Accuracy(ŷ, y)` — Macro F1-score against ground truth, delivered at termination
- `α`, `β` — tunable weighting parameters (α ≪ β)

---

## Data Sources

| Source | URL | Role | Cost |
|--------|-----|------|------|
| STATS19 (DfT) | [github.com/ropensci/stats19](https://github.com/ropensci/stats19) | Primary dataset + labels | Free |
| DfT AADF Traffic | [roadtraffic.dft.gov.uk/downloads](https://roadtraffic.dft.gov.uk/downloads) | Traffic volume feature | Free |
| OpenStreetMap | [overpass-api.de](https://overpass-api.de) | Road infrastructure features | Free |
| Open-Meteo Historical | [open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api) | Hourly weather at exact location | Paid (€99 Professional) |

### STATS19 Tables

The primary dataset links three tables via `accident_index`:

- **Accidents** — location (lat/lon), date/time, road type, weather, light conditions, speed limit, junction type, severity
- **Casualties** — age, sex, casualty class, injury severity per person
- **Vehicles** — vehicle type, manoeuvre, driver age, engine capacity

---

## System Components

```
┌─────────────────────────────────────────────────────────┐
│  1. Data Acquisition                                    │
│     Downloads STATS19 + DfT AADF; caches OSM & Weather │
├─────────────────────────────────────────────────────────┤
│  2. Feature Engineering Pipeline                        │
│     Spatial + temporal joins; annotates acquisition     │
│     costs per feature                                   │
├─────────────────────────────────────────────────────────┤
│  3. Preprocessing Module                                │
│     Missing value imputation, encoding, normalisation   │
├─────────────────────────────────────────────────────────┤
│  4. Classifier Training (×1024)                         │
│     One model per acquisition-bundle subset, cached to  │
│     models/subsets/ ; frozen before RL training begins  │
├─────────────────────────────────────────────────────────┤
│  5. Evaluation & Experiment Runner 1 (pre-RL)           │
│     Ablation study, leakage audit, officer vs. data     │
│     gap, temporal drift analysis                        │
├─────────────────────────────────────────────────────────┤
│  6. RL Agent Training                                   │
│     AFA policy trained over frozen classifier env.      │
│     One episode = one accident record                   │
├─────────────────────────────────────────────────────────┤
│  7. Evaluation & Experiment Runner 2 (post-RL)          │
│     RL agent vs. 3 baselines; tables + plots            │
├─────────────────────────────────────────────────────────┤
│  8. Inference Pipeline                                  │
│     Live accident → RL policy → real-time API calls     │
│     → predicted severity + confidence score             │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation & Baselines

The RL agent is benchmarked against three baselines:

| Baseline | Description | Cost | Accuracy |
|----------|-------------|------|----------|
| **No-Acquisition** | Classifier on the free timestamp only | Zero | Lower bound |
| **Full-Acquisition** | Always acquires all 10 bundles | Maximum | Upper bound |
| **Best Fixed Subset** | The single best-scoring subset at the RL agent's average budget, chosen offline from the sweep below | Same as RL (avg) | Cannot adapt per-accident |
| **Static Feature Priority** | Fixed acquisition *order*, stopping at the RL agent's average budget | Same as RL (avg) | Cannot adapt per-accident |

**Best Fixed Subset is the baseline that matters.** Because all 1024 classifiers are pre-trained, the best non-adaptive policy at any budget is directly computable. The RL agent only justifies itself if it beats that — i.e. only if the *right* subset genuinely differs from one collision to the next.

### Value-of-Information Curve (1024-subset sweep)

`CODE/training/train/bundle_sweep.py` trains all 1024 acquisition subsets (binary Slight-vs-Severe, `normalized` variant, ~29 min) and caches them to `models/subsets/`. Results:

**Timestamp only: 0.5487 valid / 0.5408 test macro F1 → full acquisition: 0.7406 / 0.7377.** The whole action space is worth **+0.19 macro F1**.

Mean marginal value of each bundle, averaged over all 512 subsets that lack it:

| Bundle | Kind | Mean marginal | Alone | Helps in |
|---|---|---|---|---|
| `casualties` | ask caller | **+0.0402** | +0.106 | 100% of subsets |
| `vehicles` | ask caller | **+0.0351** | +0.093 | 100% |
| `casualty_demographics` | ask caller | **+0.0319** | +0.088 | 100% |
| `road_environment` | map lookup | +0.0097 | +0.070 | 98% |
| `crash_history` | map lookup | +0.0079 | +0.067 | 96% |
| `junction` | map lookup | +0.0061 | +0.052 | 95% |
| `road_class` | map lookup | +0.0056 | +0.057 | 88% |
| `osm` | external API | +0.0032 | +0.046 | 81% |
| `aadf` | external API | +0.0015 | +0.035 | 68% |
| `meteo` | external API | −0.0008 | +0.002 | 36% |

Two findings shape the RL problem:

1. **What the caller tells you dominates.** The three caller bundles are worth 4–25× any lookup or API bundle, and they are the only ones that help in *every* context. The external sources contribute very little — which is what motivated widening the action space to the STATS19 fields in the first place.
2. **Value saturates at about five bundles.** The three caller bundles plus `road_class` and `road_environment` reach 0.7364 test — statistically indistinguishable from acquiring all ten (0.7377). Past that point every further bundle is worth ≤ +0.004, and the ranking among the top subsets is within noise.

**Open question this raises.** These are *population-average* values, so they describe the best **fixed** policy. If the right subset is much the same for every collision, an adaptive agent cannot beat a hardcoded five-bundle rule and the AFA framing has no headroom. Measuring the gap between a per-instance oracle and the best fixed subset — cheap, since all 1024 models are cached — is the next step, and it gates the RL work.

---

## Research Extensions

- **Confidence Scores** — Calibrated uncertainty estimates flagging low-confidence predictions (especially Fatal/Serious) for human review.
- **Leakage Audit** — Measures the accuracy gap between a model trained on all STATS19 features vs. only dispatch-time features, quantifying the real-world deployability cost.
- **Ablation Study / Value-of-Information Curve** — Full evaluation of all 1024 bundle subsets (`CODE/training/train/bundle_sweep.py`) to isolate each bundle's independent and combined contribution; results inform cost penalty calibration `C_i`. See below.
- **Officer vs. Data Gap** — Compares the subjective STATS19 weather field (filled by the attending officer) against objective Open-Meteo measurements at the same location and time.
- **Temporal Drift** — Trains on earlier STATS19 years and tests on recent ones to assess stability of accident severity patterns over time.

#hello
