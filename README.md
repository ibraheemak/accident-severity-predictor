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

### Multiclass Classification

The target variable is the STATS19 `collision_severity` field (Fatal / Serious / Slight). Eight supervised classifiers are trained — one per possible combination of the three external data sources (OSM, AADF, Weather) — each with random masking of STATS19 fields to handle partial inputs gracefully.

### Feature Engineering Pipeline

Rather than relying solely on STATS19, each accident record is enriched with three external sources requiring non-trivial spatial or temporal joining logic:

| Source | Features Added | Join Method |
|--------|---------------|-------------|
| **OpenStreetMap** | Speed cameras, lane count, road dividers, street lighting | Spatial radius query |
| **DfT AADF Traffic** | Annual Average Daily Flow (road busyness) | Spatial nearest-neighbour (500m) |
| **Open-Meteo Historical** | Temperature, precipitation, wind speed, visibility, humidity, snowfall | API call by lat/lon + datetime |

### Active Feature Acquisition (RL)

The system is extended into an **Active Feature Acquisition (AFA)** framework modelled as a Markov Decision Process (MDP), solved with Reinforcement Learning.

**State:** `s_t = (x_base, x_acquired, m_t, p_t)`
- `x_base` — STATS19 fields available automatically at the 999 call
- `x_acquired` — features acquired so far during the episode
- `m_t` — binary mask of which features have been obtained
- `p_t` — current predicted probability distribution over severity classes

**Actions:**
- `A0` — **Predict & Terminate** (use current `p_t` as final output)
- `A1…Ak` — **Acquire a STATS19 field** (ask the dispatcher)
- `A(k+1)` — **Acquire OSM road infrastructure data**
- `A(k+2)` — **Acquire DfT traffic volume data**
- `A(k+3)` — **Acquire Open-Meteo weather data**

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
│     Generates randomly masked training copies           │
├─────────────────────────────────────────────────────────┤
│  4. Classifier Training (×8)                            │
│     One model per external-source subset                │
│     Frozen before RL training begins                    │
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
| **No-Acquisition** | Classifier on base STATS19 only | Zero | Lower bound |
| **Full-Acquisition** | Always acquires all 3 external sources | Maximum | Upper bound |
| **Static Feature Priority** | Fixed acquisition order (e.g. Weather → OSM → AADF) given the same average budget as the RL agent | Same as RL (avg) | Cannot adapt per-accident |

---

## Research Extensions

- **Confidence Scores** — Calibrated uncertainty estimates flagging low-confidence predictions (especially Fatal/Serious) for human review.
- **Leakage Audit** — Measures the accuracy gap between a model trained on all STATS19 features vs. only dispatch-time features, quantifying the real-world deployability cost.
- **Ablation Study** — Full evaluation of all 8 external-source subsets to isolate each source's independent and combined contribution; results inform cost penalty calibration `C_i`.
- **Officer vs. Data Gap** — Compares the subjective STATS19 weather field (filled by the attending officer) against objective Open-Meteo measurements at the same location and time.
- **Temporal Drift** — Trains on earlier STATS19 years and tests on recent ones to assess stability of accident severity patterns over time.

