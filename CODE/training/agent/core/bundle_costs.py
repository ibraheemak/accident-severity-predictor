#!/usr/bin/env python3
"""Acquisition costs for the 10 bundles.

Costs represent the real-world time required to acquire each bundle:
- ask_caller: seconds to ask a question and hear the answer
- map_lookup: local GIS queries against a confirmed location
- external_api: network round trips

Philosophy:
- Caller questions are expensive (a question plus the caller's answer)
- Demographics harder than casualties (more detail to extract)
- External APIs are cheapest (automated, parallelizable)

These values treat a cost as the MARGINAL DECISION DELAY an acquisition adds,
not the wall-clock duration of the act. A handler asks "is anyone injured?" on
every call regardless of any model, and the questions overlap with dispatch
rather than postponing it, so what the agent spends is the extra delay beyond
what would have happened anyway. Read as question durations these numbers are
far too low; read as marginal delay they are the intended quantity. State which
definition is in use before quoting any acquisition time.

Only the RATIOS matter to the learned policy. The reward uses
C_i = lambda * seconds_i, so scaling every cost by k and dividing lambda by k
leaves C_i -- and the policy -- identical.
"""

import json
import os

# Bundle costs in seconds
BUNDLE_COSTS = {
    # ask_caller (phone time — seconds, most expensive)
    0: 1.0,    # vehicles: "What vehicles involved?"
    1: 2.0,    # casualties: "How many hurt? Types?"
    2: 3.0,    # casualty_demographics: "Ages, genders, who?" (hardest caller question)

    # map_lookup (local GIS — seconds, medium)
    3: 0.050,  # road_class: Road class lookup
    4: 0.050,  # road_environment: Speed limit, urban/rural
    5: 0.040,  # junction: Junction lookup
    6: 0.060,  # crash_history: 500m radius query (most expensive lookup)

    # external_api (network calls — seconds, cheapest)
    7: 0.020,  # aadf: Traffic flow API (find the nearest count point)
    8: 0.010,  # osm: OpenStreetMap query (lightweight)
    9: 0.030,  # meteo: Weather API (simple point lookup)
}

# Cost tier labels (for analysis, not for MDP)
BUNDLE_TIERS_COSTS = {
    "ask_caller": (0, 1, 2),
    "map_lookup": (3, 4, 5, 6),
    "external_api": (7, 8, 9),
}

def get_costs_dict() -> dict:
    """Return bundle costs as {bundle_index: cost_seconds}.

    RL_BUNDLE_COSTS, if set, overrides the table above. It holds a JSON object
    keyed by bundle index, e.g. '{"0": 5, "1": 8, "2": 20}' -- keys given are
    replaced, keys omitted keep their value here. This exists so a cost
    SENSITIVITY sweep can vary the assumption across runs without rewriting
    this file between them, which would make it impossible to tell afterwards
    which costs produced which output directory.

    It is an override for sweeps, not a way to pick a flattering setting: any
    analysis using it has to report every model it ran, not the best one.
    """
    costs = BUNDLE_COSTS.copy()
    raw = os.environ.get("RL_BUNDLE_COSTS")
    if raw:
        costs.update({int(k): float(v) for k, v in json.loads(raw).items()})
    return costs

def print_costs_summary():
    """Print cost summary by tier."""
    import sys
    from pathlib import Path
    # train_classifier lives at CODE/training/train, two levels up from core/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
    from train_classifier import BUNDLE_NAMES

    print("\nBundle Acquisition Costs:")
    print("=" * 70)

    for tier_name, bundle_indices in BUNDLE_TIERS_COSTS.items():
        print(f"\n{tier_name.upper()}:")
        costs_in_tier = [BUNDLE_COSTS[i] for i in bundle_indices]
        total = sum(costs_in_tier)
        avg = total / len(costs_in_tier)

        for i in bundle_indices:
            name = BUNDLE_NAMES[i]
            cost = BUNDLE_COSTS[i]
            unit = "sec" if cost >= 0.1 else "ms"
            if cost >= 0.1:
                print(f"  [{i}] {name:20s} {cost:.3f} {unit}")
            else:
                print(f"  [{i}] {name:20s} {cost*1000:.1f} {unit}")

        print(f"  → Total tier: {total:.3f}s (avg: {avg:.3f}s)")

    print(f"\nTotal time if acquiring all 10 bundles: {sum(BUNDLE_COSTS.values()):.3f}s")
    print("=" * 70)

if __name__ == "__main__":
    print_costs_summary()
