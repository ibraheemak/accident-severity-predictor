#!/usr/bin/env python3
"""Acquisition costs for the 10 bundles.

Costs represent the real-world time required to acquire each bundle:
- ask_caller: seconds to ask questions on the phone
- map_lookup: milliseconds for local GIS database queries
- external_api: milliseconds for network API calls

Philosophy:
- Caller questions are expensive (seconds to ask and listen)
- Demographics harder than casualties (more detail to extract)
- External APIs are cheapest (automated, parallelizable)
"""

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
    """Return bundle costs as {bundle_index: cost_seconds}."""
    return BUNDLE_COSTS.copy()

def print_costs_summary():
    """Print cost summary by tier."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
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
