#!/usr/bin/env python3
"""weather_agreement.py -- does the officer's weather field agree with the sky?

Part 1 of the "Officer vs. Data Gap" experiment. Compares STATS19's subjective
weather field against Open-Meteo's objective hourly reconstruction.

Output for fig1 (agreement_matrix.png):
  agreement_crosstab.csv        officer category x objective condition, counts
  agreement_officer_profile.csv open-meteo conditions per officer category

Usage:
  python CODE/experiments/officer_vs_data/weather_agreement.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "train"))

import pandas as pd
from train_classifier import SOURCE_COLS, TARGET_COL

OUT_DIR = Path("docs/figures/officer_vs_data")
SPLIT_DIR = Path("data/splits")
SPLITS = ("train", "valid", "test")

OFFICER_LABELS = {
    1: "Fine, no high winds", 2: "Raining, no high winds", 3: "Snowing, no high winds",
    4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds",
    7: "Fog or mist", 8: "Other", 9: "Unknown",
}

def objective_condition(code: pd.Series) -> pd.Series:
    out = pd.Series("Dry", index=code.index, dtype=object)
    out[code.between(45, 48)] = "Fog"
    out[code.between(51, 57)] = "Drizzle"
    out[code.between(61, 67)] = "Rain"
    out[code.between(71, 77)] = "Snow"
    out[code.between(80, 82)] = "Rain"
    out[code.between(85, 86)] = "Snow"
    return out

OBJ_ORDER = ["Dry", "Drizzle", "Rain", "Snow", "Fog"]


def load_raw() -> pd.DataFrame:
    cols = ["weather_conditions", *SOURCE_COLS["meteo"], TARGET_COL]
    frames = []
    for name in SPLITS:
        df = pd.read_csv(SPLIT_DIR / f"{name}.csv", usecols=cols)
        df["split"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["obj_condition"] = objective_condition(df["weather_code"])
    return df


def officer_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, label in OFFICER_LABELS.items():
        sub = df[df["weather_conditions"] == code]
        if sub.empty:
            continue
        rows.append({
            "officer_code": code, "officer_label": label, "n": len(sub),
            "share": len(sub) / len(df),
            "mean_precip_mm": sub["precipitation_mm"].mean(),
            "pct_hours_wet": (sub["precipitation_mm"] > 0.1).mean(),
            "mean_wind_kmh": sub["wind_speed_10m_kmh"].mean(),
            "pct_high_wind": (sub["wind_speed_10m_kmh"] >= 29.0).mean(),
            "mean_temp_c": sub["temperature_2m_c"].mean(),
            "mean_snowfall_cm": sub["snowfall_cm"].mean(),
            "pct_obj_snow": (sub["obj_condition"] == "Snow").mean(),
        })
    return pd.DataFrame(rows)


def crosstab(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["weather_conditions"])
    ct = pd.crosstab(sub["weather_conditions"].map(OFFICER_LABELS), sub["obj_condition"])
    ct = ct.reindex(columns=[c for c in OBJ_ORDER if c in ct.columns], fill_value=0)
    ct = ct.reindex([OFFICER_LABELS[c] for c in OFFICER_LABELS if OFFICER_LABELS[c] in ct.index])
    return ct


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    print(f"{len(df):,} collisions across {'/'.join(SPLITS)}")

    profile = officer_profile(df)
    profile.to_csv(OUT_DIR / "agreement_officer_profile.csv", index=False)
    print("Wrote agreement_officer_profile.csv")

    ct = crosstab(df)
    ct.to_csv(OUT_DIR / "agreement_crosstab.csv")
    print("Wrote agreement_crosstab.csv")
    print(f"\nAll CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
