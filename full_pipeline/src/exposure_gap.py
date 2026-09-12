"""
STEP 6 (part A) — POPULATION-WEIGHTED EXPOSURE GAP
=====================================================
What this does
---------------
For each city:
  1. Loads the aligned grid (every population cell x every month, with the
     raw satellite PM2.5 already interpolated onto it -- from Step 2).
  2. Runs the SELECTED calibration model (Step 5's best_model.joblib) on
     every grid cell/month to get a bias-corrected PM2.5 estimate. This is
     the best available estimate of "true" PM2.5 everywhere in the city,
     not just at the handful of official monitor locations.
  3. Computes the POPULATION-WEIGHTED exposure:
         exposure = sum(population_i * corrected_pm25_i) / sum(population_i)
     averaged across all 12 months to get one annual figure per city. This
     answers "what PM2.5 level does the average resident actually breathe."
  4. Computes the OFFICIAL MONITOR reading: the simple average of ground
     (OpenAQ) PM2.5 across that city's existing CPCB stations -- i.e. what
     the city's real monitoring network would report today, using only its
     own (non-representative) station locations.
  5. Reports the GAP = population-weighted exposure - official monitor
     average, in absolute ug/m3 and as a percentage. A positive gap means
     the official network is UNDER-reporting true population exposure
     (typically because monitors cluster away from the most exposed,
     usually poorer or peripheral, high-density areas).

Output: outputs/results/exposure_gap.json (used by the API and dashboard)
"""

import json

import numpy as np
import pandas as pd

from config import CITIES, DATA_PROCESSED_DIR, DATA_RAW_DIR, RESULTS_DIR
from src.calibration_models import predict as predict_pm25, load_model


def compute_corrected_grid(city: str, model) -> pd.DataFrame:
    grid = pd.read_csv(DATA_PROCESSED_DIR / f"aligned_grid_{city}.csv").copy()
    # predict_pm25() clips to a physically valid floor (see calibration_models.py) --
    # always go through it rather than calling model.predict() directly.
    grid["pm25_corrected"] = predict_pm25(model, grid)
    return grid


def population_weighted_exposure(grid: pd.DataFrame) -> float:
    """Annual population-weighted mean PM2.5: average the monthly weighted means."""
    monthly_means = []
    for month, month_df in grid.groupby("month"):
        w_mean = np.average(month_df["pm25_corrected"], weights=month_df["population"])
        monthly_means.append(w_mean)
    return float(np.mean(monthly_means))


def official_monitor_average(city: str) -> float:
    ground = pd.read_csv(DATA_RAW_DIR / f"openaq_ground_{city}.csv")
    # Simple (unweighted) average across the city's own stations & months --
    # this mirrors how a city would compute its officially reported figure.
    return float(ground["pm25_ground"].mean())


def run() -> dict:
    model = load_model("best_model")
    results = {}

    for city in CITIES:
        print(f"[exposure_gap] Computing exposure gap for {city} ...")
        grid = compute_corrected_grid(city, model)
        grid.to_csv(DATA_PROCESSED_DIR / f"corrected_grid_{city}.csv", index=False)

        pop_weighted = population_weighted_exposure(grid)
        official = official_monitor_average(city)
        gap = pop_weighted - official
        gap_pct = (gap / official) * 100 if official else None

        results[city] = {
            "population_weighted_exposure_ugm3": round(pop_weighted, 2),
            "official_monitor_average_ugm3": round(official, 2),
            "exposure_gap_ugm3": round(gap, 2),
            "exposure_gap_pct": round(gap_pct, 1),
            "interpretation": (
                "Official monitors UNDER-report true population exposure"
                if gap > 0 else
                "Official monitors OVER-report true population exposure"
            ),
        }
        print(
            f"    pop-weighted={pop_weighted:.1f} ug/m3 | "
            f"official={official:.1f} ug/m3 | gap={gap:+.1f} ug/m3 ({gap_pct:+.1f}%)"
        )

    with open(RESULTS_DIR / "exposure_gap.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[exposure_gap] Wrote outputs/results/exposure_gap.json")

    return results


if __name__ == "__main__":
    run()
