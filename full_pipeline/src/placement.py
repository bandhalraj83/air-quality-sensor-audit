"""
STEP 6 (part B) — GREEDY SENSOR-PLACEMENT ALGORITHM
======================================================
What this does
---------------
Recommends N_RECOMMENDED_STATIONS new monitor locations per city -- the
locations that would most reduce the population-weighted exposure gap if a
monitor were installed there.

Basis for comparison (read this before trusting the numbers)
----------------------------------------------------------------
We only have OpenAQ ground truth at the handful of EXISTING station
locations, never at arbitrary candidate cells. To compare "existing network
value" and "candidate cell value" on the same footing, this algorithm uses
the bias-CORRECTED satellite estimate (Step 6a's model) at every location --
including the existing stations -- rather than mixing raw ground readings
with model predictions. That keeps the greedy objective internally
consistent, at the cost of trusting the calibration model's extrapolation
away from monitored sites (a limitation worth stating in the report).

Algorithm (greedy, run once per city)
----------------------------------------
Naively picking whichever single cell nudges the network's numeric average
closest to the population-weighted target can select an odd, sparsely
populated cell purely because its value happens to balance the arithmetic --
that closes the number without actually improving the network's real-world
representativeness. Instead the score below directly captures "how much
does this cell matter and how badly is it currently unrepresented":

    score_i = population_i * |pm25_corrected_i - current_network_avg|

High score = a lot of people (population_i) currently breathing air that
looks nothing like what the existing network reports (large divergence from
current_network_avg) -- exactly the underrepresented, high-impact locations
a new monitor should prioritize.

1. `current_values` = annual-mean corrected PM2.5 at every EXISTING official
   station (this is what today's network "shows", on the model's footing);
   `current_avg` = their mean.
2. `target` = the annual population-weighted exposure for the city (Step 6a)
   -- reported alongside each pick for context, but not the selection score.
3. `candidates` = every population-grid cell at least MIN_SEPARATION_DEG
   away from all existing stations (no point recommending a monitor on top
   of one that already exists).
4. Repeat N_RECOMMENDED_STATIONS times:
     a. Score every remaining candidate as above, using the CURRENT
        `current_avg` (recomputed after each pick, so later picks account
        for sites already added this round).
     b. Pick the highest-scoring candidate; record the network average and
        remaining gap-to-target if it were added.
     c. Add it to `current_values`/`current_avg`; drop every remaining
        candidate within MIN_SEPARATION_DEG of it so later picks spread out
        instead of clustering next to the 1st (diminishing returns from
        redundant siting).
5. Return the ordered list of picks with each one's population, estimated
   PM2.5, and the resulting network average/gap -- so a reader can see both
   WHY a site was chosen (population x divergence) and its effect on the
   headline gap number.

This is intentionally a transparent, auditable greedy heuristic -- not a
full facility-location optimization -- matching the "simple placement
algorithm" called for in the project brief. A natural extension (noted in
the report template's Limitations section) is an integer-program formulation
that jointly optimizes all N sites and adds real-world siting constraints
(land access, power, security).
"""

import json

import numpy as np
import pandas as pd

from config import CITIES, DATA_PROCESSED_DIR, DATA_RAW_DIR, RESULTS_DIR, N_RECOMMENDED_STATIONS, MIN_SEPARATION_DEG
from src.calibration_models import predict as predict_pm25, load_model


def _haversine_deg(lat1, lon1, lat2, lon2):
    """Simple planar (degree-space) distance -- good enough at city scale for
    the small bounding boxes used here; swap for true haversine if cities
    grow larger."""
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def _existing_station_values(city: str, model) -> pd.DataFrame:
    """Annual-mean corrected PM2.5 at each existing CPCB station (model footing)."""
    training = pd.read_csv(DATA_PROCESSED_DIR / f"training_table_{city}.csv").copy()
    training["pm25_corrected"] = predict_pm25(model, training)
    annual = training.groupby(["station_id", "lat", "lon"], as_index=False)["pm25_corrected"].mean()
    return annual


def _candidate_cells(city: str, existing_stations: pd.DataFrame) -> pd.DataFrame:
    """Annual-mean corrected PM2.5 + population at every grid cell, minus
    cells too close to an existing station."""
    corrected = pd.read_csv(DATA_PROCESSED_DIR / f"corrected_grid_{city}.csv")
    annual = corrected.groupby(["lat", "lon"], as_index=False).agg(
        pm25_corrected=("pm25_corrected", "mean"),
        population=("population", "mean"),
    )

    keep_mask = np.ones(len(annual), dtype=bool)
    for _, stn in existing_stations.iterrows():
        d = _haversine_deg(annual["lat"].values, annual["lon"].values, stn["lat"], stn["lon"])
        keep_mask &= d >= MIN_SEPARATION_DEG
    return annual[keep_mask].reset_index(drop=True)


def greedy_placement(city: str, model, n_new: int = N_RECOMMENDED_STATIONS) -> dict:
    stations = _existing_station_values(city, model)
    candidates = _candidate_cells(city, stations)

    corrected_grid = pd.read_csv(DATA_PROCESSED_DIR / f"corrected_grid_{city}.csv")
    from src.exposure_gap import population_weighted_exposure
    target = population_weighted_exposure(corrected_grid)

    current_values = list(stations["pm25_corrected"].values)
    current_avg = float(np.mean(current_values))
    gap_before = target - current_avg

    remaining = candidates.copy()
    picks = []

    for i in range(n_new):
        if remaining.empty:
            break

        # Score = population x how far this cell's air diverges from what
        # the current network reports. High score = a lot of people
        # currently unrepresented by the existing monitors.
        divergence = np.abs(remaining["pm25_corrected"].values - current_avg)
        scores = remaining["population"].values * divergence
        best_idx = int(np.argmax(scores))
        best_row = remaining.iloc[best_idx]

        new_avg = float(np.mean(current_values + [float(best_row["pm25_corrected"])]))
        picks.append({
            "rank": i + 1,
            "lat": round(float(best_row["lat"]), 5),
            "lon": round(float(best_row["lon"]), 5),
            "estimated_pm25_ugm3": round(float(best_row["pm25_corrected"]), 1),
            "population_in_cell": round(float(best_row["population"]), 0),
            "network_avg_after_ugm3": round(new_avg, 2),
            "remaining_gap_after_ugm3": round(target - new_avg, 2),
        })

        current_values.append(float(best_row["pm25_corrected"]))
        current_avg = float(np.mean(current_values))
        d = _haversine_deg(remaining["lat"].values, remaining["lon"].values, best_row["lat"], best_row["lon"])
        remaining = remaining[d >= MIN_SEPARATION_DEG].reset_index(drop=True)

    return {
        "city": city,
        "target_population_weighted_exposure_ugm3": round(target, 2),
        "current_network_avg_ugm3": round(current_avg, 2),
        "gap_before_ugm3": round(gap_before, 2),
        "recommended_new_stations": picks,
        "gap_after_all_recommendations_ugm3": picks[-1]["remaining_gap_after_ugm3"] if picks else round(gap_before, 2),
    }


def run() -> dict:
    model = load_model("best_model")
    results = {}

    for city in CITIES:
        print(f"[placement] Running greedy placement for {city} ...")
        result = greedy_placement(city, model)
        results[city] = result
        print(
            f"    gap before={result['gap_before_ugm3']:+.1f} -> "
            f"gap after {len(result['recommended_new_stations'])} new stations="
            f"{result['gap_after_all_recommendations_ugm3']:+.1f} ug/m3"
        )
        for pick in result["recommended_new_stations"]:
            print(
                f"      #{pick['rank']}: ({pick['lat']}, {pick['lon']}) "
                f"est. PM2.5={pick['estimated_pm25_ugm3']} pop={int(pick['population_in_cell'])}"
            )

    with open(RESULTS_DIR / "placement_recommendations.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[placement] Wrote outputs/results/placement_recommendations.json")

    return results


if __name__ == "__main__":
    run()
