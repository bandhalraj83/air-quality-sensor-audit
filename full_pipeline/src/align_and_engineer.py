"""
STEP 2 — DATA ALIGNMENT & FEATURE ENGINEERING
===============================================
What this does
---------------
1. Loads the (coarser-resolution) satellite PM2.5 grid and the
   (finer-resolution) population grid for each city.
2. REGRIDS the satellite data onto the population grid's cell centers using
   bilinear interpolation (scipy.interpolate.RegularGridInterpolator) --
   this is the literal "align the satellite PM2.5 grid with the population
   grid" step from Week 2 of the plan, needed because the two sources come
   at different native resolutions/coordinate grids in the real datasets.
3. Extracts the satellite value at each CPCB station's location/month, for
   later joining against OpenAQ ground truth (used to train the bias
   correction models in Step 3).
4. Writes one tidy, aligned table per city to data/processed/, with columns:
   lat, lon, month, population, pm25_satellite
   This is the "population-weighted exposure feature" table referenced in
   the plan -- population-weighting itself (multiplying by population and
   summing) happens later in exposure_gap.py, once we have a BIAS-CORRECTED
   PM2.5 value to weight. Here we just make sure every population cell has a
   matching satellite estimate for every month.

Why bilinear interpolation via RegularGridInterpolator
--------------------------------------------------------
Both the satellite and population sources are regular lat/lon grids (just at
different resolutions), so RegularGridInterpolator is the natural, fast
choice; for irregular/swath data you'd use scipy.interpolate.griddata
instead. Edge cells that fall outside the satellite grid's convex hull are
filled by nearest-neighbor extrapolation (bounds_error=False, fill_value=None
falls back to nearest edge value) so no NaNs leak into later steps.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from config import CITIES, DATA_RAW_DIR, DATA_PROCESSED_DIR, MONTHS


def _load_city_sources(city: str):
    pop = pd.read_csv(DATA_RAW_DIR / f"population_grid_{city}.csv")
    sat = pd.read_csv(DATA_RAW_DIR / f"satpm_grid_{city}.csv")
    stations = pd.read_csv(DATA_RAW_DIR / f"cpcb_stations_{city}.csv")
    ground = pd.read_csv(DATA_RAW_DIR / f"openaq_ground_{city}.csv")
    return pop, sat, stations, ground


def _build_interpolator(sat_month_df: pd.DataFrame) -> RegularGridInterpolator:
    """Build a bilinear interpolator for one month's satellite grid."""
    lats = np.sort(sat_month_df["lat"].unique())
    lons = np.sort(sat_month_df["lon"].unique())
    grid = sat_month_df.pivot(index="lat", columns="lon", values="pm25_satellite")
    grid = grid.reindex(index=lats, columns=lons)  # ensure sorted order matches lats/lons
    values = grid.values
    # Fill any stray NaNs (shouldn't happen with synthetic data, but real
    # satellite products have cloud-gap NaNs) via simple forward/backward fill.
    if np.isnan(values).any():
        values = pd.DataFrame(values).ffill().bfill().values
    return RegularGridInterpolator(
        (lats, lons), values, method="linear", bounds_error=False, fill_value=None
    )


def _interpolate_onto_points(interp: RegularGridInterpolator, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    pts = np.column_stack([lat, lon])
    return interp(pts)


def align_city(city: str) -> pd.DataFrame:
    """Return the population grid with an aligned pm25_satellite column, per month."""
    pop, sat, stations, ground = _load_city_sources(city)

    aligned_rows = []
    for month in MONTHS:
        sat_month = sat[sat["month"] == month]
        interp = _build_interpolator(sat_month)
        month_df = pop.copy()
        month_df["month"] = month
        month_df["pm25_satellite"] = _interpolate_onto_points(
            interp, month_df["lat"].values, month_df["lon"].values
        )
        aligned_rows.append(month_df)

    aligned = pd.concat(aligned_rows, ignore_index=True)
    aligned = aligned[["lat", "lon", "month", "population", "pm25_satellite"]]
    return aligned


def align_stations_to_satellite(city: str) -> pd.DataFrame:
    """
    Build the training table: for each CPCB station and month, the
    satellite-estimated PM2.5 at that exact point (interpolated), joined with
    the OpenAQ ground-truth reading. This is the supervised dataset for the
    bias-correction models in Step 3.
    """
    pop, sat, stations, ground = _load_city_sources(city)

    rows = []
    for month in MONTHS:
        sat_month = sat[sat["month"] == month]
        interp = _build_interpolator(sat_month)
        sat_at_stations = _interpolate_onto_points(
            interp, stations["lat"].values, stations["lon"].values
        )
        month_df = stations.copy()
        month_df["month"] = month
        month_df["pm25_satellite"] = sat_at_stations
        rows.append(month_df)

    sat_at_station_df = pd.concat(rows, ignore_index=True)
    training_df = sat_at_station_df.merge(
        ground, on=["station_id", "lat", "lon", "month"], how="inner"
    )
    training_df["city"] = city
    return training_df[["city", "station_id", "lat", "lon", "month", "pm25_satellite", "pm25_ground"]]


def run() -> None:
    all_training_rows = []
    for city in CITIES:
        print(f"[align_and_engineer] Aligning grids for {city} ...")

        aligned = align_city(city)
        aligned.to_csv(DATA_PROCESSED_DIR / f"aligned_grid_{city}.csv", index=False)

        training_df = align_stations_to_satellite(city)
        training_df.to_csv(DATA_PROCESSED_DIR / f"training_table_{city}.csv", index=False)
        all_training_rows.append(training_df)

        print(f"    aligned grid: {len(aligned)} rows | training rows: {len(training_df)}")

    combined_training = pd.concat(all_training_rows, ignore_index=True)
    combined_training.to_csv(DATA_PROCESSED_DIR / "training_table_all_cities.csv", index=False)
    print(
        f"[align_and_engineer] Done. Combined training table: "
        f"{len(combined_training)} rows -> data/processed/training_table_all_cities.csv"
    )


if __name__ == "__main__":
    run()
