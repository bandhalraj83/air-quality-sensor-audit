"""
STEP 1 — DATA ACQUISITION (synthetic stand-in)
================================================
What this does
---------------
The real project ships CPCB station lists, SatPM satellite PM2.5, WorldPop
population grids, and OpenAQ ground readings as a ready-to-use extract. Since
that package isn't attached here, this script GENERATES synthetic files with
the exact same schema/columns the rest of the pipeline expects, so the whole
project runs end-to-end right now.

Swap-in instructions: once you have the real extract, save each source as
data/raw/<name>_<city>.csv with the columns below, and skip this script --
every downstream step reads only from data/raw/, so nothing else changes.

Files produced (per city):
  1. cpcb_stations_<city>.csv   columns: station_id, lat, lon
  2. population_grid_<city>.csv columns: lat, lon, population           (WorldPop-like)
  3. satpm_grid_<city>.csv      columns: lat, lon, month, pm25_satellite (SatPM-like)
  4. openaq_ground_<city>.csv   columns: station_id, lat, lon, month, pm25_ground (OpenAQ-like)

Why the synthetic numbers look the way they do
------------------------------------------------
- Population is modelled as a mixture of Gaussian "urban cores" per city
  (config.CITIES[...]['pop_centers']) so some grid cells are dense and some
  are sparse, like a real city.
- True PM2.5 is spatially smooth and rises toward dense population centers
  (real pollution correlates with traffic/industry/density), plus a seasonal
  wave (higher in winter months 11,12,1,2 - matching real Indian PM2.5
  seasonality) and per-cell noise.
- Satellite PM2.5 = true PM2.5 passed through a SYSTEMATIC BIAS (an
  under-reporting multiplicative factor plus an additive offset that gets
  worse at high concentrations, mimicking known satellite AOD-to-PM2.5
  retrieval issues in polluted urban cores) + measurement noise. This is the
  bias Model 1 / Model 2 have to learn to correct.
- CPCB official stations are placed in a CLUSTER around a city's
  administrative/central area (station_center), not spread proportional to
  population -- this is the real-world siting pattern the whole project is
  auditing.
- OpenAQ ground readings = the TRUE pm25 at station locations/months (plus
  small sensor noise), used as ground truth to train/validate the
  bias-correction models.
"""

import numpy as np
import pandas as pd

from config import CITIES, DATA_RAW_DIR, POP_GRID_RESOLUTION_DEG, MONTHS, RANDOM_SEED


def _make_grid(bbox: dict, resolution: float) -> pd.DataFrame:
    """Return a regular lat/lon grid of cell-center points covering bbox."""
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], resolution)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()})


def _gaussian_bump(lat, lon, center_lat, center_lon, weight, sigma=0.05):
    d2 = (lat - center_lat) ** 2 + (lon - center_lon) ** 2
    return weight * np.exp(-d2 / (2 * sigma ** 2))


def _population_field(grid: pd.DataFrame, city_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    field = np.zeros(len(grid))
    for (clat, clon, w) in city_cfg["pop_centers"]:
        field += _gaussian_bump(grid["lat"].values, grid["lon"].values, clat, clon, w, sigma=0.06)
    field += 0.02 * rng.random(len(grid))  # background population noise
    field = np.clip(field, 0, None)
    # scale so total matches the configured city population (rough demo scaling)
    field = field / field.sum() * city_cfg["total_population"]
    return field


def _true_pm25_field(grid: pd.DataFrame, city_cfg: dict, month: int, rng: np.random.Generator) -> np.ndarray:
    """Spatially-smooth 'true' PM2.5 surface: higher near dense cores, seasonal, noisy."""
    density_signal = np.zeros(len(grid))
    for (clat, clon, w) in city_cfg["pop_centers"]:
        density_signal += _gaussian_bump(grid["lat"].values, grid["lon"].values, clat, clon, w, sigma=0.08)
    density_signal = density_signal / density_signal.max()

    # Winter months (Nov-Feb) run higher in most Indian cities; monsoon (Jun-Sep) lower.
    seasonal_multiplier = {
        1: 1.35, 2: 1.20, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.75,
        7: 0.70, 8: 0.72, 9: 0.85, 10: 1.05, 11: 1.30, 12: 1.40,
    }[month]

    base = 45 + 70 * density_signal          # roughly 45-115 ug/m3 baseline by location
    true_pm25 = base * seasonal_multiplier
    true_pm25 += rng.normal(0, 4, size=len(grid))  # local noise
    return np.clip(true_pm25, 5, None)


def _satellite_bias(true_pm25: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a systematic, concentration-dependent bias + noise to simulate
    satellite-derived PM2.5 retrieval error relative to truth.
    """
    multiplicative = 0.80                       # satellite tends to UNDER-read
    additive_penalty = 0.05 * np.clip(true_pm25 - 60, 0, None)  # worse at high pollution
    noise = rng.normal(0, 6, size=len(true_pm25))
    sat = true_pm25 * multiplicative - additive_penalty + noise + 8  # small constant offset
    return np.clip(sat, 3, None)


def _place_official_stations(city_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = city_cfg["n_official_stations"]
    clat, clon = city_cfg["station_center"]
    spread = city_cfg["station_spread"]
    lats = rng.normal(clat, spread, size=n)
    lons = rng.normal(clon, spread, size=n)
    # clip into bbox
    bbox = city_cfg["bbox"]
    lats = np.clip(lats, bbox["lat_min"], bbox["lat_max"])
    lons = np.clip(lons, bbox["lon_min"], bbox["lon_max"])
    return pd.DataFrame({
        "station_id": [f"STN_{i:03d}" for i in range(n)],
        "lat": lats,
        "lon": lons,
    })


def generate_all(seed: int = RANDOM_SEED) -> None:
    rng = np.random.default_rng(seed)

    for city, city_cfg in CITIES.items():
        print(f"[generate_synthetic_data] Building synthetic sources for {city} ...")

        # --- WorldPop-like population grid (fine resolution) -----------------
        pop_grid = _make_grid(city_cfg["bbox"], POP_GRID_RESOLUTION_DEG)
        pop_grid["population"] = _population_field(pop_grid, city_cfg, rng)
        pop_grid.to_csv(DATA_RAW_DIR / f"population_grid_{city}.csv", index=False)

        # --- CPCB station list -------------------------------------------------
        stations = _place_official_stations(city_cfg, rng)
        stations.to_csv(DATA_RAW_DIR / f"cpcb_stations_{city}.csv", index=False)

        # --- SatPM-like satellite grid (coarser native resolution), monthly ---
        sat_rows = []
        # Use a coarser grid for the "native" satellite resolution; alignment
        # step will regrid this onto the fine population grid.
        from config import SATPM_NATIVE_RESOLUTION_DEG
        sat_grid_coords = _make_grid(city_cfg["bbox"], SATPM_NATIVE_RESOLUTION_DEG)
        for month in MONTHS:
            true_pm25 = _true_pm25_field(sat_grid_coords, city_cfg, month, rng)
            sat_pm25 = _satellite_bias(true_pm25, rng)
            month_df = sat_grid_coords.copy()
            month_df["month"] = month
            month_df["pm25_satellite"] = sat_pm25
            sat_rows.append(month_df)
        sat_df = pd.concat(sat_rows, ignore_index=True)
        sat_df.to_csv(DATA_RAW_DIR / f"satpm_grid_{city}.csv", index=False)

        # --- OpenAQ-like ground truth at station locations, monthly -----------
        ground_rows = []
        for month in MONTHS:
            # true PM2.5 evaluated AT STATION locations (not the coarse grid)
            station_pts = stations[["lat", "lon"]].copy()
            true_at_stations = _true_pm25_field(station_pts, city_cfg, month, rng)
            sensor_noise = rng.normal(0, 2.5, size=len(stations))
            ground_month = stations.copy()
            ground_month["month"] = month
            ground_month["pm25_ground"] = np.clip(true_at_stations + sensor_noise, 3, None)
            ground_rows.append(ground_month)
        ground_df = pd.concat(ground_rows, ignore_index=True)
        ground_df.to_csv(DATA_RAW_DIR / f"openaq_ground_{city}.csv", index=False)

        print(
            f"    population grid: {len(pop_grid)} cells | "
            f"stations: {len(stations)} | "
            f"satellite cells: {len(sat_grid_coords)} x {len(MONTHS)} months | "
            f"ground obs: {len(ground_df)}"
        )

    print("[generate_synthetic_data] Done. Files written to data/raw/")


if __name__ == "__main__":
    generate_all()
