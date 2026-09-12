"""
config.py
=========
Single source of truth for paths, city definitions, and pipeline constants.
Every other module imports from here instead of hard-coding values, so the
whole project can be re-pointed at real data (or new cities) by editing only
this file.

WHERE THE REAL DATA GOES
-------------------------
This project ships with a synthetic data generator (src/generate_synthetic_data.py)
that produces files with the SAME shape/columns as the real sources described in
the problem statement. Once you have the real project data package, drop the
files into data/raw/ using the same filenames used here and skip step 1
(generation) — steps 2 onward read from data/raw/ and don't care where the
files came from.

    data/raw/cpcb_stations_<city>.csv   <- CPCB CAAQMS Station List (per city)
    data/raw/satpm_grid_<city>.csv      <- SatPM V6 satellite PM2.5 (per city, per month)
    data/raw/population_grid_<city>.csv <- WorldPop population grid (per city)
    data/raw/openaq_ground_<city>.csv   <- OpenAQ ground-truth readings (per city, per month)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = DATA_PROCESSED_DIR / "models"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study cities
# ---------------------------------------------------------------------------
# Bounding boxes are approximate city-proper extents (degrees, WGS84).
# `station_clusters` describes WHERE official CPCB monitors tend to sit today
# (e.g. clustered near central/government areas) -- this non-random placement
# is the root of the "gap" the project measures.
CITIES = {
    "Delhi": {
        "bbox": {"lat_min": 28.40, "lat_max": 28.90, "lon_min": 76.85, "lon_max": 77.35},
        "n_official_stations": 12,
        "station_center": (28.63, 77.22),   # official stations cluster near here
        "station_spread": 0.08,
        "pop_centers": [                     # (lat, lon, relative_weight) - multiple urban cores
            (28.65, 77.23, 1.0),
            (28.70, 76.95, 0.6),
            (28.50, 77.30, 0.5),
            (28.45, 77.05, 0.4),
        ],
        "total_population": 11_000_000,
    },
    "Mumbai": {
        "bbox": {"lat_min": 18.90, "lat_max": 19.30, "lon_min": 72.75, "lon_max": 72.98},
        "n_official_stations": 10,
        "station_center": (19.05, 72.87),
        "station_spread": 0.05,
        "pop_centers": [
            (19.08, 72.88, 1.0),
            (19.20, 72.85, 0.7),
            (18.95, 72.83, 0.5),
        ],
        "total_population": 9_800_000,
    },
    "Bengaluru": {
        "bbox": {"lat_min": 12.85, "lat_max": 13.15, "lon_min": 77.45, "lon_max": 77.75},
        "n_official_stations": 8,
        "station_center": (12.97, 77.59),
        "station_spread": 0.06,
        "pop_centers": [
            (12.97, 77.59, 1.0),
            (13.05, 77.65, 0.6),
            (12.90, 77.60, 0.4),
        ],
        "total_population": 6_500_000,
    },
    "Chennai": {
        "bbox": {"lat_min": 12.90, "lat_max": 13.20, "lon_min": 80.10, "lon_max": 80.35},
        "n_official_stations": 7,
        "station_center": (13.06, 80.24),
        "station_spread": 0.05,
        "pop_centers": [
            (13.06, 80.24, 1.0),
            (12.95, 80.18, 0.5),
            (13.15, 80.25, 0.4),
        ],
        "total_population": 5_200_000,
    },
}

CITY_LIST = list(CITIES.keys())

# ---------------------------------------------------------------------------
# Grid / raster constants
# ---------------------------------------------------------------------------
POP_GRID_RESOLUTION_DEG = 0.01     # ~1.1 km -- stand-in for WorldPop 100m (coarsened for a runnable demo)
SATPM_NATIVE_RESOLUTION_DEG = 0.05  # ~5.5 km -- coarser native satellite resolution, must be aligned to pop grid

MONTHS = list(range(1, 13))  # 1..12, one year of monthly SatPM + OpenAQ data

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Sensor placement
# ---------------------------------------------------------------------------
N_RECOMMENDED_STATIONS = 3          # how many new monitors to recommend per city
MIN_SEPARATION_DEG = 0.03           # don't recommend two new stations closer than this to each other
