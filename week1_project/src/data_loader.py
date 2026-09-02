"""
data_loader.py
================
Week 1 data-loading utilities for the Air Quality Sensor Network Audit project.

DESIGN PRINCIPLE
-----------------
Every loader function below first looks for a REAL data file at the expected
path under `data/raw/`. If it isn't found, it falls back to generating
SYNTHETIC data with the same schema/structure as the real source, so the
rest of the pipeline (Weeks 1-8) can be built and tested end-to-end right
now, and will work unmodified once you drop the real files in.

When you receive the real project data package, just place the files as:
    data/raw/cpcb_caaqms_stations.xlsx
    data/raw/satpm_v6_pm25.nc          (or .tif per city)
    data/raw/worldpop_population.tif   (or per-city .tif files)

and re-run the notebook. The loaders will automatically pick them up
(a console message tells you which mode -- REAL or SYNTHETIC -- was used).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_DIR = os.path.abspath(RAW_DIR)

# ---------------------------------------------------------------------------
# City registry: approximate bounding boxes (lon_min, lat_min, lon_max, lat_max)
# and rough center coordinates for our 4 chosen NCAP cities.
# These bounding boxes are intentionally generous city-extent boxes, not
# official administrative boundaries (Week 1 doesn't require exact wards).
# ---------------------------------------------------------------------------
CITY_REGISTRY = {
    "Delhi": {
        "type": "heavily_monitored_metro",
        "center": (28.6139, 77.2090),
        "bbox": (76.84, 28.40, 77.35, 28.88),   # lon_min, lat_min, lon_max, lat_max
        "n_official_stations_expected": 40,      # Delhi has a dense CAAQMS network
    },
    "Lucknow": {
        "type": "mid_tier_city",
        "center": (26.8467, 80.9462),
        "bbox": (80.80, 26.72, 81.10, 26.98),
        "n_official_stations_expected": 8,
    },
    "Patna": {
        "type": "mid_tier_city",
        "center": (25.5941, 85.1376),
        "bbox": (85.00, 25.50, 85.28, 25.68),
        "n_official_stations_expected": 6,
    },
    "Muzaffarpur": {
        "type": "small_sparse_ncap_city",
        "center": (26.1225, 85.3906),
        "bbox": (85.30, 26.05, 85.48, 26.20),
        "n_official_stations_expected": 1,       # sparsely monitored NCAP city
    },
}

SELECTED_CITIES = ["Delhi", "Lucknow", "Patna", "Muzaffarpur"]


def _report_mode(name, path, is_real):
    mode = "REAL FILE" if is_real else "SYNTHETIC (fallback)"
    loc = path if is_real else "(generated in-memory)"
    print(f"[data_loader] {name}: {mode} -> {loc}")


# ---------------------------------------------------------------------------
# 1. CPCB CAAQMS Station List
# ---------------------------------------------------------------------------
def load_cpcb_stations(cities=None):
    """
    Returns a GeoDataFrame of CPCB CAAQMS stations with columns:
        station_id, station_name, city, agency, latitude, longitude,
        parameters (list-like str), geometry (Point, EPSG:4326)

    Looks for: data/raw/cpcb_caaqms_stations.xlsx
    Expected real-file columns (typical CPCB export) are auto-detected from
    a set of common aliases; adjust COLUMN_ALIASES below if your extract
    differs.
    """
    cities = cities or SELECTED_CITIES
    real_path = os.path.join(RAW_DIR, "cpcb_caaqms_stations.xlsx")

    if os.path.exists(real_path):
        _report_mode("CPCB station list", real_path, True)
        df = pd.read_excel(real_path)

        COLUMN_ALIASES = {
            "station_id": ["station_id", "Station ID", "StationId", "station_code"],
            "station_name": ["station_name", "Station Name", "StationName"],
            "city": ["city", "City", "City/Town/Village/Area"],
            "agency": ["agency", "Agency", "State"],
            "latitude": ["latitude", "Latitude", "lat"],
            "longitude": ["longitude", "Longitude", "long", "lon"],
            "parameters": ["parameters", "Parameters", "Pollutants Monitored"],
        }

        def find_col(aliases):
            for a in aliases:
                if a in df.columns:
                    return a
            return None

        rename_map = {}
        for std_name, aliases in COLUMN_ALIASES.items():
            col = find_col(aliases)
            if col:
                rename_map[col] = std_name
        df = df.rename(columns=rename_map)

        missing = {"station_id", "city", "latitude", "longitude"} - set(df.columns)
        if missing:
            raise ValueError(
                f"CPCB file is missing expected columns after alias mapping: {missing}. "
                f"Available columns: {list(df.columns)}. Update COLUMN_ALIASES in data_loader.py."
            )

        df = df[df["city"].isin(cities)].copy()

    else:
        _report_mode("CPCB station list", real_path, False)
        rng = np.random.default_rng(42)
        rows = []
        sid = 1000
        for city in cities:
            info = CITY_REGISTRY[city]
            lon_min, lat_min, lon_max, lat_max = info["bbox"]
            n_stations = info["n_official_stations_expected"]
            for i in range(n_stations):
                lon = rng.uniform(lon_min, lon_max)
                lat = rng.uniform(lat_min, lat_max)
                rows.append({
                    "station_id": f"SYN-{sid}",
                    "station_name": f"{city} CAAQMS Station {i+1}",
                    "city": city,
                    "agency": "SPCB (synthetic)",
                    "latitude": lat,
                    "longitude": lon,
                    "parameters": "PM2.5, PM10, NO2, SO2, CO, O3",
                })
                sid += 1
        df = pd.DataFrame(rows)

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# 2. SatPM V6 Satellite-Derived PM2.5
# ---------------------------------------------------------------------------
def load_satpm_pm25(city, month="2024-01", resolution_deg=0.01):
    """
    Returns an xarray.DataArray of PM2.5 (ug/m3) on a regular lon/lat grid
    covering the given city's bounding box, with attrs: crs, month.

    Looks for: data/raw/satpm_v6_pm25.nc  (a NetCDF with dims lat/lon/time,
    variable 'pm25', global or India-wide extent -- we'll subset to city bbox)

    Falls back to a synthetic smooth PM2.5 surface with a plausible spatial
    pattern (higher near city center / traffic corridors, seasonal-ish
    magnitude) if the real file isn't present.
    """
    real_path = os.path.join(RAW_DIR, "satpm_v6_pm25.nc")
    lon_min, lat_min, lon_max, lat_max = CITY_REGISTRY[city]["bbox"]

    if os.path.exists(real_path):
        _report_mode(f"SatPM PM2.5 ({city}, {month})", real_path, True)
        ds = xr.open_dataset(real_path)
        da = ds["pm25"].sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )
        if "time" in da.dims:
            da = da.sel(time=month, method="nearest")
        da = da.rio.write_crs("EPSG:4326") if hasattr(da, "rio") else da
        da.attrs["source"] = "SatPM V6 (real)"
        da.attrs["month"] = month
        return da

    _report_mode(f"SatPM PM2.5 ({city}, {month})", real_path, False)
    lons = np.arange(lon_min, lon_max, resolution_deg)
    lats = np.arange(lat_min, lat_max, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    center_lat, center_lon = CITY_REGISTRY[city]["center"]
    dist_from_center = np.sqrt((lon_grid - center_lon) ** 2 + (lat_grid - center_lat) ** 2)

    rng = np.random.default_rng(hash(city + month) % (2**32))
    base_level = {"Delhi": 110, "Lucknow": 90, "Patna": 95, "Muzaffarpur": 80}.get(city, 85)
    pm25 = base_level * np.exp(-dist_from_center * 8) + (base_level * 0.35)
    pm25 += rng.normal(0, 4, size=pm25.shape)
    pm25 = np.clip(pm25, 15, None)

    da = xr.DataArray(
        pm25,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="pm25",
        attrs={
            "units": "ug/m3",
            "source": "SYNTHETIC (SatPM V6 fallback)",
            "month": month,
            "crs": "EPSG:4326",
        },
    )
    return da


# ---------------------------------------------------------------------------
# 3. WorldPop 100m Population Grid
# ---------------------------------------------------------------------------
def load_worldpop_population(city, resolution_deg=0.001):
    """
    Returns an xarray.DataArray of population count per grid cell (~100m
    when resolution_deg ~= 0.001 deg near India's latitude) covering the
    city bbox.

    Looks for: data/raw/worldpop_population.tif (India-wide GeoTIFF; will
    be windowed/read for the city bbox using rasterio).

    Falls back to a synthetic population surface: dense core, decaying
    outward, with a couple of secondary population clusters, roughly
    matching real Indian city population totals in order of magnitude.
    """
    real_path = os.path.join(RAW_DIR, "worldpop_population.tif")
    lon_min, lat_min, lon_max, lat_max = CITY_REGISTRY[city]["bbox"]

    if os.path.exists(real_path):
        _report_mode(f"WorldPop population ({city})", real_path, True)
        with rasterio.open(real_path) as src:
            window = rasterio.windows.from_bounds(
                lon_min, lat_min, lon_max, lat_max, transform=src.transform
            )
            arr = src.read(1, window=window)
            win_transform = src.window_transform(window)
        nrows, ncols = arr.shape
        lons = win_transform.c + win_transform.a * (np.arange(ncols) + 0.5)
        lats = win_transform.f + win_transform.e * (np.arange(nrows) + 0.5)
        da = xr.DataArray(
            arr, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"], name="population",
            attrs={"source": "WorldPop (real)", "crs": "EPSG:4326"},
        )
        return da

    _report_mode(f"WorldPop population ({city})", real_path, False)
    lons = np.arange(lon_min, lon_max, resolution_deg)
    lats = np.arange(lat_min, lat_max, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    center_lat, center_lon = CITY_REGISTRY[city]["center"]
    rng = np.random.default_rng(hash(city + "pop") % (2**32))

    dist_core = np.sqrt((lon_grid - center_lon) ** 2 + (lat_grid - center_lat) ** 2)
    pop_core = np.exp(-dist_core * 25)

    # one or two secondary clusters (satellite towns / suburbs)
    n_secondary = 2
    pop_secondary = np.zeros_like(pop_core)
    for _ in range(n_secondary):
        sub_lon = rng.uniform(lon_min, lon_max)
        sub_lat = rng.uniform(lat_min, lat_max)
        d = np.sqrt((lon_grid - sub_lon) ** 2 + (lat_grid - sub_lat) ** 2)
        pop_secondary += 0.4 * np.exp(-d * 40)

    pop_density = pop_core + pop_secondary
    pop_density = pop_density / pop_density.sum()

    city_total_pop = {
        "Delhi": 16_800_000, "Lucknow": 3_400_000,
        "Patna": 2_000_000, "Muzaffarpur": 400_000,
    }.get(city, 1_000_000)

    pop_grid = pop_density * city_total_pop
    pop_grid += rng.normal(0, pop_grid.std() * 0.05, size=pop_grid.shape)
    pop_grid = np.clip(pop_grid, 0, None)

    da = xr.DataArray(
        pop_grid,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="population",
        attrs={"source": "SYNTHETIC (WorldPop fallback)", "crs": "EPSG:4326",
               "note": f"scaled to approx total pop {city_total_pop:,}"},
    )
    return da


# ---------------------------------------------------------------------------
# 4. City boundary polygons (simple bbox-derived, for Week 1 exploration)
# ---------------------------------------------------------------------------
def get_city_boundary(city):
    """Returns a GeoDataFrame with a single bbox polygon for the city."""
    lon_min, lat_min, lon_max, lat_max = CITY_REGISTRY[city]["bbox"]
    geom = box(lon_min, lat_min, lon_max, lat_max)
    gdf = gpd.GeoDataFrame({"city": [city]}, geometry=[geom], crs="EPSG:4326")
    return gdf


def get_all_city_boundaries(cities=None):
    cities = cities or SELECTED_CITIES
    parts = [get_city_boundary(c) for c in cities]
    return pd.concat(parts, ignore_index=True)
