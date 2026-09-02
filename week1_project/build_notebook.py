import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------
md("""# Week 1 — Setup, Data Loading & Exploration
### Air Quality Sensor Network Audit — Population-Weighted PM2.5 Exposure vs. Official Monitoring

**Goal:** Select cities, load the three core datasets (CPCB stations, SatPM PM2.5, WorldPop population),
align coordinate systems, and produce initial exploratory maps.

**Note on data:** This notebook is written to work identically whether you have the real project data
package or not. `src/data_loader.py` looks for real files under `data/raw/`; if they're absent, it
generates realistic **synthetic** stand-ins with the same schema, clearly labelled in every printout
and plot title. Once the real data package arrives, drop the files in `data/raw/` (see README in that
folder) and re-run — no code changes needed.
""")

code("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap

from data_loader import (
    load_cpcb_stations,
    load_satpm_pm25,
    load_worldpop_population,
    get_city_boundary,
    get_all_city_boundaries,
    CITY_REGISTRY,
    SELECTED_CITIES,
)

pd.set_option('display.max_columns', 50)
plt.rcParams['figure.dpi'] = 100
print("Setup complete.")
""")

# ---------------------------------------------------------------------
md("""## 1. Select NCAP Cities

We pick 4 cities spanning the monitoring-density spectrum required by the problem statement:

| City | Type | Rationale |
|---|---|---|
| **Delhi** | Heavily monitored metro | Dense CAAQMS network (~40 stations) — good baseline for calibration (Weeks 3-5) |
| **Lucknow** | Mid-tier city | Moderate station count, large population — good NCAP funding-relevance case |
| **Patna** | Mid-tier city | Similar tier to Lucknow, different geography/pollution profile — cross-city robustness check |
| **Muzaffarpur** | Small, sparsely monitored NCAP city | Only ~1 official station — the case where the exposure-gap / placement recommendation matters most |

This mix directly supports the project's core question: does monitor placement correlate with where
people actually live and breathe polluted air, or with something else (funding incentives, ease of
access, legacy siting)?
""")

code("""for city, info in CITY_REGISTRY.items():
    marker = " <-- SELECTED" if city in SELECTED_CITIES else ""
    print(f"{city:15s} | type: {info['type']:25s} | expected official stations: "
          f"{info['n_official_stations_expected']:>3d}{marker}")
""")

# ---------------------------------------------------------------------
md("""## 2. Load CPCB CAAQMS Station List

Extract `station_id`, `station_name`, `city`, `latitude`, `longitude`, and `parameters` monitored
for each selected city. Returned as a `GeoDataFrame` (CRS: EPSG:4326 / WGS84) so it's immediately
mappable and joinable with the raster grids later.
""")

code("""stations_gdf = load_cpcb_stations(cities=SELECTED_CITIES)

print(f"Total stations loaded: {len(stations_gdf)}")
print(f"CRS: {stations_gdf.crs}")
print()
display(stations_gdf.groupby('city').size().rename('n_stations').to_frame())
""")

code("""stations_gdf[['station_id', 'station_name', 'city', 'latitude', 'longitude', 'parameters']].head(10)
""")

md("""**Observation to carry into Week 6:** the huge disparity in station counts (Delhi: ~40 vs.
Muzaffarpur: ~1) is exactly the pattern the project investigates — is that disparity justified by
population and pollution levels, or does it reflect something else?
""")

# ---------------------------------------------------------------------
md("""## 3. Load SatPM V6 Satellite-Derived PM2.5

Monthly gridded PM2.5 surface for each city's bounding box. Returned as an `xarray.DataArray`
(dims: `lat`, `lon`) so it composes naturally with the population grid in Week 2's zonal-statistics step.
""")

code("""SELECTED_MONTH = "2024-01"  # adjust once real SatPM monthly coverage is confirmed

pm25_by_city = {}
for city in SELECTED_CITIES:
    da = load_satpm_pm25(city, month=SELECTED_MONTH)
    pm25_by_city[city] = da
    print(f"{city:15s} | grid shape: {str(da.shape):12s} | "
          f"PM2.5 range: {float(da.min()):6.1f} - {float(da.max()):6.1f} ug/m3 | "
          f"mean: {float(da.mean()):6.1f} ug/m3 | source: {da.attrs['source']}")
""")

# ---------------------------------------------------------------------
md("""## 4. Load WorldPop 100m Population Grid

Gridded population counts per cell, same city bounding boxes. This will be combined with the PM2.5
grid in Week 2 to compute population-weighted exposure.
""")

code("""pop_by_city = {}
for city in SELECTED_CITIES:
    da = load_worldpop_population(city)
    pop_by_city[city] = da
    print(f"{city:15s} | grid shape: {str(da.shape):14s} | "
          f"total population: {float(da.sum()):>12,.0f} | source: {da.attrs['source']}")
""")

md("""**Note the resolution mismatch:** SatPM (~0.01°, roughly 1km) and WorldPop (~0.001°, roughly 100m)
grids are on different resolutions and different pixel grids. Week 2's first task is reconciling this
(resampling one to match the other) before the population-weighting can be computed correctly — we
deliberately leave them unaligned here so that step is visible and explicit rather than hidden.
""")

# ---------------------------------------------------------------------
md("""## 5. Coordinate Reference System (CRS) Check

All three datasets should share a common CRS before any spatial join or area-based calculation.
We standardize on **EPSG:4326 (WGS84, lat/lon)** for Week 1 exploration and mapping. For Week 2's
area-accurate zonal statistics, we'll additionally reproject to a local **UTM zone** (UTM 44N,
EPSG:32644, covers all 4 cities) since EPSG:4326 degrees aren't equal-area and would bias
population-weighting near the grid edges.
""")

code("""print("=== CRS audit ===")
print(f"CPCB stations (GeoDataFrame):  {stations_gdf.crs}")

for city in SELECTED_CITIES:
    pm25_crs = pm25_by_city[city].attrs.get('crs', 'not set')
    pop_crs = pop_by_city[city].attrs.get('crs', 'not set')
    print(f"{city:15s} | SatPM CRS: {pm25_crs:12s} | WorldPop CRS: {pop_crs:12s}")

print()
print("Target analysis CRS for Week 2 area-accurate calculations: EPSG:32644 (UTM Zone 44N)")

# Quick check: does UTM 44N reasonably cover all 4 cities without excessive distortion?
utm_test = stations_gdf.to_crs("EPSG:32644")
print(f"\\nStations reprojected to UTM 44N successfully. Sample coordinates (meters):")
print(utm_test.geometry.head(3))
""")

# ---------------------------------------------------------------------
md("""## 6. City Boundaries

Simple bounding-box polygons per city for Week 1 (sufficient for clipping/exploration). Ward-level
boundaries — needed for the finer-grained Week 6 placement analysis — should be sourced separately
(Census India shapefiles or city GIS portals); flagged here as a Week 2/6 follow-up.
""")

code("""city_boundaries = get_all_city_boundaries(SELECTED_CITIES)
city_boundaries['area_km2'] = city_boundaries.to_crs("EPSG:32644").geometry.area / 1e6
city_boundaries[['city', 'area_km2']]
""")

# ---------------------------------------------------------------------
md("""## 7. Exploratory Visualization

### 7a. Station locations on an interactive map (folium)
""")

code("""def make_station_map(city, stations_gdf, city_boundaries):
    city_stations = stations_gdf[stations_gdf['city'] == city]
    center = CITY_REGISTRY[city]['center']

    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    # city boundary
    boundary = city_boundaries[city_boundaries['city'] == city]
    folium.GeoJson(
        boundary.geometry.iloc[0],
        style_function=lambda x: {'fillOpacity': 0, 'color': 'gray', 'weight': 1, 'dashArray': '4,4'}
    ).add_to(m)

    # stations
    for _, row in city_stations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='crimson',
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['station_name']}<br>ID: {row['station_id']}<br>Params: {row['parameters']}",
        ).add_to(m)

    return m

# Example: Delhi station map (swap city name to view others)
delhi_map = make_station_map('Delhi', stations_gdf, city_boundaries)
delhi_map
""")

code("""# Sparse-monitoring case for contrast
muzaffarpur_map = make_station_map('Muzaffarpur', stations_gdf, city_boundaries)
muzaffarpur_map
""")

md("""### 7b. Raw satellite PM2.5 heatmap (per city)""")

code("""fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, city in zip(axes, SELECTED_CITIES):
    da = pm25_by_city[city]
    im = da.plot(ax=ax, cmap='YlOrRd', add_colorbar=True,
                 cbar_kwargs={'label': 'PM2.5 (ug/m3)'})

    city_stations = stations_gdf[stations_gdf['city'] == city]
    ax.scatter(city_stations['longitude'], city_stations['latitude'],
               c='blue', s=25, marker='^', edgecolor='white', linewidth=0.5,
               label='CAAQMS station', zorder=5)

    source_tag = "(synthetic)" if "SYNTHETIC" in da.attrs['source'] else "(real)"
    ax.set_title(f"{city} — satellite PM2.5 {source_tag}\\nmean={float(da.mean()):.1f} ug/m3")
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('../data/processed/week1_pm25_heatmaps.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: data/processed/week1_pm25_heatmaps.png")
""")

md("""### 7c. Population density (per city)""")

code("""fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, city in zip(axes, SELECTED_CITIES):
    da = pop_by_city[city]
    da.plot(ax=ax, cmap='viridis', add_colorbar=True,
            cbar_kwargs={'label': 'Population per cell'})

    city_stations = stations_gdf[stations_gdf['city'] == city]
    ax.scatter(city_stations['longitude'], city_stations['latitude'],
               c='red', s=25, marker='^', edgecolor='white', linewidth=0.5,
               label='CAAQMS station', zorder=5)

    source_tag = "(synthetic)" if "SYNTHETIC" in da.attrs['source'] else "(real)"
    ax.set_title(f"{city} — population density {source_tag}\\ntotal pop={float(da.sum()):,.0f}")
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('../data/processed/week1_population_heatmaps.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: data/processed/week1_population_heatmaps.png")
""")

md("""### 7d. Side-by-side: pollution vs. population vs. monitor coverage (Delhi example)

This is a preview of the exact comparison the whole project builds toward — visually, do monitors
sit where pollution AND population are both high?
""")

code("""city = "Delhi"
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pm25_by_city[city].plot(ax=axes[0], cmap='YlOrRd', add_colorbar=True, cbar_kwargs={'label': 'PM2.5 (ug/m3)'})
pop_by_city[city].plot(ax=axes[1], cmap='viridis', add_colorbar=True, cbar_kwargs={'label': 'Population'})

for ax, title in zip(axes, ['PM2.5 exposure', 'Population density']):
    city_stations = stations_gdf[stations_gdf['city'] == city]
    ax.scatter(city_stations['longitude'], city_stations['latitude'],
               c='blue' if title.startswith('PM') else 'red', s=30, marker='^',
               edgecolor='white', linewidth=0.6, label='CAAQMS station', zorder=5)
    ax.set_title(f"{city}: {title}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------
md("""## 8. Week 1 Summary & Handoff to Week 2

**What we have:**
- 4 NCAP cities selected, spanning heavily-monitored to sparsely-monitored
- CPCB station GeoDataFrame (id, name, city, lat/lon, parameters, geometry) — EPSG:4326
- Per-city SatPM PM2.5 grids (xarray DataArray, ~0.01° resolution) — EPSG:4326
- Per-city WorldPop population grids (xarray DataArray, ~0.001° resolution) — EPSG:4326
- City bounding-box boundaries + area calculations (via UTM 44N reprojection)
- Exploratory maps: interactive station maps, PM2.5 heatmaps, population heatmaps

**Flagged for Week 2:**
1. **Resolution mismatch** between SatPM (~1km) and WorldPop (~100m) grids must be resolved
   (resample one to the other's grid) before population-weighted exposure can be computed.
2. **CRS**: reproject both grids to EPSG:32644 (UTM 44N) for area-accurate zonal statistics.
3. Consider sourcing **ward-level boundaries** now if finer granularity is wanted for Week 6.

**Data status:** {}
""".format(
    "All datasets are currently **SYNTHETIC** fallbacks generated by `src/data_loader.py`. "
    "Replace with real files in `data/raw/` (see module docstring for expected filenames/schemas) "
    "and re-run — no other code changes required."
))

code("""# Persist Week 1 outputs for Week 2 to consume
import pickle

week1_outputs = {
    'stations_gdf': stations_gdf,
    'pm25_by_city': pm25_by_city,
    'pop_by_city': pop_by_city,
    'city_boundaries': city_boundaries,
    'selected_cities': SELECTED_CITIES,
    'month': SELECTED_MONTH,
}

with open('../data/processed/week1_outputs.pkl', 'wb') as f:
    pickle.dump(week1_outputs, f)

print("Week 1 outputs saved to data/processed/week1_outputs.pkl")
print("Keys:", list(week1_outputs.keys()))
""")

nb['cells'] = cells

with open('notebooks/week1_setup_data_loading.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
