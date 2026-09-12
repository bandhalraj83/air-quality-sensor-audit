"""
VISUALIZATION HELPERS
=======================
What this does
---------------
Produces one static PNG per city (outputs/plots/<city>_exposure_map.png)
showing:
  - a population-weighted PM2.5 heatmap (the corrected grid, annual mean)
  - existing CPCB station locations (white circles)
  - the algorithm's recommended new station locations (red stars)

These are quick, dependency-light (matplotlib only) sanity-check maps
generated as part of the pipeline run. The interactive Streamlit dashboard
(dashboard/app.py) renders a richer, zoomable version of the same
information using folium.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CITIES, DATA_PROCESSED_DIR, DATA_RAW_DIR, PLOTS_DIR, RESULTS_DIR


def plot_city(city: str) -> str:
    corrected = pd.read_csv(DATA_PROCESSED_DIR / f"corrected_grid_{city}.csv")
    annual = corrected.groupby(["lat", "lon"], as_index=False)["pm25_corrected"].mean()

    stations = pd.read_csv(DATA_RAW_DIR / f"cpcb_stations_{city}.csv")

    with open(RESULTS_DIR / "placement_recommendations.json") as f:
        placement = json.load(f)[city]["recommended_new_stations"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        annual["lon"], annual["lat"], c=annual["pm25_corrected"],
        cmap="YlOrRd", s=14, marker="s", alpha=0.9,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Bias-corrected annual PM2.5 (µg/m³)")

    ax.scatter(
        stations["lon"], stations["lat"], facecolors="white", edgecolors="black",
        s=70, marker="o", label="Existing CPCB station", zorder=5,
    )

    if placement:
        rec_lons = [p["lon"] for p in placement]
        rec_lats = [p["lat"] for p in placement]
        ax.scatter(
            rec_lons, rec_lats, c="red", marker="*", s=260,
            edgecolors="black", linewidths=0.6, label="Recommended new station", zorder=6,
        )
        for p in placement:
            ax.annotate(f"#{p['rank']}", (p["lon"], p["lat"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, fontweight="bold")

    ax.set_title(f"{city}: Population-Weighted PM2.5 Exposure & Monitor Coverage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    out_path = PLOTS_DIR / f"{city}_exposure_map.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def run() -> None:
    for city in CITIES:
        path = plot_city(city)
        print(f"[visualize] Saved {path}")


if __name__ == "__main__":
    run()
