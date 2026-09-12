"""
STEP 8 — DASHBOARD
====================
What this does
---------------
An interactive Streamlit app that lets a user pick a city and see:
  1. Headline metrics: population-weighted exposure vs. official monitor
     average vs. the gap between them.
  2. An interactive folium map: a PM2.5 heatmap of the corrected grid,
     existing CPCB station markers, and the algorithm's recommended new
     station markers (with population + estimated PM2.5 in the popup).
  3. The model evaluation table (both calibration models' cross-validated
     metrics, and which one was selected).
  4. A downloadable table of the placement recommendations.

This reads directly from outputs/results/*.json and data/processed/*.csv --
it does NOT call the FastAPI service (the API and the dashboard are two
independent consumers of the same pipeline output, which is the point of
writing results to files rather than baking them into either app).

Run it:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"

st.set_page_config(page_title="AQ Sensor Network Audit", layout="wide")


@st.cache_data
def load_json(name: str) -> dict:
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


@st.cache_data
def load_corrected_grid(city: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"corrected_grid_{city}.csv")
    return df.groupby(["lat", "lon"], as_index=False)["pm25_corrected"].mean()


@st.cache_data
def load_stations(city: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"cpcb_stations_{city}.csv")


def build_map(city: str, grid: pd.DataFrame, stations: pd.DataFrame, picks: list) -> folium.Map:
    center = [grid["lat"].mean(), grid["lon"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    # PM2.5 heatmap layer (weighted by concentration)
    heat_data = grid[["lat", "lon", "pm25_corrected"]].values.tolist()
    HeatMap(heat_data, radius=14, blur=18, min_opacity=0.35, name="Corrected PM2.5").add_to(m)

    # Existing stations
    existing_layer = folium.FeatureGroup(name="Existing CPCB stations")
    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            popup=f"Existing station {row['station_id']}",
        ).add_to(existing_layer)
    existing_layer.add_to(m)

    # Recommended new stations
    rec_layer = folium.FeatureGroup(name="Recommended new stations")
    for p in picks:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
            popup=(
                f"<b>Recommended #{p['rank']}</b><br>"
                f"Est. PM2.5: {p['estimated_pm25_ugm3']} µg/m³<br>"
                f"Population in cell: {int(p['population_in_cell']):,}<br>"
                f"Network avg if added: {p['network_avg_after_ugm3']} µg/m³"
            ),
        ).add_to(rec_layer)
    rec_layer.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    st.title("🌫️ Is the Monitor Where the People Are?")
    st.caption("Auditing India's Air-Quality Sensor Network — population-weighted exposure vs. official monitors")

    try:
        gap_data = load_json("exposure_gap.json")
        placement_data = load_json("placement_recommendations.json")
        eval_data = load_json("model_evaluation.json")
    except FileNotFoundError:
        st.error("No results found. Run `python run_pipeline.py` first, then reload this page.")
        st.stop()

    cities = list(gap_data.keys())
    city = st.sidebar.selectbox("City", cities)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Selected calibration model:** `{eval_data['selected_model']}`")
    st.sidebar.markdown(f"Trained on {eval_data['n_training_rows']} station-months across {eval_data['n_stations']} stations.")

    city_gap = gap_data[city]
    city_placement = placement_data[city]

    col1, col2, col3 = st.columns(3)
    col1.metric("Population-weighted exposure", f"{city_gap['population_weighted_exposure_ugm3']} µg/m³")
    col2.metric("Official monitor average", f"{city_gap['official_monitor_average_ugm3']} µg/m³")
    col3.metric(
        "Exposure gap",
        f"{city_gap['exposure_gap_ugm3']:+.1f} µg/m³",
        delta=f"{city_gap['exposure_gap_pct']:+.1f}%",
        delta_color="inverse",
    )
    st.caption(city_gap["interpretation"])

    st.subheader(f"{city}: exposure map, existing stations & recommended new stations")
    grid = load_corrected_grid(city)
    stations = load_stations(city)
    fmap = build_map(city, grid, stations, city_placement["recommended_new_stations"])
    st_folium(fmap, width=None, height=520)

    st.subheader("Recommended new monitor locations")
    rec_df = pd.DataFrame(city_placement["recommended_new_stations"])
    st.dataframe(rec_df, use_container_width=True)
    st.download_button(
        "Download recommendations as CSV",
        rec_df.to_csv(index=False),
        file_name=f"{city}_placement_recommendations.csv",
        mime="text/csv",
    )

    with st.expander("Model evaluation details (both calibration models)"):
        m1 = eval_data["model_1_ridge"]["average"]
        m2 = eval_data["model_2_gbm"]["average"]
        comp_df = pd.DataFrame(
            [{"model": "Model 1 — Ridge regression", **m1}, {"model": "Model 2 — Gradient boosting", **m2}]
        )
        st.dataframe(comp_df, use_container_width=True)
        st.caption(f"Selection rule: {eval_data['selection_rule']}")

    with st.expander("All cities at a glance"):
        summary_df = pd.DataFrame(gap_data).T.reset_index().rename(columns={"index": "city"})
        st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
