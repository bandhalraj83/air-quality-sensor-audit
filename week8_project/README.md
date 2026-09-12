# Week 8 — Dashboard & Finalization

**Plan deliverable:** Build a dashboard with city maps showing exposure versus monitor coverage; finalize code and report.

## Where this lives
- Dashboard: [`../full_pipeline/dashboard/app.py`](../full_pipeline/dashboard/app.py)
- Static maps: [`../full_pipeline/src/visualize.py`](../full_pipeline/src/visualize.py)
- Tests: [`../full_pipeline/tests/test_pipeline.py`](../full_pipeline/tests/test_pipeline.py) (9 passing smoke tests)

## What the dashboard shows
- Headline metrics (population-weighted exposure, official average, gap) per selected city.
- An interactive folium map: corrected PM2.5 heatmap, existing CPCB stations, and recommended new stations with popups.
- Model evaluation and placement-recommendation tables, with CSV download.

## Run it
```bash
cd ../full_pipeline
streamlit run dashboard/app.py
```

## Finalized artifacts
See [`../full_pipeline/README.md`](../full_pipeline/README.md) for the full pipeline documentation, and `full_pipeline/outputs/` for committed sample results and maps.
