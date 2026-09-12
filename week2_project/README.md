# Week 2 — Alignment & Feature Engineering

**Plan deliverable:** Preprocess and align the satellite PM2.5 grid with the population grid; engineer population-weighted exposure features per city.

## Where this lives
Implemented in [`../full_pipeline/src/align_and_engineer.py`](../full_pipeline/src/align_and_engineer.py), as part of the integrated `full_pipeline/` codebase (see its [README](../full_pipeline/README.md) for why the pipeline is one continuous codebase rather than 8 separate per-week modules — later weeks build directly on the config/data conventions established in Week 1/2, so splitting them apart would mean duplicating most of the pipeline into every folder).

> **Note on cities/data format:** `full_pipeline/` uses a different city set (Delhi, Mumbai, Bengaluru, Chennai) and a simpler CSV-only data format than `week1_project/`'s real-format loader (Delhi, Lucknow, Patna, Muzaffarpur; `.xlsx`/`.nc`/`.tif`). See the [root README](../README.md#a-note-on-week1_project-vs-full_pipeline) for why, and how to reconcile them if you want one continuous city set end-to-end.

## What it does
- Regrids the satellite PM2.5 surface (coarser native resolution) onto the population grid (finer resolution) using bilinear interpolation (`scipy.interpolate.RegularGridInterpolator`).
- Extracts the satellite estimate at each CPCB station's exact coordinates and joins it against the OpenAQ ground-truth reading for that station/month, building the supervised training table used from Week 3 onward.

## Reproduce this step
```bash
cd ../full_pipeline
python run_pipeline.py           # runs Week 1 (data gen) + this step + everything downstream
# or, to run just this step against already-generated raw data:
python -c "from src import align_and_engineer; align_and_engineer.run()"
```
Outputs: `full_pipeline/data/processed/aligned_grid_<city>.csv`, `training_table_<city>.csv`.
