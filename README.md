# Air Quality Sensor Network Audit

**Is the Monitor Where the People Are?** — Auditing India's air-quality sensor network by comparing satellite-derived, population-weighted PM2.5 exposure against what official monitors report, and recommending where new monitors would close the gap.

## Repository layout

```
├── PGCP AIML Capstone - Grp-3-Digital_Twins_v1.0.docx   Capstone group document
├── Problem Statement - Air Quality Sensor Network Audit.docx
├── week1_project/     Week 1 deliverable (notebook-based, real-format loader)
├── week2_project/  →  pointers into full_pipeline/ for each week's deliverable
├── week3_project/         "
├── week4_project/         "
├── week5_project/         "
├── week6_project/         "
├── week7_project/         "
├── week8_project/         "
└── full_pipeline/     Complete, integrated, end-to-end implementation (Weeks 1–8)
```

`week1_project/` is the original Week 1 notebook-based deliverable. `full_pipeline/` is a complete, tested, end-to-end implementation covering every week of the plan (data alignment → two bias-correction models → cross-validated selection → hyperparameter tuning → population-weighted exposure gap → greedy sensor placement → FastAPI service → Streamlit dashboard), built as one integrated codebase rather than 8 separate silos, since later weeks depend on shared state (config, aligned data, trained model) from earlier ones. `week2_project` through `week8_project` are short pointer READMEs mapping each week's plan item to the exact file in `full_pipeline/` that implements it, plus the command to reproduce it and that week's actual results.

Start with **[`full_pipeline/README.md`](full_pipeline/README.md)** for full setup, architecture, and how-to-run instructions.

## A note on week1_project vs. full_pipeline

These two were built independently and **do not currently share a data pipeline**:

| | `week1_project` | `full_pipeline` |
|---|---|---|
| Cities | Delhi, Lucknow, Patna, Muzaffarpur | Delhi, Mumbai, Bengaluru, Chennai |
| Real-data formats | `.xlsx`, `.nc` (NetCDF), `.tif` (GeoTIFF) | `.csv` |
| Geospatial stack | geopandas, rasterio, xarray | numpy, pandas, scipy |

`full_pipeline/` was built as a complete, self-contained solution covering the entire 8-week plan, and does not currently continue from `week1_project`'s specific city choices or its real-data loader. **If a single, continuous pipeline (one city set, one data format) is needed for final submission, `full_pipeline/`'s calibration/exposure-gap/placement/API/dashboard logic would need to be pointed at `week1_project`'s data loader and city list** — the modeling logic itself (Ridge/GBM calibration, population-weighting, greedy placement) is independent of which cities or file formats feed it, so this is a data-adapter change, not a redesign.

## Quick start (full_pipeline)

```bash
cd full_pipeline
pip install -r requirements.txt
python run_pipeline.py                              # full run, ~3 seconds
uvicorn api.main:app --reload --port 8000            # serve results via API
streamlit run dashboard/app.py                       # interactive dashboard
PYTHONPATH=. pytest tests/ -v                        # 9 smoke tests
```
