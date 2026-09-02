# Air Quality Sensor Network Audit — Week 1 Deliverable

## Structure
```
week1_project/
├── notebooks/
│   └── week1_setup_data_loading.ipynb   <- Main deliverable, already executed
├── src/
│   └── data_loader.py                    <- All data-loading logic (real + synthetic fallback)
├── data/
│   ├── raw/                              <- PUT REAL DATA FILES HERE (currently empty)
│   └── processed/                        <- Saved outputs (maps, pickled Week 1 results)
└── build_notebook.py                     <- Script that generates the notebook (for reference/editing)
```

## How this works
Since the real dataset package (CPCB station list, SatPM V6, WorldPop) wasn't available yet,
`src/data_loader.py` was built so the ENTIRE Week 1 pipeline runs today using realistic
synthetic data with the same schema as the real sources, clearly labeled everywhere
("SYNTHETIC (fallback)" vs "REAL FILE" in console output and plot titles).

## To plug in real data later
Drop these files into `data/raw/` with these exact names:
- `cpcb_caaqms_stations.xlsx` — CPCB CAAQMS station list
- `satpm_v6_pm25.nc` — SatPM V6 PM2.5 NetCDF (lat/lon/time dims, variable `pm25`)
- `worldpop_population.tif` — WorldPop population GeoTIFF (India-wide or per-city)

Then just re-run the notebook top to bottom — no code changes needed. The loader
auto-detects the files and switches from synthetic to real data automatically.

If your real file's column names or structure differ from what's expected, see the
`COLUMN_ALIASES` dict in `load_cpcb_stations()` (data_loader.py) — add your column
names there.

## Cities selected
Delhi (heavily monitored), Lucknow & Patna (mid-tier), Muzaffarpur (sparse, 1 station) —
chosen to span the monitoring-density spectrum central to the project's question.

## Next: Week 2
Resample SatPM and WorldPop grids to a common resolution/CRS (UTM 44N) and compute
population-weighted PM2.5 exposure per city. `data/processed/week1_outputs.pkl`
contains everything Week 2 needs (stations, PM2.5 grids, population grids, boundaries).
