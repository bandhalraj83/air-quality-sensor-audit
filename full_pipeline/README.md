# Is the Monitor Where the People Are?
### Auditing India's Air-Quality Sensor Network — full pipeline

This is a complete, runnable implementation of the 8-week project: it estimates
population-weighted PM2.5 exposure from satellite data, compares it against what
a city's official monitors report, and recommends where a new monitor would most
reduce that gap — with a bias-correction ML pipeline, a FastAPI endpoint, and a
Streamlit dashboard on top.

**No real data attached?** No problem — `src/generate_synthetic_data.py` builds
synthetic CPCB/SatPM/WorldPop/OpenAQ-shaped files so the whole pipeline runs today.
Drop the real project data package into `data/raw/` using the same filenames
(see `config.py` docstring) and run with `--skip-generate` — nothing else changes.

---

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## 2. Run the full pipeline

```bash
python run_pipeline.py
```

This runs every step below in order and prints a summary. Takes a few seconds
on the synthetic data.

Re-run anytime with `python run_pipeline.py --skip-generate` once you've
swapped in real data (skips regenerating synthetic files).

## 3. Serve the results

```bash
uvicorn api.main:app --reload --port 8000
# then: curl http://localhost:8000/exposure-gap/Delhi
# interactive docs at http://localhost:8000/docs
```

## 4. Explore the dashboard

```bash
streamlit run dashboard/app.py
```

## 5. Run the smoke tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Project structure

```
config.py                        City definitions, paths, constants (edit this to add cities)
run_pipeline.py                  Orchestrates every step end-to-end

src/
  generate_synthetic_data.py     STEP 1  Data acquisition (synthetic stand-in for the real extract)
  align_and_engineer.py          STEP 2  Align satellite grid to population grid; build training table
  calibration_models.py          STEP 3-4 Model 1 (Ridge) and Model 2 (Gradient Boosting) definitions
  evaluate_models.py             STEP 5  Spatial cross-validation, metrics, model selection
  exposure_gap.py                STEP 6a Apply best model; population-weighted exposure gap per city
  placement.py                   STEP 6b Greedy algorithm recommending new monitor locations
  visualize.py                   Static PNG maps (population/exposure + stations) per city

api/
  main.py                        STEP 7  FastAPI endpoint serving exposure gap & placement recommendations

dashboard/
  app.py                         STEP 8  Streamlit dashboard: interactive maps + tables

tests/
  test_pipeline.py               Smoke tests validating pipeline invariants (no NaNs, physical PM2.5, etc.)

data/raw/                        Input data (synthetic, or your real CPCB/SatPM/WorldPop/OpenAQ extract)
data/processed/                  Aligned grids, training tables, saved model (.joblib)
outputs/results/                 exposure_gap.json, placement_recommendations.json, model_evaluation.json
outputs/plots/                   Per-city PNG maps
```

## How each step maps to the 8-week plan

| Week | Plan item | Code |
|---|---|---|
| 1 | Select cities; load CPCB/SatPM/WorldPop; explore formats | `config.py` (cities), `src/generate_synthetic_data.py` |
| 2 | Align satellite grid with population grid; engineer exposure features | `src/align_and_engineer.py` |
| 3 | Model 1 — bias-correction regression | `src/calibration_models.py::build_model_1` |
| 4 | Model 2 — alternate calibration (gradient boosting) | `src/calibration_models.py::build_model_2` |
| 5 | Evaluate both models; select the better one | `src/evaluate_models.py` |
| 6 | Tune model; compute exposure gap; greedy placement algorithm | `src/evaluate_models.py` (tuning), `src/exposure_gap.py`, `src/placement.py` |
| 7 | FastAPI endpoint | `api/main.py` |
| 8 | Dashboard; finalize | `dashboard/app.py` |

## Key design choices worth knowing about (and revisiting with real data)

- **Spatial cross-validation, not random K-fold.** `evaluate_models.py` groups
  folds by `station_id` (`GroupKFold`) so the model is always tested on
  stations it never trained on — a fair proxy for generalizing to a brand-new
  location. Random K-fold would leak information between months of the same
  station and overstate accuracy.
- **Model selection rule:** lowest cross-validated RMSE wins (ties broken by
  bias). On the synthetic data the simple Ridge regression beats gradient
  boosting, because the injected satellite bias is close to linear — a useful
  reminder that a fancier model isn't automatically better; with real data,
  re-run and let the numbers decide.
- **Hyperparameter tuning (Week 6).** Once a model family is selected,
  `evaluate_models.py` grid-searches a small set of that family's
  hyperparameters (Ridge's `alpha`; GBM's `learning_rate`/`max_depth`) using
  the same GroupKFold methodology, and refits the best-performing
  configuration. On the synthetic data this finds `alpha=0.1` marginally
  beats the default `alpha=1.0`, and shows RMSE degrading sharply past
  `alpha=30` — a genuine (if small) search, not a fixed guess.
- **Predictions are clipped to a physical floor (`MIN_PHYSICAL_PM25 = 1.0`)** in
  `calibration_models.py::predict`. A linear model can extrapolate to
  non-physical negative PM2.5 at the edges of a city's grid; this is caught by
  `tests/test_pipeline.py::test_corrected_grid_pm25_is_nonnegative`.
- **Placement scoring = `population × |local PM2.5 − current network average|`.**
  This deliberately prioritizes densely populated areas whose air looks
  nothing like what the existing network reports, rather than picking
  whichever single cell happens to nudge a summary statistic into place. See
  the long comment at the top of `src/placement.py` for the reasoning and an
  earlier (rejected) alternative.
- **Greedy, not globally optimal.** The placement algorithm is a transparent,
  auditable heuristic — pick the best site, lock it in, repeat — as called for
  in the brief. A natural v2 (noted in the report template's Limitations
  section) is a joint optimization (e.g. integer programming) over all new
  sites at once, plus real siting constraints (land access, power, security).

## Extending to real data

1. Save the real extract's files into `data/raw/` using the filenames in
   `config.py`'s module docstring.
2. If your real cities differ from the four defaults, edit `CITIES` in
   `config.py` (bounding box; everything else derives from the data itself
   for real sources — the synthetic-only fields like `pop_centers` and
   `station_center` are only used by the synthetic generator).
3. Run `python run_pipeline.py --skip-generate`.
4. Everything downstream (models, gap analysis, placement, API, dashboard)
   runs unchanged.
