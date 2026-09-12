"""
STEP 7 — FASTAPI ENDPOINT
===========================
What this does
---------------
Serves the pipeline's results (outputs/results/*.json, written by
run_pipeline.py) over HTTP so any client -- the dashboard, a city
administrator's tool, a notebook -- can query a city and get back its
exposure gap and recommended new monitor locations without re-running the
pipeline.

Endpoints
----------
GET /                              health check
GET /cities                        list of cities the pipeline has results for
GET /exposure-gap/{city}           population-weighted exposure vs. official
                                    monitor average + the gap, for one city
GET /placement/{city}?k=3          top-k recommended new monitor locations
                                    for one city (k defaults to all computed)
GET /model-evaluation               cross-validated metrics for both
                                    calibration models + which was selected
GET /summary                       exposure gap for every city in one call
                                    (handy for a dashboard's landing view)

Run it:
    uvicorn api.main:app --reload --port 8000
Then e.g.:
    curl http://localhost:8000/exposure-gap/Delhi
    curl "http://localhost:8000/placement/Delhi?k=2"

Notes on design
-----------------
- This layer is deliberately READ-ONLY and serves pre-computed JSON -- it
  does not re-run the ML pipeline per-request (that would be slow and
  wasteful for data that only changes when new satellite/ground data
  arrives). Re-run `python run_pipeline.py` and restart the API whenever the
  underlying data or models are refreshed.
- Pydantic response models give FastAPI's automatic docs (visit /docs) real
  schemas instead of raw dicts.
"""

from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "results"

app = FastAPI(
    title="Air Quality Sensor Network Audit API",
    description="Population-weighted PM2.5 exposure gap and monitor placement recommendations for NCAP cities.",
    version="1.0.0",
)

# Allow a local dashboard/frontend on a different port to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{filename} not found -- run `python run_pipeline.py` first to generate results.",
        )
    with open(path) as f:
        return json.load(f)


class ExposureGapResponse(BaseModel):
    city: str
    population_weighted_exposure_ugm3: float
    official_monitor_average_ugm3: float
    exposure_gap_ugm3: float
    exposure_gap_pct: float
    interpretation: str


class PlacementPick(BaseModel):
    rank: int
    lat: float
    lon: float
    estimated_pm25_ugm3: float
    population_in_cell: float
    network_avg_after_ugm3: float
    remaining_gap_after_ugm3: float


class PlacementResponse(BaseModel):
    city: str
    target_population_weighted_exposure_ugm3: float
    current_network_avg_ugm3: float
    gap_before_ugm3: float
    recommended_new_stations: list[PlacementPick]


@app.get("/")
def health():
    return {"status": "ok", "service": "air-quality-sensor-audit-api"}


@app.get("/cities")
def list_cities():
    gap_data = _load_json("exposure_gap.json")
    return {"cities": list(gap_data.keys())}


@app.get("/exposure-gap/{city}", response_model=ExposureGapResponse)
def get_exposure_gap(city: str):
    gap_data = _load_json("exposure_gap.json")
    if city not in gap_data:
        raise HTTPException(status_code=404, detail=f"No results for city '{city}'. Try one of: {list(gap_data.keys())}")
    return {"city": city, **gap_data[city]}


@app.get("/placement/{city}", response_model=PlacementResponse)
def get_placement(city: str, k: Optional[int] = None):
    placement_data = _load_json("placement_recommendations.json")
    if city not in placement_data:
        raise HTTPException(status_code=404, detail=f"No results for city '{city}'. Try one of: {list(placement_data.keys())}")
    result = dict(placement_data[city])
    if k is not None:
        result["recommended_new_stations"] = result["recommended_new_stations"][:k]
    return result


@app.get("/model-evaluation")
def get_model_evaluation():
    return _load_json("model_evaluation.json")


@app.get("/summary")
def get_summary():
    gap_data = _load_json("exposure_gap.json")
    return {"cities": gap_data}
