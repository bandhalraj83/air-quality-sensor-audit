# Week 7 — FastAPI Endpoint

**Plan deliverable:** Build a FastAPI endpoint serving the exposure gap and recommended placement for a queried city or ward.

## Where this lives
[`../full_pipeline/api/main.py`](../full_pipeline/api/main.py).

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/exposure-gap/{city}` | Population-weighted exposure, official average, gap |
| GET | `/placement/{city}?k=3` | Top-k recommended new monitor locations |
| GET | `/model-evaluation` | Cross-validated metrics for both models |
| GET | `/summary` | Exposure gap for every city in one call |

## Run it
```bash
cd ../full_pipeline
uvicorn api.main:app --reload --port 8000
curl http://localhost:8000/exposure-gap/Delhi
```
Interactive docs: `http://localhost:8000/docs`.
