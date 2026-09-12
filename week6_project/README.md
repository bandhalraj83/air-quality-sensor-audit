# Week 6 — Tuning, Exposure Gap & Sensor Placement

**Plan deliverable:** Tune the selected model; apply it to compute the population-weighted exposure gap per city; implement a greedy sensor-placement algorithm recommending next monitor locations.

## Where this lives
- Tuning: [`../full_pipeline/src/evaluate_models.py`](../full_pipeline/src/evaluate_models.py) (`_tune_selected_model`)
- Exposure gap: [`../full_pipeline/src/exposure_gap.py`](../full_pipeline/src/exposure_gap.py)
- Sensor placement: [`../full_pipeline/src/placement.py`](../full_pipeline/src/placement.py)

## Tuning
Once Model 1 (Ridge) was selected in Week 5, its `alpha` hyperparameter is grid-searched over `[0.1, 0.3, 1, 3, 10, 30, 100]` using the same GroupKFold methodology. Best found: `alpha=0.1` (RMSE 8.477, marginally better than the default `alpha=1.0`); RMSE degrades sharply past `alpha=30`, confirming regularization strength matters.

## Population-weighted exposure gap
`Exposure = Σ(populationᵢ × PM2.5ᵢ) / Σ populationᵢ`, compared against the official monitors' simple average. Results:

| City | Pop.-weighted | Official avg. | Gap |
|---|---|---|---|
| Delhi | 89.4 | 90.9 | -1.6 |
| Mumbai | 92.1 | 99.1 | -7.0 |
| Bengaluru | 93.6 | 92.7 | +0.8 |
| Chennai | 98.5 | 106.7 | -8.2 |

## Greedy sensor placement
`scoreᵢ = populationᵢ × |PM2.5ᵢ − network_avg|` — prioritizes densely populated cells most unlike what the current network reports. Top pick per city is visualized in `full_pipeline/outputs/plots/<city>_exposure_map.png`.

## Reproduce this step
```bash
cd ../full_pipeline
python -c "from src import evaluate_models, exposure_gap, placement; evaluate_models.run(); exposure_gap.run(); placement.run()"
```
