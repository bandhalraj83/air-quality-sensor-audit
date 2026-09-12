# Week 5 — Model Evaluation & Selection

**Plan deliverable:** Evaluate both calibration models (cross-validation, error metrics); select the better-performing approach.

## Where this lives
[`../full_pipeline/src/evaluate_models.py`](../full_pipeline/src/evaluate_models.py).

## What it does
- Cross-validates both Model 1 and Model 2 using `GroupKFold` grouped by `(city, station_id)` — not random splits — so every fold tests on stations the model never trained on.
- Computes RMSE, MAE, R², and bias for both models.
- Selects the model with the lowest cross-validated RMSE.

## Result on the current (synthetic) data
| Model | RMSE (µg/m³) | MAE | R² | Bias |
|---|---|---|---|---|
| Model 1 — Ridge | 8.48 | 6.80 | 0.904 | -0.02 |
| Model 2 — Gradient Boosting | 9.16 | 7.25 | 0.885 | +0.42 |

**Selected: Model 1 (Ridge)** — the injected satellite bias is close to linear, and with only 444 training rows across 37 stations, the more flexible Model 2 doesn't have enough data to reliably out-perform a well-regularized linear model.

## Reproduce this step
```bash
cd ../full_pipeline
python -c "from src import evaluate_models; evaluate_models.run()"
```
Output: `full_pipeline/outputs/results/model_evaluation.json`.
