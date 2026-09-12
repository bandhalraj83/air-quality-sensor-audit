# Week 3 — Model 1: Bias-Correction Regression

**Plan deliverable:** Build Model 1 — a bias-correction regression calibrating satellite PM2.5 against OpenAQ ground readings.

## Where this lives
Defined in [`../full_pipeline/src/calibration_models.py`](../full_pipeline/src/calibration_models.py) (`build_model_1`) and trained/evaluated in [`../full_pipeline/src/evaluate_models.py`](../full_pipeline/src/evaluate_models.py), as part of the integrated `full_pipeline/` codebase (see its [README](../full_pipeline/README.md)).

## What it is
A Ridge regression (linear, L2-regularized) that predicts `pm25_ground` (OpenAQ) from `pm25_satellite`, `month`, `lat`, `lon`. Chosen as a simple, interpretable baseline that's stable even with a small training set.

## Reproduce this step
```bash
cd ../full_pipeline
python -c "
from src.calibration_models import build_model_1, fit_model
import pandas as pd
df = pd.read_csv('data/processed/training_table_all_cities.csv')
model = build_model_1()
fit_model(model, df)
print('Model 1 trained on', len(df), 'rows')
"
```
Full cross-validated performance is computed in Week 5.
