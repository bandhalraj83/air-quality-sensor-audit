# Week 4 — Model 2: Alternate Calibration Approach

**Plan deliverable:** Build Model 2 — an alternate calibration approach (e.g. gradient boosting) and compare against Model 1.

## Where this lives
Defined in [`../full_pipeline/src/calibration_models.py`](../full_pipeline/src/calibration_models.py) (`build_model_2`); compared against Model 1 in [`../full_pipeline/src/evaluate_models.py`](../full_pipeline/src/evaluate_models.py).

## What it is
A Histogram Gradient Boosting Regressor (non-linear, tree-based ensemble) using the same feature set as Model 1, so the Week 5 comparison is apples-to-apples.

## Reproduce this step
```bash
cd ../full_pipeline
python -c "
from src.calibration_models import build_model_2, fit_model
import pandas as pd
df = pd.read_csv('data/processed/training_table_all_cities.csv')
model = build_model_2()
fit_model(model, df)
print('Model 2 trained on', len(df), 'rows')
"
```
See Week 5 for the actual head-to-head comparison and metrics.
