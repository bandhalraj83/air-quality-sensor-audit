"""
STEP 3 & 4 — BIAS-CORRECTION MODELS
=====================================
What this does
---------------
Trains TWO calibration models that learn to correct satellite PM2.5 toward
ground truth (OpenAQ), then hands both to evaluate_models.py for comparison.

  Model 1 (Week 3): Ridge regression        -- simple, interpretable baseline
  Model 2 (Week 4): Gradient boosting       -- HistGradientBoostingRegressor,
                                                a non-linear alternative that
                                                can capture interactions
                                                (e.g. bias worsens with month/
                                                season and with concentration)

Features used (kept deliberately simple / defensible from available data):
  - pm25_satellite     the raw satellite estimate (the value being corrected)
  - month              calendar month 1-12, as a proxy for season
  - lat, lon           spatial position (lets the model learn regional bias
                        patterns beyond what pm25_satellite alone captures)

Target:
  - pm25_ground        OpenAQ ground-truth reading at the same station/month

Both models are fit on the SAME train split (produced by evaluation.py's
spatial group split) so the comparison in Step 5 is apples-to-apples.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR

FEATURE_COLUMNS = ["pm25_satellite", "month", "lat", "lon"]
TARGET_COLUMN = "pm25_ground"


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline


def build_model_1(alpha: float = 1.0) -> Pipeline:
    """Model 1: standardized Ridge regression -- linear bias correction."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=0)),
    ])


def build_model_2(learning_rate: float = 0.08, max_depth: int = 4, max_iter: int = 200) -> Pipeline:
    """Model 2: gradient boosting -- non-linear bias correction."""
    return Pipeline([
        ("gbm", HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=0.1,
            random_state=0,
        )),
    ])


def fit_model(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    X = train_df[FEATURE_COLUMNS].values
    y = train_df[TARGET_COLUMN].values
    pipeline.fit(X, y)
    return pipeline


MIN_PHYSICAL_PM25 = 1.0  # ug/m3 -- PM2.5 can't be negative or truly zero in practice


def predict(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """
    Predict bias-corrected PM2.5, clipped to a physically valid floor.

    Why the clip is needed: Model 1 is a LINEAR model, so at grid cells whose
    features sit outside the range seen during training (e.g. very low
    satellite readings at a city's edge in a clean month) it can extrapolate
    to non-physical negative values. Gradient boosting can't extrapolate
    below its training targets the same way, but we clip both models here
    for a single, consistent guarantee downstream.
    """
    X = df[FEATURE_COLUMNS].values
    preds = pipeline.predict(X)
    return np.clip(preds, MIN_PHYSICAL_PM25, None)


def save_model(pipeline: Pipeline, name: str) -> str:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return str(path)


def load_model(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
