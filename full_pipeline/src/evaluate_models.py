"""
STEP 5 — MODEL EVALUATION & SELECTION
========================================
What this does
---------------
1. Loads the combined training table (all cities' station/month observations).
2. Runs SPATIAL cross-validation using GroupKFold grouped by station_id, NOT
   plain random K-fold. This matters: a random split would put readings from
   the SAME station in both train and test folds (different months of the
   same site), which leaks spatial information and overstates accuracy. By
   grouping on station_id, each fold tests on stations the model has never
   seen -- a fair proxy for "how well would this generalize to a brand-new
   monitoring point/city ward."
3. Computes RMSE, MAE, R^2, and mean bias for both Model 1 (Ridge) and
   Model 2 (Gradient Boosting) across folds.
4. Once a model family is selected, grid-searches a small set of that
   family's hyperparameters (Week 6 — "tune the selected model"), using the
   same GroupKFold methodology, and picks the best-performing configuration.
5. Refits the tuned model on ALL available data (once selected, we don't
   want to waste any ground-truth signal) and saves it to
   data/processed/models/best_model.joblib for use by exposure_gap.py.
6. Writes outputs/results/model_evaluation.json with per-fold and averaged
   metrics for both models, the tuning grid searched, and which
   model+hyperparameters were ultimately selected and why.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.calibration_models import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model_1,
    build_model_2,
    fit_model,
    predict,
    save_model,
)

N_FOLDS = 5

# Small, explicit hyperparameter grids -- kept intentionally short so tuning
# runs in seconds on this data volume, while still being a genuine search
# rather than a fixed guess. This directly implements the plan's Week 6
# "tune the selected model" step.
RIDGE_ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
GBM_PARAM_GRID = [
    {"learning_rate": 0.05, "max_depth": 3},
    {"learning_rate": 0.08, "max_depth": 4},
    {"learning_rate": 0.1, "max_depth": 5},
    {"learning_rate": 0.15, "max_depth": 3},
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),  # mean signed error
    }


def _cross_validate(build_fn, df: pd.DataFrame, n_splits: int) -> dict:
    # IMPORTANT: station_id strings (STN_000, STN_001, ...) are reused across
    # cities, so grouping on station_id alone would wrongly fuse Delhi's
    # STN_000 with Mumbai's STN_000 into a single CV group. Group on the
    # (city, station_id) pair instead so every physically distinct station
    # is its own group.
    groups = (df["city"].astype(str) + "__" + df["station_id"].astype(str)).values
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)  # can't have more folds than stations
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        pipeline = build_fn()
        fit_model(pipeline, train_df)
        y_pred = predict(pipeline, test_df)
        y_true = test_df[TARGET_COLUMN].values
        m = _metrics(y_true, y_pred)
        m["fold"] = fold_i
        m["n_test"] = len(test_df)
        fold_metrics.append(m)

    avg = {
        k: float(np.mean([f[k] for f in fold_metrics]))
        for k in ["rmse", "mae", "r2", "bias"]
    }
    return {"folds": fold_metrics, "average": avg}


def _tune_selected_model(selected_name: str, df: pd.DataFrame, n_splits: int) -> dict:
    """
    Week 6 — 'tune the selected model'. Once Model 1 vs Model 2 has been
    decided (above), grid-search a small set of hyperparameters for THAT
    model family only, using the same GroupKFold methodology, and return the
    best-performing configuration. This is a genuine (if intentionally
    small) search, not a fixed guess -- every candidate is actually
    cross-validated.
    """
    candidates = []
    if selected_name == "model_1_ridge":
        for alpha in RIDGE_ALPHA_GRID:
            candidates.append({"alpha": alpha})
    else:
        candidates = GBM_PARAM_GRID

    print(f"[evaluate_models] Tuning {selected_name} over {len(candidates)} candidate configuration(s) ...")
    tried = []
    for params in candidates:
        build_fn = (lambda p=params: build_model_1(**p)) if selected_name == "model_1_ridge" else (lambda p=params: build_model_2(**p))
        cv = _cross_validate(build_fn, df, n_splits)
        tried.append({"params": params, "average": cv["average"]})
        print(f"    {params} -> RMSE={cv['average']['rmse']:.3f}")

    best = min(tried, key=lambda t: t["average"]["rmse"])
    print(f"[evaluate_models] Best hyperparameters for {selected_name}: {best['params']} (RMSE={best['average']['rmse']:.3f})")
    return {"grid_searched": tried, "best_params": best["params"], "best_average": best["average"]}


def run() -> dict:
    df = pd.read_csv(DATA_PROCESSED_DIR / "training_table_all_cities.csv")
    n_unique_stations = (df["city"].astype(str) + "__" + df["station_id"].astype(str)).nunique()
    print(f"[evaluate_models] Training table: {len(df)} rows, {n_unique_stations} unique stations (across all cities)")

    print("[evaluate_models] Cross-validating Model 1 (Ridge regression) ...")
    cv_model1 = _cross_validate(build_model_1, df, N_FOLDS)
    print(f"    Model 1 avg: {cv_model1['average']}")

    print("[evaluate_models] Cross-validating Model 2 (Gradient Boosting) ...")
    cv_model2 = _cross_validate(build_model_2, df, N_FOLDS)
    print(f"    Model 2 avg: {cv_model2['average']}")

    # Selection rule: lower RMSE wins (ties broken by lower |bias|)
    rmse1, rmse2 = cv_model1["average"]["rmse"], cv_model2["average"]["rmse"]
    if rmse1 <= rmse2:
        selected_name, selected_builder = "model_1_ridge", build_model_1
    else:
        selected_name, selected_builder = "model_2_gbm", build_model_2

    print(f"[evaluate_models] Selected: {selected_name} (lower cross-validated RMSE)")

    # Week 6: tune the selected model's hyperparameters via grid search.
    tuning = _tune_selected_model(selected_name, df, N_FOLDS)
    best_params = tuning["best_params"]
    selected_builder = (lambda: build_model_1(**best_params)) if selected_name == "model_1_ridge" else (lambda: build_model_2(**best_params))

    # Refit the TUNED model on ALL data and persist it.
    final_pipeline = selected_builder()
    fit_model(final_pipeline, df)
    model_path = save_model(final_pipeline, "best_model")
    print(f"[evaluate_models] Refit tuned model on full data and saved -> {model_path}")

    results = {
        "n_training_rows": len(df),
        "n_stations": int(n_unique_stations),
        "n_folds": min(N_FOLDS, n_unique_stations),
        "model_1_ridge": cv_model1,
        "model_2_gbm": cv_model2,
        "selected_model": selected_name,
        "selection_rule": "lowest cross-validated RMSE (GroupKFold by city+station_id)",
        "tuning": tuning,
    }
    with open(RESULTS_DIR / "model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate_models] Wrote outputs/results/model_evaluation.json")

    return results


if __name__ == "__main__":
    run()
