"""
tests/test_pipeline.py
========================
Lightweight smoke tests -- NOT a substitute for real validation against
ground truth once real CPCB/SatPM/WorldPop/OpenAQ data is loaded, but they
catch broken plumbing (a column renamed in one step and not another, a NaN
leaking through, a city silently missing from an output file) before you
waste time debugging the API or dashboard.

Run: PYTHONPATH=. pytest tests/ -v
(Run `python run_pipeline.py` at least once first so the files these tests
check for actually exist.)
"""

import json

import numpy as np
import pandas as pd
import pytest

from config import CITIES, DATA_PROCESSED_DIR, RESULTS_DIR


@pytest.fixture(scope="module")
def exposure_gap_results():
    with open(RESULTS_DIR / "exposure_gap.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def placement_results():
    with open(RESULTS_DIR / "placement_recommendations.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_evaluation():
    with open(RESULTS_DIR / "model_evaluation.json") as f:
        return json.load(f)


def test_all_cities_present_in_exposure_gap(exposure_gap_results):
    assert set(exposure_gap_results.keys()) == set(CITIES.keys())


def test_exposure_values_are_physically_plausible(exposure_gap_results):
    for city, r in exposure_gap_results.items():
        assert 0 < r["population_weighted_exposure_ugm3"] < 500, city
        assert 0 < r["official_monitor_average_ugm3"] < 500, city


def test_gap_arithmetic_is_internally_consistent(exposure_gap_results):
    for city, r in exposure_gap_results.items():
        expected_gap = round(
            r["population_weighted_exposure_ugm3"] - r["official_monitor_average_ugm3"], 2
        )
        assert abs(expected_gap - r["exposure_gap_ugm3"]) < 0.05, city


def test_aligned_grid_has_no_nans():
    for city in CITIES:
        df = pd.read_csv(DATA_PROCESSED_DIR / f"aligned_grid_{city}.csv")
        assert not df["pm25_satellite"].isna().any(), city
        assert not df["population"].isna().any(), city


def test_corrected_grid_pm25_is_nonnegative():
    for city in CITIES:
        df = pd.read_csv(DATA_PROCESSED_DIR / f"corrected_grid_{city}.csv")
        assert (df["pm25_corrected"] > 0).all(), city


def test_placement_recommendations_within_city_bbox(placement_results):
    for city, result in placement_results.items():
        bbox = CITIES[city]["bbox"]
        for pick in result["recommended_new_stations"]:
            assert bbox["lat_min"] <= pick["lat"] <= bbox["lat_max"], (city, pick)
            assert bbox["lon_min"] <= pick["lon"] <= bbox["lon_max"], (city, pick)


def test_placement_recommendations_are_spatially_separated(placement_results):
    from config import MIN_SEPARATION_DEG
    for city, result in placement_results.items():
        picks = result["recommended_new_stations"]
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                d = np.sqrt(
                    (picks[i]["lat"] - picks[j]["lat"]) ** 2
                    + (picks[i]["lon"] - picks[j]["lon"]) ** 2
                )
                assert d >= MIN_SEPARATION_DEG * 0.999, (city, i, j)  # tiny tolerance for float rounding


def test_model_evaluation_selected_model_is_valid(model_evaluation):
    assert model_evaluation["selected_model"] in {"model_1_ridge", "model_2_gbm"}


def test_model_evaluation_metrics_are_reasonable(model_evaluation):
    for key in ["model_1_ridge", "model_2_gbm"]:
        avg = model_evaluation[key]["average"]
        assert avg["rmse"] > 0
        assert -1.0 <= avg["r2"] <= 1.0
