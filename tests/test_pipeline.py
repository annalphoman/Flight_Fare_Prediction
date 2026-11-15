"""
test_pipeline.py
================
Unit & Integration Tests for the Indian Flight Awareness Pipeline.
"""

import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import validate_schema, EXPECTED_COLUMNS
from src.preprocessing import (
    engineer_flight_features,
    create_inference_sample,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_flight_data():
    return pd.DataFrame({
        "airline": ["Vistara", "Indigo"],
        "flight": ["UK-995", "6E-512"],
        "source_city": ["Delhi", "Mumbai"],
        "departure_time": ["Morning", "Evening"],
        "stops": ["zero", "one"],
        "arrival_time": ["Afternoon", "Night"],
        "destination_city": ["Mumbai", "Bangalore"],
        "class": ["Economy", "Business"],
        "duration": [2.25, 5.5],
        "days_left": [15, 2],
        "price": [5955, 48200],
    })


def test_validate_schema_success(sample_flight_data):
    """Asserts schema validation succeeds on conformant flight data."""
    assert validate_schema(sample_flight_data) is True


def test_validate_schema_failure(sample_flight_data):
    """Asserts schema validation raises ValueError if mandatory columns are missing."""
    incomplete_df = sample_flight_data.drop(columns=["price"])
    with pytest.raises(ValueError):
        validate_schema(incomplete_df)


def test_feature_engineering_outputs(sample_flight_data):
    """Validates that engineered features exist with expected types."""
    featured = engineer_flight_features(sample_flight_data)

    assert "route" in featured.columns
    assert "duration_category" in featured.columns
    assert "days_left_bucket" in featured.columns
    assert "is_peak_departure" in featured.columns
    assert "is_peak_arrival" in featured.columns
    assert "is_same_day_urgency" in featured.columns

    # Check values
    assert featured.loc[0, "route"] == "Delhi_to_Mumbai"
    assert featured.loc[0, "duration_category"] == "Short (<3h)"
    assert featured.loc[1, "duration_category"] == "Medium (3-6h)"
    assert featured.loc[0, "days_left_bucket"] == "Moderate (11-25d)"
    assert featured.loc[1, "days_left_bucket"] == "Critical (1-3d)"
    assert featured.loc[0, "is_same_day_urgency"] == 0
    assert featured.loc[1, "is_same_day_urgency"] == 1


def test_create_inference_sample():
    """Validates single flight inference preparation format."""
    single_input = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "arrival_time": "Afternoon",
        "stops": "zero",
        "class": "Economy",
        "duration": 2.2,
        "days_left": 10,
    }
    sample_df = create_inference_sample(single_input)
    assert len(sample_df) == 1
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        assert col in sample_df.columns


def test_model_pipeline_prediction():
    """Tests that the saved model pipeline loads and makes sensible predictions."""
    import joblib
    pipeline_path = PROJECT_ROOT / "models" / "flight_fare_pipeline.joblib"
    if not pipeline_path.exists():
        pytest.skip("Trained pipeline not found at models/flight_fare_pipeline.joblib")

    pipeline = joblib.load(pipeline_path)

    # Economy flight
    eco_input = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "arrival_time": "Afternoon",
        "stops": "zero",
        "class": "Economy",
        "duration": 2.2,
        "days_left": 25,
    }
    eco_df = create_inference_sample(eco_input)
    eco_pred = pipeline.predict(eco_df)[0]

    # Business flight (same flight details)
    biz_input = eco_input.copy()
    biz_input["class"] = "Business"
    biz_df = create_inference_sample(biz_input)
    biz_pred = pipeline.predict(biz_df)[0]

    # Basic physical sanity assertions
    assert eco_pred > 1000.0
    assert biz_pred > eco_pred
    assert biz_pred > 20000.0  # Business cabins command high fare
