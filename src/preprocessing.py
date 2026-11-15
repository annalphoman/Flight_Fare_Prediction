"""
preprocessing.py
================
Feature Engineering, Schema Transformation, and Preprocessing Pipelines
for the Indian Flight Awareness Project.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so modules can be run directly or as a package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.data_loader import load_clean_flight_data
except ImportError:
    from data_loader import load_clean_flight_data



# Canonical feature groups
NUMERICAL_FEATURES: List[str] = [
    "duration",
    "days_left",
    "is_peak_departure",
    "is_peak_arrival",
    "is_same_day_urgency",
]

CATEGORICAL_FEATURES: List[str] = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class",
    "route",
    "duration_category",
    "days_left_bucket",
]


def engineer_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain feature engineering to raw Indian flight records.

    Engineered features:
    - route: Composite string 'source_to_destination' (e.g. 'Delhi_to_Mumbai')
    - duration_category: Flight duration bracket ('Short (<3h)', 'Medium (3-6h)', 'Long (>6h)')
    - days_left_bucket: Lead-time urgency tier ('Critical (1-3d)', 'Urgent (4-10d)', 'Moderate (11-25d)', 'Early (26+d)')
    - is_peak_departure: Binary flag for morning and evening business departures
    - is_peak_arrival: Binary flag for morning, evening, and night arrivals
    - is_same_day_urgency: Binary flag for booking made within 48 hours of departure

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned flight records DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added engineered features.
    """
    data = df.copy()

    # 1. Route composite pair
    data["route"] = data["source_city"] + "_to_" + data["destination_city"]

    # 2. Flight Duration Categories
    duration_conditions = [
        data["duration"] < 3.0,
        (data["duration"] >= 3.0) & (data["duration"] < 6.0),
        data["duration"] >= 6.0,
    ]
    duration_choices = ["Short (<3h)", "Medium (3-6h)", "Long (>6h)"]
    data["duration_category"] = np.select(duration_conditions, duration_choices, default="Medium (3-6h)")

    # 3. Urgency Buckets based on Days Left
    urgency_conditions = [
        data["days_left"] <= 3,
        (data["days_left"] > 3) & (data["days_left"] <= 10),
        (data["days_left"] > 10) & (data["days_left"] <= 25),
        data["days_left"] > 25,
    ]
    urgency_choices = ["Critical (1-3d)", "Urgent (4-10d)", "Moderate (11-25d)", "Early (26+d)"]
    data["days_left_bucket"] = np.select(urgency_conditions, urgency_choices, default="Moderate (11-25d)")

    # 4. Peak Departure Indicator (Morning & Evening rush hours)
    data["is_peak_departure"] = data["departure_time"].isin(["Morning", "Evening"]).astype(int)

    # 5. Peak Arrival Indicator (Morning, Evening, Night rush)
    data["is_peak_arrival"] = data["arrival_time"].isin(["Morning", "Evening", "Night"]).astype(int)

    # 6. Same-day / Next-day Urgency flag
    data["is_same_day_urgency"] = (data["days_left"] <= 2).astype(int)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer that scales numerical variables
    and one-hot encodes categorical variables without data leakage.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor instance.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Applies feature engineering and splits the dataset into train and test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight records DataFrame.
    test_size : float
        Fraction of data for the test split. Defaults to 0.2.
    random_state : int
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    featured_df = engineer_flight_features(df)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = featured_df[feature_cols]
    y = featured_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def create_inference_sample(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a single flight dictionary from the web UI into a feature-engineered
    DataFrame ready to pass through the trained Scikit-learn pipeline.

    Parameters
    ----------
    input_data : Dict[str, Any]
        Dictionary of raw inputs with keys:
        'airline', 'source_city', 'destination_city', 'departure_time',
        'arrival_time', 'stops', 'class', 'duration', 'days_left'.

    Returns
    -------
    pd.DataFrame
        Single-row feature-engineered DataFrame.
    """
    raw_df = pd.DataFrame([input_data])
    featured_df = engineer_flight_features(raw_df)
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return featured_df[feature_cols]


if __name__ == "__main__":
    print("[*] Testing Preprocessing & Feature Engineering Module...")
    df = load_clean_flight_data(verbose=False)
    X_train, X_test, y_train, y_test = prepare_data_splits(df)
    print(f"[✓] Feature Engineering Succeeded!")
    print(f"    X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"    X_test shape:  {X_test.shape} | y_test shape:  {y_test.shape}")
    print(f"    Numerical Features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"    Categorical Features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
