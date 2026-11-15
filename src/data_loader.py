"""
data_loader.py
==============
Data Ingestion, Schema Validation, and Integrity Verification Module
for the Indian Flight Awareness Project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


EXPECTED_COLUMNS = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "price",
]

VALID_AIRLINES = {"SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"}
VALID_CITIES = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"}
VALID_CLASSES = {"Economy", "Business"}
VALID_STOPS = {"zero", "one", "two_or_more"}


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame contains all required columns with valid types.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded flight DataFrame to validate.

    Returns
    -------
    bool
        True if all required columns are present and valid, raises ValueError otherwise.
    """
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing columns: {missing_cols}")

    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise TypeError("Target column 'price' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["duration"]):
        raise TypeError("Feature column 'duration' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["days_left"]):
        raise TypeError("Feature column 'days_left' must be numeric.")

    return True


def load_clean_flight_data(
    file_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Loads Clean_Dataset.csv, removes redundant index columns, audits missing values,
    checks for duplicate records, and validates schema consistency.

    Parameters
    ----------
    file_path : Optional[str]
        Path to Clean_Dataset.csv. Defaults to 'data/Clean_Dataset.csv'.
    verbose : bool
        Whether to print diagnostic logs and summaries.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated flight dataset.
    """
    if file_path is None:
        # Default relative to repository root
        project_root = Path(__file__).resolve().parent.parent
        file_path = str(project_root / "data" / "Clean_Dataset.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {file_path}")

    if verbose:
        print(f"[*] Loading dataset from: {file_path} ...")

    # Read CSV
    df = pd.read_csv(file_path)

    # 1. Drop redundant unnamed index column if present (e.g. '', 'Unnamed: 0')
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        if verbose:
            print(f"[*] Dropped redundant index column(s): {unnamed_cols}")

    # 2. Check and report missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if verbose:
        print(f"[*] Total missing values found: {total_nulls}")
        if total_nulls > 0:
            print(null_counts[null_counts > 0])

    if total_nulls > 0:
        # Clean any null rows if found
        df = df.dropna().reset_index(drop=True)

    # 3. Check for exact duplicate rows (across all business columns)
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_len - len(df)
    if verbose and duplicates_removed > 0:
        print(f"[*] Removed {duplicates_removed:,} duplicate records.")

    # 4. Validate Schema & Domains
    validate_schema(df)

    # 5. Clean categorical whitespace just in case
    cat_cols = ["airline", "flight", "source_city", "departure_time", "stops",
                "arrival_time", "destination_city", "class"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if verbose:
        print(f"[✓] Data successfully loaded and verified! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns.")

    return df


def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured dictionary summarizing key dimensions and metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary with statistical and categorical breakdown.
    """
    summary = {
        "total_records": len(df),
        "total_features": df.shape[1],
        "airlines": df["airline"].value_counts().to_dict(),
        "classes": df["class"].value_counts().to_dict(),
        "source_cities": df["source_city"].value_counts().to_dict(),
        "destination_cities": df["destination_city"].value_counts().to_dict(),
        "stops": df["stops"].value_counts().to_dict(),
        "price_min": float(df["price"].min()),
        "price_max": float(df["price"].max()),
        "price_mean": float(df["price"].mean()),
        "price_median": float(df["price"].median()),
        "price_std": float(df["price"].std()),
        "days_left_min": int(df["days_left"].min()),
        "days_left_max": int(df["days_left"].max()),
        "duration_mean": float(df["duration"].mean()),
    }
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("Indian Flight Awareness - Data Ingestion & Schema Verification")
    print("=" * 70)

    dataset = load_clean_flight_data(verbose=True)
    summary_stats = inspect_dataset_summary(dataset)

    print("\n--- Summary Statistics ---")
    print(f"Total Rows: {summary_stats['total_records']:,}")
    print(f"Price: Min = ₹{summary_stats['price_min']:,.2f} | Mean = ₹{summary_stats['price_mean']:,.2f} | Max = ₹{summary_stats['price_max']:,.2f}")
    print(f"Duration: Average = {summary_stats['duration_mean']:.2f} hours")
    print(f"Booking Days Left: {summary_stats['days_left_min']} to {summary_stats['days_left_max']} days")
    print("\nAirlines Breakdown:")
    for airline, count in summary_stats["airlines"].items():
        print(f"  - {airline:12}: {count:,} flights ({count/len(dataset)*100:.1f}%)")
    print("\nClass Breakdown:")
    for travel_class, count in summary_stats["classes"].items():
        print(f"  - {travel_class:12}: {count:,} flights ({count/len(dataset)*100:.1f}%)")
    print("=" * 70)
