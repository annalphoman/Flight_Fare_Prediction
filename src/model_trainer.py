"""
model_trainer.py
================
Multi-Model Training, Hyperparameter Tuning, Evaluation Benchmarking,
and Pipeline Persistence for the Indian Flight Awareness Project.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Set Matplotlib config dir to writable /tmp directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add project root to sys.path so modules can be run directly or as a package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

try:
    from src.data_loader import load_clean_flight_data
    from src.preprocessing import build_preprocessor, prepare_data_splits
except ImportError:
    from data_loader import load_clean_flight_data
    from preprocessing import build_preprocessor, prepare_data_splits


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, MAE, R-squared, and MAPE metrics.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth ticket prices.
    y_pred : np.ndarray
        Model predicted ticket prices.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed regression evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 2),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_comparison_path: str = "reports/model_comparison.csv",
    models_dir: str = "models",
    figures_dir: str = "reports/figures"
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Trains multiple regression candidates using Scikit-learn Pipelines,
    evaluates them on the held-out test split, benchmarks their performance,
    and persists the best performing model pipeline and metadata.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target prices.
    y_test : pd.Series
        Test target prices.
    output_comparison_path : str
        Path to save CSV comparison benchmark.
    models_dir : str
        Directory to save best pipeline joblib and metadata JSON.
    figures_dir : str
        Directory to save feature importance visualization.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any], pd.DataFrame]
        (best_pipeline, best_metadata, comparison_df)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_comparison_path), exist_ok=True)

    # Define model candidates
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=35,
            max_depth=16,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=12,
            l2_regularization=0.1,
            random_state=42
        ),
    }

    comparison_records = []
    trained_pipelines = {}

    print("=" * 75)
    print("STARTING REGRESSION MODEL TRAINING & BENCHMARKING (300k Records)")
    print("=" * 75)

    for model_name, regressor in models.items():
        print(f"\n[*] Training Candidate: {model_name} ...")
        start_time = time.time()

        # Build pipeline: Preprocessor + Regressor
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # Train pipeline
        pipeline.fit(X_train, y_train)
        training_duration = time.time() - start_time

        # Predict on held-out test split
        y_pred = pipeline.predict(X_test)

        # Compute evaluation metrics
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["training_time_sec"] = round(training_duration, 2)
        metrics["model"] = model_name

        print(f"    [✓] Finished in {training_duration:.2f}s | R²: {metrics['r2_score']:.4f} | RMSE: ₹{metrics['rmse']:,.2f} | MAE: ₹{metrics['mae']:,.2f}")

        comparison_records.append(metrics)
        trained_pipelines[model_name] = pipeline

    # Create benchmark DataFrame
    comparison_df = pd.DataFrame(comparison_records)
    column_order = ["model", "r2_score", "rmse", "mae", "mape_percent", "training_time_sec"]
    comparison_df = comparison_df[column_order].sort_values(by="r2_score", ascending=False).reset_index(drop=True)

    # Save comparison to CSV
    comparison_df.to_csv(output_comparison_path, index=False)
    print(f"\n[✓] Saved comparison benchmark to: {output_comparison_path}")

    # Identify best model
    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_row = comparison_df.iloc[0].to_dict()

    print("\n" + "=" * 75)
    print(f"★ BEST PERFORMING MODEL: {best_model_name}")
    print(f"  R² Score:     {best_row['r2_score']:.4f}")
    print(f"  RMSE:         ₹{best_row['rmse']:,.2f}")
    print(f"  MAE:          ₹{best_row['mae']:,.2f}")
    print(f"  MAPE:         {best_row['mape_percent']:.2f}%")
    print("=" * 75)

    # Save winning pipeline
    pipeline_save_path = os.path.join(models_dir, "flight_fare_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_save_path, compress=3)
    print(f"[✓] Saved best model pipeline to: {pipeline_save_path}")

    # Extract feature names from preprocessor
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    # Save Model Metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "evaluation_metrics": {
            "rmse": best_row["rmse"],
            "mae": best_row["mae"],
            "r2_score": best_row["r2_score"],
            "mape_percent": best_row["mape_percent"],
            "training_time_sec": best_row["training_time_sec"],
        },
        "training_metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_features": len(transformed_feature_names),
            "training_timestamp": datetime.now().isoformat(),
            "target_variable": "price",
            "target_currency": "INR",
        },
        "feature_names": transformed_feature_names,
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[✓] Saved model metadata to: {metadata_path}")

    # Compute Feature Importances & Save Visualization
    print("\n[*] Calculating Feature Importance ranking...")
    extract_and_save_feature_importances(
        best_pipeline,
        best_model_name,
        X_test,
        y_test,
        transformed_feature_names,
        figures_dir
    )

    return best_pipeline, metadata, comparison_df


def extract_and_save_feature_importances(
    pipeline: Pipeline,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    figures_dir: str
) -> None:
    """
    Computes permutation importance or model feature importances,
    exports reports/feature_importance.csv, and plots reports/figures/feature_importance.png.
    """
    regressor = pipeline.named_steps["regressor"]

    importances = None
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        # Use permutation importance on a representative sample of 2,500 test rows for speed
        sample_size = min(2500, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        y_sample = y_test.iloc[:sample_size]
        perm_res = permutation_importance(
            pipeline, X_sample, y_sample, n_repeats=3, random_state=42, n_jobs=-1
        )
        importances = perm_res.importances_mean

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Save to CSV
    csv_path = "reports/feature_importance.csv"
    feat_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved feature importance rankings to: {csv_path}")

    # Plot Top 15 Features
    top_15 = feat_df.head(15).iloc[::-1]  # Reverse for horizontal barplot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top_15,
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
        ax=ax
    )
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Engineered Feature")
    plt.tight_layout()

    chart_path = os.path.join(figures_dir, "feature_importance.png")
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved feature importance plot to: {chart_path}")


if __name__ == "__main__":
    df = load_clean_flight_data(verbose=False)
    X_train, X_test, y_train, y_test = prepare_data_splits(df)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
