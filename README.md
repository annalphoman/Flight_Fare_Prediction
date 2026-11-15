# ✈️ Indian Flight Awareness — Flight Fare Prediction & Market Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-informational.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/pytest-passing-success.svg)](https://pytest.org/)

An end-to-end, modular machine learning project analyzing dynamic pricing behaviors across the Indian domestic aviation market and providing instant, real-time ticket fare predictions in Indian Rupees (₹).

---

## 📌 Project Overview

In the Indian aviation sector, flight fares fluctuate aggressively due to airline yield management algorithms, booking urgency (`days_left`), cabin class segregation, flight duration, and route demand.

This project delivers:
1. **Data Ingestion & Integrity Auditing (`src/data_loader.py`):** Automated validation and schema enforcement over 300,153 real flight records.
2. **Exploratory Data Analysis (`src/eda.py`):** Empirical analysis of airline fleet positioning, class disparity, and the exponential last-minute booking curve.
3. **Domain Feature Engineering (`src/preprocessing.py`):** Construction of domain indicators (`route`, `duration_category`, `urgency_bucket`, peak flight windows) encased in leak-free Scikit-learn `Pipeline` and `ColumnTransformer` instances.
4. **Multi-Model Regression Benchmark (`src/model_trainer.py`):** Comparative training and evaluation of Linear Regression, Ridge, Random Forest, and HistGradientBoosting regressors on held-out test splits.
5. **Interactive Web Application (`app.py`):** A modern 3-tab Streamlit web application featuring instant fare estimation, confidence bands ($\pm \text{MAE}$), traveler booking advisories, and interactive Plotly market analytics.

---

## 📂 Project Structure

```text
Flight_Fare_Prediction/
├── data/
│   ├── Clean_Dataset.csv          # Primary training dataset (300,153 verified flight records)
│   ├── economy.csv                # Class-specific split (kept standalone to prevent duplicates)
│   └── business.csv               # Class-specific split (kept standalone to prevent duplicates)
├── src/
│   ├── __init__.py                # Source package initialization
│   ├── data_loader.py             # Schema validation, missing value audits, index cleaning
│   ├── eda.py                     # Market pricing analytics, chart export, eda_summary.txt
│   ├── preprocessing.py           # Feature engineering, ColumnTransformer, train/test split
│   └── model_trainer.py           # Multi-model training, benchmarking, pipeline persistence
├── models/
│   ├── flight_fare_pipeline.joblib# Winning persisted pipeline (Preprocessor + Regressor)
│   └── model_metadata.json        # Test evaluation metrics, feature names, hyperparams
├── reports/
│   ├── figures/                   # High-resolution analytical charts (PNG)
│   │   ├── airline_price_distribution.png
│   │   ├── class_fare_comparison.png
│   │   ├── days_left_vs_price.png
│   │   ├── feature_importance.png
│   │   ├── route_price_heatmap.png
│   │   ├── stops_fare_distribution.png
│   │   └── time_slot_pricing.png
│   ├── eda_summary.txt            # Narrative market analysis report
│   ├── feature_importance.csv     # Ranked feature importances
│   └── model_comparison.csv       # Side-by-side model evaluation metrics
├── tests/
│   └── test_pipeline.py           # Unit and integration test suite
├── app.py                         # Interactive Streamlit application
├── requirements.txt               # Locked project dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 📊 Dataset Profile & Context

The project utilizes `data/Clean_Dataset.csv`, which contains **300,153 flight records** across 6 major Indian metropolitan hubs:
- **Metropolitan Cities:** Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai.
- **Airlines Represented:**
  - `Vistara` (127,859 flights — 42.6%)
  - `Air_India` (80,892 flights — 27.0%)
  - `Indigo` (43,120 flights — 14.4%)
  - `GO_FIRST` (23,173 flights — 7.7%)
  - `AirAsia` (16,098 flights — 5.4%)
  - `SpiceJet` (9,011 flights — 3.0%)
- **Travel Classes:** Economy (68.9%) and Business (31.1%).
- **Ticket Price Range:** ₹1,105.00 to ₹123,071.00 (Mean: ₹20,889.66).
- **Target Variable:** `price` (Supervised Continuous Regression).

> **Note on Data Cleanliness:** `data/economy.csv` and `data/business.csv` represent subsets that are already consolidated inside `Clean_Dataset.csv`. Concatenating them is avoided to ensure zero duplicate records. Redundant index columns (`Unnamed: 0`) are stripped at ingestion.

---

## 🧠 Feature Engineering & Preprocessing

All transformations are encapsulated inside Scikit-learn transformers to guarantee zero data leakage between training and evaluation splits:

1. **Composite Route (`route`):** Creates directional metro pairs (e.g., `Delhi_to_Mumbai`, `Bangalore_to_Kolkata`).
2. **Duration Buckets (`duration_category`):**
   - `Short (<3h)`: Fast non-stop metro flights.
   - `Medium (3-6h)`: Mid-haul or short layovers.
   - `Long (>6h)`: Extended connecting itineraries.
3. **Booking Urgency Tiers (`days_left_bucket`):**
   - `Critical (1-3d)`: Last-minute dynamic pricing surge window.
   - `Urgent (4-10d)`: Accelerating demand curve.
   - `Moderate (11-25d)`: Stable baseline pricing.
   - `Early (26+d)`: Early bird discounted inventory.
4. **Peak Rush Indicators:**
   - `is_peak_departure`: Morning & Evening peak business travel windows.
   - `is_peak_arrival`: Morning, Evening, and Night peak airport arrivals.
   - `is_same_day_urgency`: Same-day and next-day flights ($\le 2$ days).
5. **Column Transformer Structure:**
   - **Numerical:** `StandardScaler` applied to `duration`, `days_left`, and binary peak flags.
   - **Categorical:** `OneHotEncoder(handle_unknown='ignore')` applied to airline, route, cities, time slots, stops, and travel class.

---

## 🏆 Model Training & Benchmark Results

The training pipeline was executed on an **80/20 train-test split** (240,122 training rows, 60,031 test rows) across 4 regression architectures.

### Held-Out Test Evaluation Benchmark (`reports/model_comparison.csv`)

| Model Architecture | $R^2$ Score | RMSE (INR) | MAE (INR) | MAPE (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9762** | **₹3,499.43** | **₹1,789.73** | **12.12%** | **4.91s** |
| **HistGradientBoostingRegressor** | 0.9701 | ₹3,922.78 | ₹2,289.85 | 17.06% | 3.13s |
| **Linear Regression** | 0.9152 | ₹6,613.09 | ₹4,463.80 | 45.92% | 0.82s |
| **Ridge Regression** | 0.9152 | ₹6,613.09 | ₹4,463.79 | 45.92% | 0.45s |

### Key Machine Learning Takeaways
1. **Tree Ensembles Outperform Linear Baselines:** Random Forest and HistGradientBoosting capture complex non-linear interactions (e.g., how the `days_left` surge interacts with travel class and carrier type), pushing the $R^2$ score to **0.9762**.
2. **Low Prediction Error:** The winning Random Forest model achieves an average absolute error of just **₹1,789.73** on tickets averaging over ₹20,000, with a Mean Absolute Percentage Error (MAPE) of only **12.12%**.

### Feature Importance Hierarchy (`reports/feature_importance.csv`)
1. **`class_Business` & `class_Economy` (~90% combined importance):** Class configuration is the overwhelming primary driver of price.
2. **`duration` (~5% importance):** Direct flight distance and layover time.
3. **`days_left` (~1.2% importance):** Dynamic inventory yield management.
4. **`airline_Vistara` & `airline_Air_India`:** Full-service airline brand premium.

---

## 📈 Key Market Intelligence (EDA Insights)

From `reports/eda_summary.txt`:
- **The Class Multiplier:** Economy flights average **₹6,572**, whereas Business flights average **₹52,540** — an average **8.0x multiplier**.
- **The 72-Hour Fare Spike:** Economy tickets booked within 1-2 days of departure average **₹14,226**, compared to **₹5,045** when booked 20+ days prior (**+182% last-minute surge**).
- **Fleet Segmentation:** Vistara and Air India dominate total revenue by operating both Business and Economy cabins; budget carriers (IndiGo, SpiceJet, AirAsia, GO FIRST) compete strictly in single-class Economy configurations.
- **Connecting Flights vs. Direct:** 1-stop flights represent >83% of inventory and often cost more than direct flights due to multi-leg operating costs and extended total travel duration.

---

## 🖥️ Streamlit Web Application (`app.py`)

The application features a modern Indian aviation theme with three primary tabs:

- **Tab 1 — Flight Fare Predictor:**
  - Interactive selection of Airline, Cities, Times, Stops, Class, Duration, and Days Left.
  - Form validation prevents invalid identical origin-destination selections.
  - Predicts ticket fare in INR (₹) with confidence intervals ($\pm \text{MAE}$).
  - Dynamic **Traveler Smart Advisories** providing automated booking recommendations based on the `days_left` window.
- **Tab 2 — Indian Flight Market EDA:**
  - Interactive Plotly visualizations: Days Left surge curve, carrier fare comparisons, route volume rankings, and stops analysis.
- **Tab 3 — Model Performance & Insights:**
  - Live metric KPI cards, model comparison benchmark table, and interactive feature importance charts loaded from real metadata.

---

## 🚀 Step-by-Step Setup & Execution

### 1. Environment Setup
```bash
# Clone or navigate into the project directory
cd Flight_Fare_Prediction

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Step-by-Step Pipeline Execution

#### Step 3.1: Verify Data Ingestion & Schema
```bash
python src/data_loader.py
```
*Validates schema integrity, removes index columns, and prints diagnostic statistics.*

#### Step 3.2: Run Exploratory Data Analysis & Generate Figures
```bash
python src/eda.py
```
*Exports 6 high-resolution charts to `reports/figures/` and generates `reports/eda_summary.txt`.*

#### Step 3.3: Train, Benchmark, and Save the Best ML Pipeline
```bash
python src/model_trainer.py
```
*Trains 4 regression architectures, benchmarks evaluation metrics, updates `reports/model_comparison.csv`, and saves the winning pipeline to `models/flight_fare_pipeline.joblib`.*

#### Step 3.4: Run Pytest Test Suite
```bash
pytest tests/
```
*Executes all 5 unit and integration tests verifying schema validation, feature engineering, and inference consistency.*

#### Step 3.5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Opens the web dashboard in your browser at `http://localhost:8501`.*

---

## 🛠️ Technology Stack
- **Language:** Python 3
- **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-learn
- **Machine Learning Algorithms:** Random Forest, HistGradientBoosting, Linear Regression, Ridge
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Application Framework:** Streamlit
- **Testing:** Pytest

---

## 👨‍💻 Internship Project Summary
- **Project Title:** Indian Flight Awareness
- **Domain:** Aviation Analytics & Predictive Fare Modeling
- **Primary Dataset:** Indian Domestic Flight Records (300,153 observations)
- **Status:** Complete, Tested, and Production-Ready
