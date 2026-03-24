# Seattle Weather Prediction

A machine learning model that predicts whether it will rain tomorrow in Seattle, using XGBoost and 10 years of historical weather data.

## Overview

- **Model:** XGBoost classifier
- **Accuracy:** ~77% on 2024–2025 test data
- **Data source:** [Open-Meteo](https://open-meteo.com/) — free, no API key required
- **Target:** Binary prediction — will it rain tomorrow in Seattle? (Yes / No + probability)

## Project Structure

```
seattle-weather-prediction/
├── config.py                   # Seattle coordinates, paths, constants
├── main.py                     # Entry point — runs the full pipeline
├── requirements.txt            # Python dependencies
└── src/
    ├── data_collection.py      # Fetches weather data from Open-Meteo API
    ├── feature_engineering.py  # Builds lag, rolling, and seasonal features
    ├── train.py                # Trains and evaluates the XGBoost model
    ├── predict.py              # Predicts tomorrow's rain
    └── evaluate.py             # Generates evaluation charts and reports
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/amitanandgix/seattle-weather-prediction.git
cd seattle-weather-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
1. Download ~10 years of Seattle weather data from Open-Meteo
2. Engineer features (lag, rolling averages, seasonal signals)
3. Train the XGBoost model
4. Save evaluation charts to `/reports/`
5. Print tomorrow's rain prediction

### Other commands

```bash
python main.py --retrain    # Force re-download and retrain from scratch
python main.py --predict    # Only run prediction (model must already be trained)
```

## Features Used

| Feature | Description |
|---|---|
| Temperature (max, min, mean) | Daily temperature readings |
| Precipitation | Rain amount today (mm) |
| Wind speed & direction | Max wind speed and prevailing direction |
| Humidity | Max and min relative humidity |
| Cloud cover | Mean cloud cover percentage |
| Surface pressure | Atmospheric pressure (hPa) |
| Pressure trend | Pressure change over last 3 days |
| Lag features | Precipitation and temperature from past 1, 2, 3, 7 days |
| Rolling averages | 7-day and 30-day rolling precipitation and temperature |
| Seasonal signals | Month, and sine/cosine encoding of day-of-year |

## Model Performance

Trained on 2015–2023 data, evaluated on 2024–2025:

| Metric | Score |
|---|---|
| Accuracy | 76.7% |
| F1 Score | 0.77 |
| Precision (Rain) | 0.75 |
| Recall (Rain) | 0.80 |

## Sample Output

```
=============================================
  Seattle Weather Prediction
  Date: 2026-03-24
=============================================
  Rain tomorrow : NO
  Confidence    : 71.8%
  Model         : XGBoost
=============================================
```

## Reports

After running the pipeline, the following charts are saved to `/reports/`:

- `confusion_matrix.png` — predicted vs actual rain/no-rain
- `feature_importance.png` — top 15 most influential features
- `monthly_accuracy.png` — model accuracy broken down by month
- `shap_summary.png` — SHAP values showing how each feature drives predictions

## Notes

- Data and trained models are excluded from this repository (see `.gitignore`)
- The model uses a **time-based train/test split** — no random shuffling — to avoid data leakage
- Seattle coordinates are set in `config.py` and can be changed to predict for any location
