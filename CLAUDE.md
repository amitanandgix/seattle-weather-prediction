# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full pipeline (download data → engineer features → train → evaluate → predict)
python main.py

# Force re-download and retrain from scratch
python main.py --retrain

# Predict only (model must already be trained)
python main.py --predict

# Run individual stages directly
python src/data_collection.py      # download historical data
python src/feature_engineering.py  # process raw CSV into features
python src/train.py                 # train and evaluate the model
python src/predict.py               # predict tomorrow's weather
python src/evaluate.py              # regenerate report charts
```

Python interpreter on this machine: `C:\Users\amita\AppData\Local\Microsoft\WindowsApps\python3.exe`

## Architecture

The pipeline is linear with five stages, each skippable if its output already exists:

```
data_collection  →  feature_engineering  →  train  →  evaluate  →  predict
     ↓                      ↓                 ↓            ↓            ↓
data/raw/*.csv   data/processed/*.csv   models/*.joblib  reports/  data/predictions/
```

**`config.py`** is the single source of truth — all file paths, Seattle coordinates, date ranges, the rain threshold (0.1 mm), and the train/test cutoff date live here. Change the location or date range here before touching any other file.

**`src/feature_engineering.py`** is the most critical file. `build_features()` is called in two contexts:
- Training: `is_prediction=False` — creates the `next_day_rain` target by shifting precipitation by -1 day, then drops the last row
- Prediction: `is_prediction=True` — skips target creation, keeps the latest row as the feature vector

Rolling and lag features use `.shift(1)` before windowing to prevent data leakage (only past data is used). The feature list returned by `get_feature_columns()` must stay in sync with what `build_features()` actually produces — train and predict both use this same list.

**`src/train.py`** uses a strict time-based split at `config.TRAIN_CUTOFF_DATE`. Never use random shuffling on this data. `scale_pos_weight` is computed from the training set to handle class imbalance automatically.

**`src/predict.py`** fetches 40 days of recent data (enough for 30-day rolling windows + lag features) via the Open-Meteo forecast API, runs it through the same `build_features()` call, and takes the last row as today's feature vector.

## Key Constraints

- **No API key required** — Open-Meteo is free and unauthenticated. Historical data comes from `archive-api.open-meteo.com`, live data from `api.open-meteo.com`.
- **Data and models are gitignored** — `data/` and `models/` are not committed. Anyone cloning must run `python main.py` to regenerate them.
- **`day_of_year`** is used only internally to compute `sin_doy`/`cos_doy` and is not included in `get_feature_columns()` — do not add it directly as a feature or the model will see a discontinuity at year boundaries.
