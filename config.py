SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321
TIMEZONE = "America/Los_Angeles"

HISTORICAL_START = "2006-01-01"
HISTORICAL_END   = "2025-12-31"

RAW_DATA_PATH       = "data/raw/seattle_weather_raw.csv"
PROCESSED_DATA_PATH = "data/processed/seattle_weather_features.csv"
MODEL_PATH          = "models/ensemble_model.joblib"
SCALER_PATH         = "models/scaler.joblib"

TARGET_COLUMN = "next_day_rain"
RAIN_THRESHOLD_MM = 1.0  # meaningful rain (raised from 0.1mm to reduce noise)

TRAIN_CUTOFF_DATE = "2024-01-01"  # train on data before this, test on data after
