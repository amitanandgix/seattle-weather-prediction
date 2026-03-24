import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_features(df: pd.DataFrame, is_prediction=False) -> pd.DataFrame:
    """
    Build features from raw weather DataFrame.
    If is_prediction=True, skip target creation (we're predicting, not training).
    """
    df = df.copy()

    # Fill minor gaps
    df.interpolate(method="time", limit=3, inplace=True)

    # Calendar features
    df["month"] = df.index.month
    df["day_of_year"] = df.index.day_of_year

    # Smooth seasonal cycle (avoids a hard jump from Dec 31 -> Jan 1)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Lag features
    for lag in [1, 2, 3, 7]:
        df[f"lag_precip_{lag}"] = df["precipitation_sum"].shift(lag)
        df[f"lag_temp_max_{lag}"] = df["temperature_2m_max"].shift(lag)

    # Rolling window features
    df["rolling_precip_7"]  = df["precipitation_sum"].shift(1).rolling(7).sum()
    df["rolling_precip_30"] = df["precipitation_sum"].shift(1).rolling(30).sum()
    df["rolling_temp_7"]    = df["temperature_2m_mean"].shift(1).rolling(7).mean()
    df["rolling_humidity_7"] = df["relative_humidity_2m_max"].shift(1).rolling(7).mean()

    # Pressure trend (today vs 3 days ago — falling = incoming rain)
    df["pressure_trend_3d"] = df["surface_pressure_mean"] - df["surface_pressure_mean"].shift(3)

    if not is_prediction:
        # Target: did it rain the NEXT day?
        df[config.TARGET_COLUMN] = (
            df["precipitation_sum"].shift(-1) > config.RAIN_THRESHOLD_MM
        ).astype(int)

        # Drop last row (no tomorrow) and rows with NaN from lags
        df = df.iloc[:-1]

    df.dropna(inplace=True)
    return df


def get_feature_columns():
    return [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "precipitation_hours",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        "cloud_cover_mean",
        "surface_pressure_mean",
        "month",
        "sin_doy",
        "cos_doy",
        "lag_precip_1",
        "lag_precip_2",
        "lag_precip_3",
        "lag_precip_7",
        "lag_temp_max_1",
        "lag_temp_max_2",
        "lag_temp_max_3",
        "lag_temp_max_7",
        "rolling_precip_7",
        "rolling_precip_30",
        "rolling_temp_7",
        "rolling_humidity_7",
        "pressure_trend_3d",
    ]


def load_and_process():
    """Load raw CSV, build features, save processed CSV, return DataFrame."""
    df = pd.read_csv(config.RAW_DATA_PATH, index_col="date", parse_dates=True)
    df = build_features(df)

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH)
    print(f"Processed data saved to {config.PROCESSED_DATA_PATH} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    load_and_process()
