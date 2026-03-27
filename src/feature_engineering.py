import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _dew_point(temp_c, rh):
    """Approximate dew point (°C) using the Magnus formula."""
    rh = rh.clip(lower=1)  # avoid log(0)
    a, b = 17.625, 243.04
    gamma = np.log(rh / 100.0) + a * temp_c / (b + temp_c)
    return b * gamma / (a - gamma)


def build_features(df: pd.DataFrame, is_prediction=False) -> pd.DataFrame:
    df = df.copy()

    # Fill minor gaps
    df.interpolate(method="time", limit=3, inplace=True)

    # Calendar features
    df["month"]       = df.index.month
    df["day_of_year"] = df.index.day_of_year

    # Smooth seasonal cycle (avoids hard Dec 31 -> Jan 1 jump)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Dew point — strong rain precursor
    df["dew_point"] = _dew_point(df["temperature_2m_mean"], df["relative_humidity_2m_max"])

    # Rain streak: consecutive rainy days ending yesterday
    rain_now  = (df["precipitation_sum"] > config.RAIN_THRESHOLD_MM).astype(int)
    cumsum    = rain_now.cumsum()
    last_dry  = cumsum.where(rain_now == 0).ffill().fillna(0)
    df["rain_streak"] = (cumsum - last_dry).shift(1)

    # Wind speed trend (rising wind precedes frontal rain)
    df["wind_trend_3d"] = df["wind_speed_10m_max"] - df["wind_speed_10m_max"].shift(3)

    # Lag features — extended to 14 and 30 days
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag_precip_{lag}"] = df["precipitation_sum"].shift(lag)
    for lag in [1, 2, 3, 7]:
        df[f"lag_temp_max_{lag}"] = df["temperature_2m_max"].shift(lag)

    # Rolling window features
    df["rolling_precip_7"]   = df["precipitation_sum"].shift(1).rolling(7).sum()
    df["rolling_precip_30"]  = df["precipitation_sum"].shift(1).rolling(30).sum()
    df["rolling_temp_7"]     = df["temperature_2m_mean"].shift(1).rolling(7).mean()
    df["rolling_humidity_7"] = df["relative_humidity_2m_max"].shift(1).rolling(7).mean()

    # Pressure trends: 1, 3, and 7 days (falling = incoming rain)
    df["pressure_trend_1d"] = df["surface_pressure_mean"] - df["surface_pressure_mean"].shift(1)
    df["pressure_trend_3d"] = df["surface_pressure_mean"] - df["surface_pressure_mean"].shift(3)
    df["pressure_trend_7d"] = df["surface_pressure_mean"] - df["surface_pressure_mean"].shift(7)

    if not is_prediction:
        df[config.TARGET_COLUMN] = (
            df["precipitation_sum"].shift(-1) > config.RAIN_THRESHOLD_MM
        ).astype(int)
        df = df.iloc[:-1]

    df.dropna(inplace=True)
    return df


def get_feature_columns():
    return [
        # Raw daily variables
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
        # Calendar
        "month",
        "sin_doy",
        "cos_doy",
        # Derived
        "dew_point",
        "rain_streak",
        "wind_trend_3d",
        # Lags
        "lag_precip_1",
        "lag_precip_2",
        "lag_precip_3",
        "lag_precip_7",
        "lag_precip_14",
        "lag_precip_30",
        "lag_temp_max_1",
        "lag_temp_max_2",
        "lag_temp_max_3",
        "lag_temp_max_7",
        # Rolling windows
        "rolling_precip_7",
        "rolling_precip_30",
        "rolling_temp_7",
        "rolling_humidity_7",
        # Pressure trends
        "pressure_trend_1d",
        "pressure_trend_3d",
        "pressure_trend_7d",
        # Hourly sub-day features
        "hourly_morning_pressure",
        "hourly_morning_humidity",
        "hourly_afternoon_humidity",
        "hourly_pressure_drop",
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
