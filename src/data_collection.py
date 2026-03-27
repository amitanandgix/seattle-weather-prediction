import requests
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
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
]

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
]


def _fetch_from_api(url, params):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_hourly(url, start_date, end_date, chunk_years=2):
    """Fetch hourly data in yearly chunks to avoid API rate limits."""
    from datetime import date as date_type
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), end)
        params = {
            "latitude":   config.SEATTLE_LAT,
            "longitude":  config.SEATTLE_LON,
            "start_date": chunk_start.strftime("%Y-%m-%d"),
            "end_date":   chunk_end.strftime("%Y-%m-%d"),
            "hourly":     ",".join(HOURLY_VARIABLES),
            "timezone":   config.TIMEZONE,
        }
        print(f"  Hourly chunk: {params['start_date']} to {params['end_date']}...")
        data = _fetch_from_api(url, params)
        df_chunk = pd.DataFrame(data["hourly"])
        df_chunk["time"] = pd.to_datetime(df_chunk["time"])
        df_chunk.set_index("time", inplace=True)
        chunks.append(df_chunk)
        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(3)
    return pd.concat(chunks)


def _aggregate_hourly_to_daily(df_hourly):
    """Compute morning/afternoon sub-day features from hourly data."""
    df = df_hourly.copy()
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour

    morning   = df[df["hour"].between(6, 9)]
    afternoon = df[df["hour"].between(15, 18)]

    morn_p = morning.groupby("date")["pressure_msl"].mean().rename("hourly_morning_pressure")
    aftn_p = afternoon.groupby("date")["pressure_msl"].mean()
    morn_h = morning.groupby("date")["relative_humidity_2m"].mean().rename("hourly_morning_humidity")
    aftn_h = afternoon.groupby("date")["relative_humidity_2m"].mean().rename("hourly_afternoon_humidity")

    result = pd.concat([morn_p, morn_h, aftn_h], axis=1)
    result["hourly_pressure_drop"] = morn_p - aftn_p  # positive = pressure falling through day
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result


DAILY_CACHE = "data/raw/seattle_daily_cache.csv"
HOURLY_CACHE = "data/raw/seattle_hourly_cache.csv"

def download_historical():
    """Download historical daily + hourly weather data for Seattle and save to CSV."""
    os.makedirs("data/raw", exist_ok=True)

    # --- Daily data (cached so we don't re-download if hourly fails) ---
    if os.path.exists(DAILY_CACHE):
        print("Loading cached daily data...")
        df_daily = pd.read_csv(DAILY_CACHE, index_col="date", parse_dates=True)
    else:
        print("Downloading historical daily weather data from Open-Meteo...")
        params = {
            "latitude":   config.SEATTLE_LAT,
            "longitude":  config.SEATTLE_LON,
            "start_date": config.HISTORICAL_START,
            "end_date":   config.HISTORICAL_END,
            "daily":      ",".join(DAILY_VARIABLES),
            "timezone":   config.TIMEZONE,
        }
        data = _fetch_from_api(ARCHIVE_URL, params)
        df_daily = pd.DataFrame(data["daily"])
        df_daily.rename(columns={"time": "date"}, inplace=True)
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        df_daily.set_index("date", inplace=True)
        df_daily.to_csv(DAILY_CACHE)
        print(f"Daily data cached to {DAILY_CACHE}")

    # --- Hourly data (cached separately) ---
    if os.path.exists(HOURLY_CACHE):
        print("Loading cached hourly data...")
        df_hourly = pd.read_csv(HOURLY_CACHE, index_col="time", parse_dates=True)
    else:
        print("Downloading hourly data for sub-day features (this will take a few minutes)...")
        print("Waiting 3 minutes to clear API rate limit...")
        time.sleep(180)
        df_hourly = _fetch_hourly(ARCHIVE_URL, config.HISTORICAL_START, config.HISTORICAL_END)
        df_hourly.to_csv(HOURLY_CACHE)
        print(f"Hourly data cached to {HOURLY_CACHE}")

    df_hourly_daily = _aggregate_hourly_to_daily(df_hourly)
    df = df_daily.join(df_hourly_daily, how="left")

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH)
    print(f"Saved {len(df)} rows to {config.RAW_DATA_PATH}")
    return df


def fetch_recent_for_prediction(days=40, forecast_days=3):
    """Fetch the last `days` of historical data plus `forecast_days` of forecast ahead."""
    from datetime import date, timedelta

    end_date   = (date.today() + timedelta(days=forecast_days)).isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    print(f"Fetching recent data ({start_date} to {end_date})...")

    params = {
        "latitude":   config.SEATTLE_LAT,
        "longitude":  config.SEATTLE_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   config.TIMEZONE,
    }
    time.sleep(0.5)
    data = _fetch_from_api(FORECAST_URL, params)
    df_daily = pd.DataFrame(data["daily"])
    df_daily.rename(columns={"time": "date"}, inplace=True)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily.set_index("date", inplace=True)

    time.sleep(0.5)
    df_hourly = _fetch_hourly(FORECAST_URL, start_date, end_date)
    df_hourly_daily = _aggregate_hourly_to_daily(df_hourly)

    df = df_daily.join(df_hourly_daily, how="left")
    return df


if __name__ == "__main__":
    download_historical()
