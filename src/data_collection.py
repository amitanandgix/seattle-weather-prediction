import requests
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
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


def _fetch_from_api(url, params):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def download_historical():
    """Download historical daily weather data for Seattle and save to CSV."""
    print("Downloading historical weather data from Open-Meteo...")

    params = {
        "latitude": config.SEATTLE_LAT,
        "longitude": config.SEATTLE_LON,
        "start_date": config.HISTORICAL_START,
        "end_date": config.HISTORICAL_END,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": config.TIMEZONE,
    }

    data = _fetch_from_api(ARCHIVE_URL, params)
    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH)
    print(f"Saved {len(df)} rows to {config.RAW_DATA_PATH}")
    return df


def fetch_recent_for_prediction(days=30):
    """Fetch the last `days` of data to build features for tomorrow's prediction."""
    from datetime import date, timedelta

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    print(f"Fetching recent data ({start_date} to {end_date})...")

    params = {
        "latitude": config.SEATTLE_LAT,
        "longitude": config.SEATTLE_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": config.TIMEZONE,
    }

    time.sleep(0.5)
    data = _fetch_from_api(FORECAST_URL, params)
    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    download_historical()
