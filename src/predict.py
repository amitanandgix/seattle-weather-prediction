import joblib
import json
import os
import sys
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_collection import fetch_recent_for_prediction
from src.feature_engineering import build_features, get_feature_columns


def predict_tomorrow():
    """Fetch recent + forecast data, build features, and predict the next 4 days."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run train.py first."
        )

    model    = joblib.load(config.MODEL_PATH)
    features = get_feature_columns()

    # Fetch historical context + 3 days of forecast so we can predict 4 days out
    df_recent   = fetch_recent_for_prediction(days=40, forecast_days=5)
    df_features = build_features(df_recent, is_prediction=True)

    if df_features.empty:
        raise ValueError("Not enough recent data to build features.")

    # Select the 4 rows starting from today; each predicts the next day (today+1 … today+4)
    today_ts = pd.Timestamp(date.today())
    base_dates = [today_ts + timedelta(days=i) for i in range(4)]
    prediction_rows = df_features[features].reindex(base_dates).dropna(how="all")

    results = []
    for i, (idx, row) in enumerate(prediction_rows.iterrows()):
        prediction_date = idx.date() + timedelta(days=1)
        prob            = model.predict_proba(row.to_frame().T)[0][1]
        prediction      = int(prob >= 0.5)

        # Look up forecast high/low for the prediction date from raw fetched data
        pred_ts = str(prediction_date)
        temp_max = round(float(df_recent.loc[pred_ts, "temperature_2m_max"]), 1) if pred_ts in df_recent.index.astype(str) else None
        temp_min = round(float(df_recent.loc[pred_ts, "temperature_2m_min"]), 1) if pred_ts in df_recent.index.astype(str) else None

        def to_f(c):
            return round(c * 9 / 5 + 32, 1) if c is not None else None

        results.append({
            "prediction_date":      str(prediction_date),
            "rain_tomorrow":        bool(prediction),
            "rain_probability_pct": round(float(prob) * 100, 1),
            "temp_high_c":          temp_max,
            "temp_high_f":          to_f(temp_max),
            "temp_low_c":           temp_min,
            "temp_low_f":           to_f(temp_min),
            "model":                "Ensemble (XGB+HGBT+RF)",
            "based_on_data_up_to":  str(idx.date()),
        })

    os.makedirs("data/predictions", exist_ok=True)
    with open("data/predictions/latest_prediction.json", "w") as f:
        json.dump(results, f, indent=2)

    C0, C1, C2, C3, C4 = 14, 8, 12, 18, 18
    sep = "+" + "+".join("-" * (w + 2) for w in [C0, C1, C2, C3, C4]) + "+"

    print()
    print(f"  Seattle Weather Prediction - 4-Day Forecast")
    print("  " + sep)
    print(f"  | {'Date':<{C0}} | {'Rain?':<{C1}} | {'Confidence':>{C2}} | {'High':^{C3}} | {'Low':^{C4}} |")
    print("  " + sep)
    for r in results:
        label    = "YES" if r["rain_tomorrow"] else "NO"
        conf     = f"{r['rain_probability_pct']}%"
        date_str = pd.Timestamp(r["prediction_date"]).strftime("%b %d, %Y")
        high_str = f"{r['temp_high_c']} C / {r['temp_high_f']} F" if r["temp_high_c"] is not None else "N/A"
        low_str  = f"{r['temp_low_c']} C / {r['temp_low_f']} F"   if r["temp_low_c"]  is not None else "N/A"
        print(f"  | {date_str:<{C0}} | {label:<{C1}} | {conf:>{C2}} | {high_str:^{C3}} | {low_str:^{C4}} |")
    print("  " + sep)
    print(f"  Model: Ensemble (XGB+HGBT+RF)")

    return results


if __name__ == "__main__":
    predict_tomorrow()
