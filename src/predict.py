import joblib
import json
import os
import sys
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
    df_recent   = fetch_recent_for_prediction(days=40, forecast_days=3)
    df_features = build_features(df_recent, is_prediction=True)

    if df_features.empty:
        raise ValueError("Not enough recent data to build features.")

    # The last 4 rows cover today through today+3; predicting tomorrow through today+4
    prediction_rows = df_features[features].iloc[-4:]

    results = []
    for i, (idx, row) in enumerate(prediction_rows.iterrows()):
        prediction_date = idx.date() + timedelta(days=1)
        prob            = model.predict_proba(row.to_frame().T)[0][1]
        prediction      = int(prob >= 0.5)
        results.append({
            "prediction_date":      str(prediction_date),
            "rain_tomorrow":        bool(prediction),
            "rain_probability_pct": round(float(prob) * 100, 1),
            "model":                "Ensemble (XGB+HGBT+RF)",
            "based_on_data_up_to":  str(idx.date()),
        })

    os.makedirs("data/predictions", exist_ok=True)
    with open("data/predictions/latest_prediction.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 51)
    print(f"  Seattle Weather Prediction — 4-Day Forecast")
    print("=" * 51)
    print(f"  {'Date':<14} {'Rain?':<8} {'Confidence':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10}")
    for r in results:
        label = "YES" if r["rain_tomorrow"] else "NO"
        conf  = f"{r['rain_probability_pct']}%"
        print(f"  {r['prediction_date']:<14} {label:<8} {conf:>10}")
    print("=" * 51)
    print(f"  Model: Ensemble (XGB+HGBT+RF)")
    print("=" * 51)

    return results


if __name__ == "__main__":
    predict_tomorrow()
