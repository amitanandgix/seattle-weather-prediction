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
    """Fetch today's data, build features, and predict tomorrow's rain."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run train.py first."
        )

    model = joblib.load(config.MODEL_PATH)
    features = get_feature_columns()

    # Fetch recent data to build lag/rolling features
    df_recent = fetch_recent_for_prediction(days=40)
    df_features = build_features(df_recent, is_prediction=True)

    if df_features.empty:
        raise ValueError("Not enough recent data to build features.")

    # Use the latest row (today)
    latest = df_features[features].iloc[[-1]]
    today = df_features.index[-1].date()
    tomorrow = today + timedelta(days=1)

    prob = model.predict_proba(latest)[0][1]  # probability of rain
    prediction = int(prob >= 0.5)

    result = {
        "prediction_date": str(tomorrow),
        "rain_tomorrow": bool(prediction),
        "rain_probability_pct": round(float(prob) * 100, 1),
        "model": "XGBoost",
        "based_on_data_up_to": str(today),
    }

    os.makedirs("data/predictions", exist_ok=True)
    with open("data/predictions/latest_prediction.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 45)
    print(f"  Seattle Weather Prediction")
    print(f"  Date: {tomorrow}")
    print("=" * 45)
    print(f"  Rain tomorrow : {'YES' if prediction else 'NO'}")
    print(f"  Confidence    : {prob*100:.1f}%")
    print(f"  Model         : XGBoost")
    print("=" * 45)

    return result


if __name__ == "__main__":
    predict_tomorrow()
