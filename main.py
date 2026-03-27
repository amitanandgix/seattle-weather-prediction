"""
Seattle Weather Prediction — Main Pipeline
==========================================
Usage:
  python main.py              # Full pipeline (skips download/process if data exists)
  python main.py --retrain    # Force re-download and retrain
  python main.py --predict    # Only run prediction (model must already exist)
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Seattle Weather Prediction Pipeline")
    parser.add_argument("--retrain", action="store_true", help="Force re-download and retrain")
    parser.add_argument("--predict", action="store_true", help="Only predict tomorrow's weather")
    args = parser.parse_args()

    import config
    from src.data_collection import download_historical
    from src.feature_engineering import load_and_process
    from src.train import train
    from src.predict import predict_tomorrow
    from src.evaluate import generate_report

    if args.predict:
        predict_tomorrow()
        return

    # Step 1: Download historical data
    if args.retrain or not os.path.exists(config.RAW_DATA_PATH):
        from src.data_collection import DAILY_CACHE, HOURLY_CACHE
        if args.retrain:
            for cache in [DAILY_CACHE, HOURLY_CACHE]:
                if os.path.exists(cache):
                    os.remove(cache)
        download_historical()
    else:
        print(f"Raw data already exists at {config.RAW_DATA_PATH} — skipping download.")

    # Step 2: Feature engineering
    if args.retrain or not os.path.exists(config.PROCESSED_DATA_PATH):
        load_and_process()
    else:
        print(f"Processed data already exists at {config.PROCESSED_DATA_PATH} — skipping.")

    # Step 3: Train model
    if args.retrain or not os.path.exists(config.MODEL_PATH):
        train()
    else:
        print(f"Model already exists at {config.MODEL_PATH} — skipping training.")
        print("Use --retrain to force retrain.")

    # Step 4: Evaluate
    generate_report()

    # Step 5: Predict tomorrow
    predict_tomorrow()


if __name__ == "__main__":
    main()
