import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import get_feature_columns


def train():
    print("Loading processed data...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col="date", parse_dates=True)

    features = get_feature_columns()
    X = df[features]
    y = df[config.TARGET_COLUMN]

    # Time-based split — never shuffle time series data
    train_mask = df.index < config.TRAIN_CUTOFF_DATE
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Rain days in train: {y_train.sum()} / {len(y_train)} ({100*y_train.mean():.1f}%)")
    print(f"Rain days in test:  {y_test.sum()} / {len(y_test)} ({100*y_test.mean():.1f}%)")

    # Compute class weight ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    print(f"\n--- Test Results ---")
    print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    train()
