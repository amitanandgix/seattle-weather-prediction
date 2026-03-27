import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import get_feature_columns


def walk_forward_report(df, features):
    """Print year-by-year accuracy using an expanding training window."""
    print("\n--- Walk-Forward Validation (expanding window) ---")
    years = sorted(df.index.year.unique())
    start_test_idx = max(5, len(years) // 2)

    for test_year in years[start_test_idx:]:
        train_mask = df.index.year < test_year
        test_mask  = df.index.year == test_year
        if test_mask.sum() < 30:
            continue

        X_tr, y_tr = df[features][train_mask], df[config.TARGET_COLUMN][train_mask]
        X_te, y_te = df[features][test_mask],  df[config.TARGET_COLUMN][test_mask]

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        print(f"  {test_year}: accuracy={acc:.3f}  f1={f1:.3f}  (n={len(y_te)})")


def train():
    print("Loading processed data...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col="date", parse_dates=True)

    features = get_feature_columns()
    X = df[features]
    y = df[config.TARGET_COLUMN]

    train_mask = df.index < config.TRAIN_CUTOFF_DATE
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Rain days in train: {y_train.sum()} / {len(y_train)} ({100*y_train.mean():.1f}%)")
    print(f"Rain days in test:  {y_test.sum()} / {len(y_test)} ({100*y_test.mean():.1f}%)")

    walk_forward_report(df[train_mask], features)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # --- Tune XGBoost ---
    print("\nTuning XGBoost hyperparameters (RandomizedSearchCV, n_iter=10)...")
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_param_dist = {
        "n_estimators":     [200, 300, 500],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [3, 4, 5, 6],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        param_distributions=xgb_param_dist,
        n_iter=10,
        scoring="f1",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X_train, y_train)
    best_params = xgb_search.best_params_
    print(f"Best XGBoost params: {best_params}")

    # --- Build ensemble ---
    print("\nTraining ensemble (XGBoost + HistGradientBoosting + RandomForest)...")
    xgb = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    hgbt = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("hgbt", hgbt), ("rf", rf)],
        voting="soft",
        n_jobs=1,
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    print(f"\n--- Test Results (Ensemble) ---")
    print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    return ensemble, X_test, y_test


if __name__ == "__main__":
    train()
