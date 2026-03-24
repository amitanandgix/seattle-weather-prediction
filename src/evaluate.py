import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import get_feature_columns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def generate_report():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col="date", parse_dates=True)
    model = joblib.load(config.MODEL_PATH)
    features = get_feature_columns()

    test_mask = df.index >= config.TRAIN_CUTOFF_DATE
    X_test = df[features][test_mask]
    y_test = df[config.TARGET_COLUMN][test_mask]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost (Test Set)")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: reports/confusion_matrix.png")

    # 2. Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importances — XGBoost")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: reports/feature_importance.png")

    # 3. Monthly Accuracy
    results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=X_test.index)
    results_df["correct"] = (results_df["y_true"] == results_df["y_pred"]).astype(int)
    results_df["month"] = results_df.index.month
    monthly_acc = results_df.groupby("month")["correct"].mean()

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(1, 13), [monthly_acc.get(m, 0) for m in range(1, 13)], color="teal")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Monthly Accuracy — XGBoost (Test Set)")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/monthly_accuracy.png", dpi=150)
    plt.close()
    print("Saved: reports/monthly_accuracy.png")

    # 4. SHAP summary (optional)
    if SHAP_AVAILABLE:
        print("Generating SHAP summary plot (this may take a moment)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: reports/shap_summary.png")
    else:
        print("shap not installed — skipping SHAP plot")

    print("\nAll reports saved to /reports/")


if __name__ == "__main__":
    generate_report()
