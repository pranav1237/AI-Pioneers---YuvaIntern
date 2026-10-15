"""Train and evaluate Linear Regression on California Housing."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": "Linear Regression",
        "dataset": "California Housing",
        "test_rows": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    print("\nRegression evaluation")
    print(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(REPORTS / "regression_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.25, s=12)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(title="California Housing: Actual vs Predicted",
           xlabel="Actual median house value (in $100,000s)",
           ylabel="Predicted median house value (in $100,000s)")
    fig.tight_layout()
    fig.savefig(REPORTS / "regression_actual_vs_predicted.png", dpi=160)
    plt.close(fig)
    joblib.dump(model, MODELS / "california_housing_linear_regression.joblib")
    print(f"Saved metrics and plot to {REPORTS}")
    print(f"Saved model to {MODELS}")


if __name__ == "__main__":
    main()
