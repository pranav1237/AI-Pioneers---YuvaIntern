"""Compare Logistic Regression, Decision Tree, and KNN on Titanic survival."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TITANIC_URL)
    data = data[FEATURES + ["survived"]].dropna(subset=["survived"])
    X, y = data[FEATURES], data["survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    rows, fitted_models = [], {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
        fitted_models[name] = pipeline
        print(f"\n{name}\n{classification_report(y_test, pred, zero_division=0)}")
        if name == "Logistic Regression":
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set(title="Titanic Survival — Logistic Regression",
                   xlabel="Predicted label", ylabel="True label")
            fig.tight_layout()
            fig.savefig(REPORTS / "titanic_logistic_confusion_matrix.png", dpi=160)
            plt.close(fig)

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("\nModel comparison")
    print(results.to_string(index=False))
    results.to_csv(REPORTS / "classification_model_comparison.csv", index=False)

    depth_rows = []
    for depth in [1, 2, 3, 4, 5, 7, 10, None]:
        tree = Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=42)),
        ])
        tree.fit(X_train, y_train)
        pred = tree.predict(X_test)
        depth_rows.append({
            "max_depth": "None" if depth is None else str(depth),
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
    depth_results = pd.DataFrame(depth_rows)
    depth_results.to_csv(REPORTS / "decision_tree_depth_tuning.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_results["max_depth"], depth_results["accuracy"], marker="o", label="Accuracy")
    ax.plot(depth_results["max_depth"], depth_results["f1"], marker="s", label="F1-score")
    ax.set(title="Decision Tree: Effect of max_depth",
           xlabel="max_depth (None means unlimited)", ylabel="Score", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "decision_tree_depth_comparison.png", dpi=160)
    plt.close(fig)

    joblib.dump(fitted_models["Logistic Regression"], MODELS / "titanic_logistic_regression.joblib")
    print(f"Saved comparison tables and plots to {REPORTS}")
    print(f"Saved Logistic Regression model to {MODELS}")


if __name__ == "__main__":
    main()
