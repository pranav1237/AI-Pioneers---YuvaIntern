# ============================================================
# WEEK 2 TASK
# Supervised Machine Learning Models
# AI Pioneers - YuvaIntern
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Regression
from sklearn.linear_model import LinearRegression

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PART A: LINEAR REGRESSION
# California Housing Dataset
# ============================================================

print("\n" + "=" * 70)
print("PART A: LINEAR REGRESSION")
print("=" * 70)


# ------------------------------------------------------------
# 2. LOAD REAL CALIFORNIA HOUSING DATASET
# ------------------------------------------------------------

# The dataset is downloaded automatically by scikit-learn.
# No manual CSV download is required.

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

housing_df = housing.frame

print("\nCalifornia Housing Dataset:")
print(housing_df.head())

print("\nDataset Shape:")
print(housing_df.shape)

print("\nDataset Information:")
print(housing_df.info())

print("\nStatistical Summary:")
print(housing_df.describe())


# ------------------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

X_reg = housing_df.drop("MedHouseVal", axis=1)

y_reg = housing_df["MedHouseVal"]


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train_reg.shape[0])
print("Testing samples:", X_test_reg.shape[0])


# ------------------------------------------------------------
# 5. CREATE LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# ------------------------------------------------------------
# 6. TRAIN LINEAR REGRESSION MODEL
# ------------------------------------------------------------

linear_model.fit(X_train_reg, y_train_reg)


# ------------------------------------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred_reg = linear_model.predict(X_test_reg)


# ------------------------------------------------------------
# 8. EVALUATE LINEAR REGRESSION
# ------------------------------------------------------------

mae = mean_absolute_error(y_test_reg, y_pred_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)

rmse = np.sqrt(mse)

r2 = r2_score(y_test_reg, y_pred_reg)


print("\nLinear Regression Performance:")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 9. ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_reg,
    y_pred_reg,
    alpha=0.5
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()


# ============================================================
# PART B: CLASSIFICATION
# Titanic Dataset
# ============================================================

print("\n" + "=" * 70)
print("PART B: CLASSIFICATION MODELS")
print("=" * 70)


# ------------------------------------------------------------
# 10. LOAD REAL TITANIC DATASET DIRECTLY
# ------------------------------------------------------------

# No manual download required.
# Pandas reads the dataset directly from this URL.

titanic_url = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)

try:
    titanic = pd.read_csv(titanic_url)

except Exception as e:
    print("\nCould not load the Titanic dataset from the URL.")
    print("Error:", e)
    print("\nCheck your internet connection in Codespaces.")
    raise


print("\nTitanic Dataset:")
print(titanic.head())

print("\nDataset Shape:")
print(titanic.shape)

print("\nDataset Information:")
titanic.info()

print("\nMissing Values:")
print(titanic.isnull().sum())


# ------------------------------------------------------------
# 11. SELECT RELEVANT FEATURES
# ------------------------------------------------------------

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"


X = titanic[features]

y = titanic[target]


# ------------------------------------------------------------
# 12. IDENTIFY COLUMN TYPES
# ------------------------------------------------------------

numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Pclass"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# ------------------------------------------------------------
# 13. PREPROCESSING PIPELINE
# ------------------------------------------------------------

# Numerical columns:
# - Fill missing values with median
# - Standardize values

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical columns:
# - Fill missing values with most frequent value
# - One-hot encode categories

from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# Combine numerical and categorical preprocessing

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ------------------------------------------------------------
# 14. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nClassification Training Samples:", X_train.shape[0])
print("Classification Testing Samples:", X_test.shape[0])


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print("\n" + "-" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("-" * 70)


logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred
)

logistic_recall = recall_score(
    y_test,
    logistic_pred
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)


print(f"Accuracy  : {logistic_accuracy:.4f}")
print(f"Precision : {logistic_precision:.4f}")
print(f"Recall    : {logistic_recall:.4f}")
print(f"F1 Score  : {logistic_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))


# ============================================================
# MODEL 2: DECISION TREE
# ============================================================

print("\n" + "-" * 70)
print("MODEL 2: DECISION TREE")
print("-" * 70)


decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])


decision_tree_model.fit(X_train, y_train)

tree_pred = decision_tree_model.predict(X_test)


tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

tree_precision = precision_score(
    y_test,
    tree_pred
)

tree_recall = recall_score(
    y_test,
    tree_pred
)

tree_f1 = f1_score(
    y_test,
    tree_pred
)


print(f"Accuracy  : {tree_accuracy:.4f}")
print(f"Precision : {tree_precision:.4f}")
print(f"Recall    : {tree_recall:.4f}")
print(f"F1 Score  : {tree_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))


# ============================================================
# MODEL 3: RANDOM FOREST
# ============================================================

print("\n" + "-" * 70)
print("MODEL 3: RANDOM FOREST")
print("-" * 70)


random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
    )
])


random_forest_model.fit(X_train, y_train)

forest_pred = random_forest_model.predict(X_test)


forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

forest_precision = precision_score(
    y_test,
    forest_pred
)

forest_recall = recall_score(
    y_test,
    forest_pred
)

forest_f1 = f1_score(
    y_test,
    forest_pred
)


print(f"Accuracy  : {forest_accuracy:.4f}")
print(f"Precision : {forest_precision:.4f}")
print(f"Recall    : {forest_recall:.4f}")
print(f"F1 Score  : {forest_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, forest_pred))


# ============================================================
# MODEL 4: K-NEAREST NEIGHBORS
# ============================================================

print("\n" + "-" * 70)
print("MODEL 4: K-NEAREST NEIGHBORS")
print("-" * 70)


knn_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        KNeighborsClassifier(n_neighbors=5)
    )
])


knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)


knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

knn_precision = precision_score(
    y_test,
    knn_pred
)

knn_recall = recall_score(
    y_test,
    knn_pred
)

knn_f1 = f1_score(
    y_test,
    knn_pred
)


print(f"Accuracy  : {knn_accuracy:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall    : {knn_recall:.4f}")
print(f"F1 Score  : {knn_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, knn_pred))


# ============================================================
# PART C: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],

    "Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        forest_accuracy,
        knn_accuracy
    ],

    "Precision": [
        logistic_precision,
        tree_precision,
        forest_precision,
        knn_precision
    ],

    "Recall": [
        logistic_recall,
        tree_recall,
        forest_recall,
        knn_recall
    ],

    "F1 Score": [
        logistic_f1,
        tree_f1,
        forest_f1,
        knn_f1
    ]
})


print("\n")
print(results.to_string(index=False))


# ============================================================
# 15. FIND BEST CLASSIFICATION MODEL
# ============================================================

best_model = results.loc[
    results["F1 Score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST CLASSIFICATION MODEL")
print("=" * 70)

print(
    f"\nBest Model: {best_model['Model']}"
)

print(
    f"Accuracy: {best_model['Accuracy']:.4f}"
)

print(
    f"F1 Score: {best_model['F1 Score']:.4f}"
)


# ============================================================
# 16. VISUALIZE MODEL ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Classification Model Accuracy Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.ylim(0, 1)

plt.tight_layout()

plt.show()


# ============================================================
# 17. VISUALIZE F1 SCORES
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="F1 Score"
)

plt.title("Classification Model F1 Score Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("F1 Score")

plt.xticks(rotation=20)

plt.ylim(0, 1)

plt.tight_layout()

plt.show()


# ============================================================
# 18. CONFUSION MATRIX FOR BEST MODEL
# ============================================================

best_model_name = best_model["Model"]


if best_model_name == "Logistic Regression":
    best_predictions = logistic_pred

elif best_model_name == "Decision Tree":
    best_predictions = tree_pred

elif best_model_name == "Random Forest":
    best_predictions = forest_pred

else:
    best_predictions = knn_pred


cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================================
# 19. SAVE CLASSIFICATION RESULTS
# ============================================================

results.to_csv(
    "week2_model_comparison.csv",
    index=False
)


print("\nModel comparison saved as:")
print("week2_model_comparison.csv")


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 2 TASK COMPLETED")
print("=" * 70)

print("""
Models implemented:

1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbors

Datasets used:

1. California Housing Dataset
2. Titanic Dataset

The models were trained, tested, evaluated,
and compared using appropriate performance metrics.
""")

print("=" * 70)