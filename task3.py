# ============================================================
# WEEK 3 TASK: CLUSTERING AND MODEL EVALUATION
# ============================================================
#
# Objectives:
# 1. Perform K-Means and Hierarchical Clustering
# 2. Apply PCA for dimensionality reduction
# 3. Evaluate clustering using Silhouette Score
# 4. Train classification models
# 5. Perform Cross-Validation
# 6. Generate Confusion Matrix
# 7. Calculate Precision, Recall, F1-Score and ROC-AUC
# 8. Perform Hyperparameter Tuning
# 9. Compare model performance
#
# Dataset:
# - Breast Cancer Wisconsin Dataset
# - Real dataset available through scikit-learn
#
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Classification
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


# -----------------------------
# 2. LOAD REAL DATASET
# -----------------------------

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("\nDataset loaded successfully.")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nTarget classes:")
print(data.target_names)

print("\nFirst 5 rows:")
print(X.head())


# -----------------------------
# 3. BASIC DATA INFORMATION
# -----------------------------

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset shape:", X.shape)

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nClass distribution:")
print(y.value_counts())

print("\nStatistical summary:")
print(X.describe())


# -----------------------------
# 4. FEATURE SCALING
# -----------------------------

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Features standardized successfully.")


# ============================================================
# PART A: CLUSTERING
# ============================================================

# -----------------------------
# 5. K-MEANS CLUSTERING
# -----------------------------

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

silhouette_scores = []

k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

    print(f"K = {k} | Silhouette Score = {score:.4f}")


# Find best K based on silhouette score
best_k = k_values[np.argmax(silhouette_scores)]

print("\nBest number of clusters:", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print("Final K-Means Silhouette Score:",
      round(kmeans_score, 4))


# -----------------------------
# 6. PLOT K-MEANS SILHOUETTE
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("K-Means: Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 7. HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL / AGGLOMERATIVE CLUSTERING")
print("=" * 70)

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print(
    "Hierarchical Clustering Silhouette Score:",
    round(hierarchical_score, 4)
)


# -----------------------------
# 8. COMPARE CLUSTERING MODELS
# -----------------------------

clustering_results = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical Clustering"
    ],
    "Silhouette Score": [
        kmeans_score,
        hierarchical_score
    ]
})

print("\n" + "=" * 70)
print("CLUSTERING MODEL COMPARISON")
print("=" * 70)

print(clustering_results)


# ============================================================
# 9. PCA DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 70)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 70)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print(
    "Explained variance by PC1:",
    round(explained_variance[0], 4)
)

print(
    "Explained variance by PC2:",
    round(explained_variance[1], 4)
)

print(
    "Total explained variance:",
    round(explained_variance.sum(), 4)
)


# -----------------------------
# 10. VISUALIZE K-MEANS WITH PCA
# -----------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    alpha=0.7
)

plt.title("K-Means Clustering After PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()


# -----------------------------
# 11. VISUALIZE HIERARCHICAL CLUSTERS
# -----------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    alpha=0.7
)

plt.title("Hierarchical Clustering After PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# PART B: SUPERVISED MODEL EVALUATION
# ============================================================

# -----------------------------
# 12. TRAIN-TEST SPLIT
# -----------------------------

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 13. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


# -----------------------------
# 14. LOGISTIC REGRESSION METRICS
# -----------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print("\nAccuracy:",
      round(logistic_accuracy, 4))

print("Precision:",
      round(logistic_precision, 4))

print("Recall:",
      round(logistic_recall, 4))

print("F1-Score:",
      round(logistic_f1, 4))

print("ROC-AUC:",
      round(logistic_auc, 4))


# -----------------------------
# 15. CONFUSION MATRIX
# -----------------------------

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=data.target_names
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_predictions,
    display_labels=data.target_names
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 16. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(X_test)

rf_probabilities = random_forest.predict_proba(
    X_test
)[:, 1]


# -----------------------------
# 17. RANDOM FOREST METRICS
# -----------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions
)

rf_recall = recall_score(
    y_test,
    rf_predictions
)

rf_f1 = f1_score(
    y_test,
    rf_predictions
)

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print("\nAccuracy:",
      round(rf_accuracy, 4))

print("Precision:",
      round(rf_precision, 4))

print("Recall:",
      round(rf_recall, 4))

print("F1-Score:",
      round(rf_f1, 4))

print("ROC-AUC:",
      round(rf_auc, 4))


# -----------------------------
# 18. RANDOM FOREST CONFUSION MATRIX
# -----------------------------

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=data.target_names
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=data.target_names
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 19. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Logistic Regression CV
logistic_cv_scores = cross_val_score(
    logistic_model,
    X_scaled,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nLogistic Regression CV Scores:")
print(logistic_cv_scores)

print(
    "Logistic Regression Mean CV Accuracy:",
    round(logistic_cv_scores.mean(), 4)
)

print(
    "Logistic Regression CV Standard Deviation:",
    round(logistic_cv_scores.std(), 4)
)


# Random Forest CV
rf_cv_scores = cross_val_score(
    random_forest,
    X_scaled,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nRandom Forest CV Scores:")
print(rf_cv_scores)

print(
    "Random Forest Mean CV Accuracy:",
    round(rf_cv_scores.mean(), 4)
)

print(
    "Random Forest CV Standard Deviation:",
    round(rf_cv_scores.std(), 4)
)


# ============================================================
# 20. HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
print("=" * 70)

logistic_parameter_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

logistic_grid = GridSearchCV(
    LogisticRegression(
        max_iter=5000,
        random_state=42
    ),
    param_grid=logistic_parameter_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

logistic_grid.fit(
    X_train,
    y_train
)

print("\nBest Logistic Regression Parameters:")
print(logistic_grid.best_params_)

print(
    "Best Cross-Validation Accuracy:",
    round(logistic_grid.best_score_, 4)
)


# ============================================================
# 21. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 70)

rf_parameter_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid=rf_parameter_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(
    X_train,
    y_train
)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

print(
    "Best Cross-Validation Accuracy:",
    round(rf_grid.best_score_, 4)
)


# ============================================================
# 22. EVALUATE TUNED MODELS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION OF TUNED MODELS")
print("=" * 70)

best_logistic = logistic_grid.best_estimator_

best_rf = rf_grid.best_estimator_


# Tuned Logistic Regression
tuned_logistic_predictions = best_logistic.predict(
    X_test
)

tuned_logistic_probabilities = best_logistic.predict_proba(
    X_test
)[:, 1]


tuned_logistic_accuracy = accuracy_score(
    y_test,
    tuned_logistic_predictions
)

tuned_logistic_precision = precision_score(
    y_test,
    tuned_logistic_predictions
)

tuned_logistic_recall = recall_score(
    y_test,
    tuned_logistic_predictions
)

tuned_logistic_f1 = f1_score(
    y_test,
    tuned_logistic_predictions
)

tuned_logistic_auc = roc_auc_score(
    y_test,
    tuned_logistic_probabilities
)


# Tuned Random Forest
tuned_rf_predictions = best_rf.predict(
    X_test
)

tuned_rf_probabilities = best_rf.predict_proba(
    X_test
)[:, 1]


tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_probabilities
)


# ============================================================
# 23. FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION REPORT")
print("=" * 70)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Tuned Logistic Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        tuned_logistic_accuracy,
        rf_accuracy,
        tuned_rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        tuned_logistic_precision,
        rf_precision,
        tuned_rf_precision
    ],
    "Recall": [
        logistic_recall,
        tuned_logistic_recall,
        rf_recall,
        tuned_rf_recall
    ],
    "F1-Score": [
        logistic_f1,
        tuned_logistic_f1,
        rf_f1,
        tuned_rf_f1
    ],
    "ROC-AUC": [
        logistic_auc,
        tuned_logistic_auc,
        rf_auc,
        tuned_rf_auc
    ]
})

print(
    results.round(4).to_string(index=False)
)


# ============================================================
# 24. ROC CURVE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ROC-AUC CURVES")
print("=" * 70)

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_probabilities
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_probabilities
)

fpr_tuned_logistic, tpr_tuned_logistic, _ = roc_curve(
    y_test,
    tuned_logistic_probabilities
)

fpr_tuned_rf, tpr_tuned_rf, _ = roc_curve(
    y_test,
    tuned_rf_probabilities
)


plt.figure(figsize=(9, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    fpr_tuned_logistic,
    tpr_tuned_logistic,
    label=f"Tuned Logistic Regression (AUC = {tuned_logistic_auc:.3f})"
)

plt.plot(
    fpr_tuned_rf,
    tpr_tuned_rf,
    label=f"Tuned Random Forest (AUC = {tuned_rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 25. PCA EXPLAINED VARIANCE
# ============================================================

print("\n" + "=" * 70)
print("PCA EXPLAINED VARIANCE")
print("=" * 70)

pca_full = PCA()

pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title("PCA Cumulative Explained Variance")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 26. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": best_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    feature_importance.head(10).to_string(index=False)
)


plt.figure(figsize=(10, 6))

top_features = feature_importance.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Top 10 Random Forest Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# 27. SAVE RESULTS TO CSV
# ============================================================

results.to_csv(
    "model_evaluation_results.csv",
    index=False
)

clustering_results.to_csv(
    "clustering_results.csv",
    index=False
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("1. model_evaluation_results.csv")
print("2. clustering_results.csv")
print("3. feature_importance.csv")


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print("""
This project covered:

1. Real-world dataset loading
2. Data exploration
3. Feature standardization
4. K-Means clustering
5. Hierarchical clustering
6. Silhouette score evaluation
7. PCA dimensionality reduction
8. PCA visualization
9. Logistic Regression
10. Random Forest Classification
11. Confusion Matrix
12. Precision
13. Recall
14. F1-Score
15. ROC-AUC
16. 5-Fold Cross-Validation
17. Hyperparameter tuning using GridSearchCV
18. ROC Curve comparison
19. Random Forest feature importance
20. Model evaluation report generation

All tasks have been completed successfully.
""")
