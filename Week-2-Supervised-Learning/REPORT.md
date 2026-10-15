# Project Report: Supervised Machine Learning

## Objective
Demonstrate supervised learning for regression and binary classification, following the Week 2 assignment: predict house values, predict Titanic survival, evaluate models, and compare classifier performance.

## Datasets
1. **California Housing:** loaded with scikit-learn. The target is median house value, in units of $100,000.
2. **Titanic:** public passenger survival dataset. Features include passenger class, sex, age, siblings/spouses aboard, parents/children aboard, fare, and embarkation port. Identifying fields are excluded.

## Workflow
- Load data and define features and target.
- Split into training and test sets with a fixed random seed.
- Impute missing values and scale numeric features; one-hot encode categorical features.
- Fit models on training data and evaluate on held-out data.
- Save metrics, plots, and fitted models locally.

## Models
- Linear Regression for house-value estimation.
- Logistic Regression, Decision Tree, and K-Nearest Neighbors for Titanic survival classification.
- Decision Trees with multiple max_depth settings for a basic hyperparameter comparison.

## Evaluation
For regression, MAE, RMSE, and R² are calculated and actual-vs-predicted values are plotted. Lower MAE/RMSE indicate smaller prediction errors; R² summarizes explained target variance.

For classification, accuracy, precision, recall, and F1-score are calculated. A confusion matrix is plotted and Decision Tree depth settings are compared.

## Results
Run both scripts to generate metrics for the downloaded datasets and installed dependency versions. Results are saved in reports/. Numerical results are intentionally not pre-filled because they should come from an actual run rather than fabricated values.

## Conclusion
The project provides reproducible baseline experiments. Performance depends on the data split and preprocessing choices. Future work could add cross-validation, a wider hyperparameter search, and a separate validation set.
