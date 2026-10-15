# Week 2 — Supervised Machine Learning

An end-to-end practice project covering regression and classification, based on the Week 2 task brief.

## Goals
- Train Linear Regression to predict California house values.
- Train Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN) on Titanic survival data.
- Evaluate regression with MAE, RMSE, and R²; classification with accuracy, precision, recall, F1-score, and a confusion matrix.
- Compare models and tune Decision Tree max_depth.
- Save evaluation plots and trained models locally.

## Structure
- src/train_regression.py
- src/train_classification.py
- requirements.txt
- REPORT.md

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/train_regression.py
python src/train_classification.py
```

The scripts fetch public datasets when needed, so an internet connection is required on the first run. Outputs are written to reports/ and models/, which are excluded from Git.

## Datasets
- California Housing: scikit-learn's public dataset loader; target is median house value in units of $100,000.
- Titanic: public CSV from the seaborn-data repository. Missing values and categorical columns are handled with scikit-learn pipelines.

## Methodology
A fixed random seed is used for reproducibility. The scripts split data into training and test sets, fit preprocessing only on training data, compare classifier metrics, and evaluate Decision Tree depth values. Results are generated locally rather than invented in advance.

This is an educational project, not a production pricing or passenger-risk system.
