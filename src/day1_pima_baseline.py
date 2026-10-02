"""
Day 1: Pima diabetes baseline
Run:  python day1_pima_baseline.py diabetes.csv
Setup: pip install pandas scikit-learn matplotlib
"""
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "diabetes.csv"

# ---------- 1. LOAD + LOOK ----------
df = pd.read_csv(path)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe().T, "\n")

# Class balance: how many diabetic (1) vs not (0)?
print("Class balance:\n", df["Outcome"].value_counts(normalize=True), "\n")

# ---------- 2. CLEAN ----------
# In this dataset a 0 in these columns is physically impossible -> it means "missing"
zero_means_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_means_missing] = df[zero_means_missing].replace(0, np.nan)
print("Missing values per column:\n", df.isna().sum(), "\n")

# ---------- 3. SPLIT ----------
X = df.drop(columns="Outcome")
y = df["Outcome"]

# stratify keeps the diabetic/non-diabetic ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. MODEL ----------
# Imputing + scaling live INSIDE the pipeline so they are learned from
# training data only (no leakage from the test set).
model = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)
model.fit(X_train, y_train)

# ---------- 5. EVALUATE ----------
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred), "\n")
print(classification_report(y_test, pred, target_names=["no diabetes", "diabetes"]))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3), "\n")

# ---------- 6. WHAT DID IT LEARN? ----------
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
print("Feature weights (bigger = pushes toward 'diabetes'):")
print(coefs.sort_values(ascending=False))

# ---------- 7. CROSS-VALIDATION ----------
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print("CV ROC-AUC:", scores.round(3), "mean:", scores.mean().round(3))

# ---------- 8. GLUCOSE HISTOGRAM ----------
import matplotlib.pyplot as plt

df[df.Outcome == 0]["Glucose"].plot.hist(alpha=0.6, bins=20, label="no diabetes")
df[df.Outcome == 1]["Glucose"].plot.hist(alpha=0.6, bins=20, label="diabetes")
plt.xlabel("Glucose")
plt.legend()
plt.show()