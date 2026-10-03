"""Day 2: train a diabetes risk model on CDC BRFSS data."""
import pandas as pd

DATA_PATH = "data/diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"

# Things a person can answer without a lab test
FEATURES = [
    "Age", "BMI", "HighBP", "HighChol",
    "PhysActivity", "GenHlth", "Sex", "Smoker",
]
BINARY = ["HighBP", "HighChol", "PhysActivity", "Sex", "Smoker", TARGET]

# 1. Skip lines with the wrong number of fields
df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
n_loaded = len(df)

# 2. Keep only rows with possible values (NaNs fail these checks too)
valid = (
    df[BINARY].isin([0, 1]).all(axis=1)
    & df["GenHlth"].between(1, 5)
    & df["Age"].between(1, 13)
    & df["BMI"].between(10, 100)
)
df = df[valid]
print("Rows loaded:", n_loaded)
print("Rows dropped as invalid:", n_loaded - len(df))

# 3. Look at what's left
print("Shape:", df.shape)
print("Diabetic share:", round(df[TARGET].mean(), 3))

X = df[FEATURES]
y = df[TARGET].astype(int)
print(X.describe().T)

# ---------- SPLIT + TRAIN ----------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# No class_weight on purpose: we want probabilities that mean something
logreg = Pipeline(
    [
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
logreg.fit(X_train, y_train)

proba = logreg.predict_proba(X_test)[:, 1]
print("\nROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print("Average predicted risk:", round(proba.mean(), 3),
      "| actual diabetic share in test set:", round(y_test.mean(), 3))

pred50 = (proba >= 0.5).astype(int)
print("Confusion matrix at 50% cutoff [[TN FP] [FN TP]]:\n",
      confusion_matrix(y_test, pred50))
print("Recall at 50% cutoff:", round(recall_score(y_test, pred50), 3))


# ---------- CHOOSE THE ALERT THRESHOLD ----------
from sklearn.metrics import precision_score, roc_curve
from sklearn.model_selection import cross_val_predict

# Pick the cutoff using TRAINING data only (out-of-fold predictions),
# so the test set stays untouched for an honest final check.
oof = cross_val_predict(logreg, X_train, y_train, cv=5, method="predict_proba")[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, oof)
threshold = thresholds[(tpr >= 0.80).argmax()]
print("\nChosen alert threshold (80% recall on training folds):",
      round(float(threshold), 3))

pred = (proba >= threshold).astype(int)
print("Confusion matrix at chosen cutoff [[TN FP] [FN TP]]:\n",
      confusion_matrix(y_test, pred))
print("Recall:", round(recall_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Share of people flagged as high risk:", round(pred.mean(), 3))

print("\ncutoff  recall  precision  flagged")
for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
    p = (proba >= t).astype(int)
    print(f"{t:.2f}    {recall_score(y_test, p):.2f}    "
          f"{precision_score(y_test, p):.2f}       {p.mean():.2f}")


    # ---------- TRAIN FINAL MODEL + SAVE ----------
import joblib
from pathlib import Path

# Refit on ALL cleaned data (the test split was only for honest evaluation)
final_model = Pipeline(
    [
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
final_model.fit(X, y)

Path("models").mkdir(exist_ok=True)
joblib.dump(
    {
        "model": final_model,
        "features": FEATURES,
        "threshold": float(threshold),
    },
    "models/diabetes_model.joblib",
)
print("\nSaved models/diabetes_model.joblib")