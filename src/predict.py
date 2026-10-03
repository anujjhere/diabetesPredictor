"""Load the saved model and score one person."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "diabetes_model.joblib"
_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
FEATURES = _bundle["features"]
THRESHOLD = _bundle["threshold"]

BINARY = ["HighBP", "HighChol", "PhysActivity", "Sex", "Smoker"]


def _validate(person: dict) -> None:
    missing = [f for f in FEATURES if f not in person]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for f in BINARY:
        if person[f] not in (0, 1):
            raise ValueError(f"{f} must be 0 or 1")
    if not 1 <= person["Age"] <= 13:
        raise ValueError("Age must be a bracket code from 1 to 13")
    if not 1 <= person["GenHlth"] <= 5:
        raise ValueError("GenHlth must be from 1 (excellent) to 5 (poor)")
    if not 10 <= person["BMI"] <= 100:
        raise ValueError("BMI must be between 10 and 100")


def predict(person: dict) -> dict:
    _validate(person)
    row = pd.DataFrame([[person[f] for f in FEATURES]], columns=FEATURES)
    risk = float(MODEL.predict_proba(row)[0, 1])
    high_risk = risk >= THRESHOLD
    if high_risk:
        message = "Your risk is higher than average. Consider getting a blood test."
    else:
        message = "Your risk is lower than the screening cutoff. Keep up healthy habits."
    return {
        "risk": round(risk, 3),
        "high_risk": high_risk,
        "message": message,
        "note": "Screening estimate only, not a diagnosis.",
    }


if __name__ == "__main__":
    higher = dict(Age=11, BMI=38, HighBP=1, HighChol=1,
                  PhysActivity=0, GenHlth=4, Sex=1, Smoker=1)
    lower = dict(Age=3, BMI=22, HighBP=0, HighChol=0,
                 PhysActivity=1, GenHlth=1, Sex=0, Smoker=0)
    print("Higher-risk person:", predict(higher))
    print("Lower-risk person: ", predict(lower))