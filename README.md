# Diabetes Risk Screener

A machine learning project that estimates a person's **risk of diabetes** from health data and tells them whether they should get tested. It is built as a *screening* tool, not a diagnostic one: the output is a risk estimate, and a real diagnosis always needs a lab test (fasting glucose, HbA1c).

> **Disclaimer:** This is a learning project. It is not medical advice and must not be used to diagnose or treat anyone. Always consult a qualified doctor.

## Status

| Stage | State |
|---|---|
| Baseline model on the Pima dataset | Done |
| Larger, non-lab model on CDC BRFSS data | Planned |
| Saved model + `predict()` function | Planned |
| FastAPI backend | Planned |
| Web form (HTML/CSS/JS) | Planned |

## Why a "risk screener" and not a "diabetes detector"?

Diagnosis depends on blood tests. A model trained on lab values like Glucose mostly rediscovers the diagnostic criteria doctors already use. A more useful tool uses things a person can answer without a lab (age, BMI, blood pressure, activity) and says *"your risk is high, get tested."* The BRFSS stage of this project is aimed at that.

## Dataset (Day 1)

**Pima Indians Diabetes Dataset**: 768 patients, 8 features, binary target `Outcome` (34.9% diabetic).

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

**Data quality note:** in Glucose, BloodPressure, SkinThickness, Insulin and BMI, a value of `0` is physically impossible and really means "missing". After converting zeros to missing values:

| Column | Missing |
|---|---|
| Insulin | 374 (49%) |
| SkinThickness | 227 (30%) |
| BloodPressure | 35 |
| BMI | 11 |
| Glucose | 5 |

Missing values are filled with the column median, learned **only from the training data** (inside a scikit-learn `Pipeline`) to avoid data leakage.

## Method

1. Stratified 80/20 train/test split (`random_state=42`)
2. Pipeline: median imputation, standard scaling, logistic regression
3. Evaluated with recall, precision, ROC-AUC and a confusion matrix, not just accuracy, because missing a diabetic patient is worse than a false alarm

## Results

Test set: 154 patients (54 diabetic, 100 non-diabetic).

| Experiment | Accuracy | Diabetic recall | Diabetic precision | ROC-AUC | Missed diabetics (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| Baseline (`class_weight="balanced"`) | 0.73 | 0.70 | 0.60 | 0.813 | 16 | 25 |
| No class weighting (`class_weight=None`) | 0.71 | 0.50 | 0.60 | n/a | 27 | 18 |
| Baseline without Insulin | n/a | n/a | n/a | 0.811 | n/a | n/a |

**5-fold cross-validated ROC-AUC:** `0.815, 0.800, 0.848, 0.885, 0.839` → **mean 0.837**.

### Findings

- **Class weighting matters for screening.** Without it, the model missed half of the diabetic patients (recall 0.50 vs 0.70). I kept `class_weight="balanced"`.
- **Insulin adds almost nothing here.** Dropping it moved ROC-AUC by 0.002, which is far smaller than the ~0.085 spread between cross-validation folds. With 49% of its values imputed, there is little real signal.
- **Single splits are noisy.** The test set has only 54 diabetic cases, so one split (0.813) understated the cross-validated score (0.837). Model comparisons should use cross-validation.
- **Glucose dominates.** It has the largest model weight (1.18), followed by BMI (0.71). Glucose is a lab value, so this model partly mirrors how diabetes is diagnosed. That is the motivation for the BRFSS stage.

### Limitations

- Pima is small and covers a single population (women of Pima heritage, aged 21+), so results will not generalize to everyone.
- The test set is small, so metrics have wide uncertainty.
- Accuracy around 73% means this baseline is useful as a learning exercise, not as a real screening tool.

## Roadmap

1. **BRFSS model:** train on the CDC BRFSS Diabetes Health Indicators dataset using non-lab features; compare logistic regression with a gradient-boosted model.
2. **Calibrated risk + alert threshold:** train without class weights so probabilities are meaningful, then choose a separate cutoff for "high risk, get tested".
3. **Save the model** with `joblib` and write a `predict(person)` function.
4. **FastAPI backend** exposing the prediction.
5. **Web form** (HTML/CSS/JS) that calls the API and shows the risk result with a clear disclaimer.

## Project structure

```
diabetesPrediction/
├── data/                 # datasets (large files are git-ignored)
├── src/
│   └── day1_pima_baseline.py
├── models/               # saved models (git-ignored)
├── app/                  # FastAPI backend (planned)
├── web/                  # frontend (planned)
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python src/day1_pima_baseline.py data/diabetes.csv
```

On Windows, if `pip` is not recognized, use `python -m pip install -r requirements.txt`.

The script prints dataset summaries, the confusion matrix, a classification report, ROC-AUC, feature weights and cross-validated ROC-AUC, then shows a Glucose histogram by outcome.

## Related work

*TODO: add one or two papers on Pima/BRFSS diabetes prediction and compare their reported accuracy or AUC with the numbers above.*

## Author

Built by Anuj Vishwakarma as a first-year IT student project.