# Diabetes Risk Screener

A machine learning project that estimates a person's **risk of diabetes** from health data and tells them whether they should get tested. It is built as a *screening* tool, not a diagnostic one: the output is a risk estimate, and a real diagnosis always needs a lab test (fasting glucose, HbA1c).

> **Disclaimer:** This is a learning project. It is not medical advice and must not be used to diagnose or treat anyone. Always consult a qualified doctor.

## Status

| Stage | State |
|---|---|
| Baseline model on the Pima dataset | Done |
| Larger, non-lab model on CDC BRFSS data | Done |
| Alert threshold ("high risk, get tested") | Done |
| Saved model + `predict()` function | Done |
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

## Stage 2: BRFSS risk model (no lab values)

### Dataset

**CDC BRFSS Diabetes Health Indicators** (`diabetes_binary_health_indicators_BRFSS2015.csv`, from Kaggle): 253,680 survey rows, 22 columns, binary target `Diabetes_binary`. Only 13.9% of respondents are positive, versus 34.9% in Pima.

*TODO: confirm on the Kaggle dataset page whether `Diabetes_binary = 1` means diabetes only or prediabetes and diabetes, and state it here.*

**Features used (8):** Age (13 bracket codes, not years), BMI, HighBP, HighChol, PhysActivity, GenHlth (1 excellent to 5 poor), Sex, Smoker. These are all questions a person can answer without a lab test.

**Deliberately left out:** Income and Education (socioeconomic proxies that a health tool should not score people on), and CholCheck (not something a user would meaningfully answer).

**Data cleaning:** 3 corrupt rows were removed out of 253,680. One line had the wrong number of fields and was skipped when loading; two more were dropped by a validity check (one with a missing value, one with `10.0` in a yes/no column). 253,677 rows remain.

### Method

1. Stratified 80/20 split (`random_state=42`): 50,736 test rows, 7,069 of them positive
2. Pipeline: standard scaling, then logistic regression, trained **without** class weights so the predicted probabilities are meaningful as risk estimates
3. Alert threshold chosen separately, using out-of-fold predictions on the **training** data only (5-fold), so the test set stays untouched for an honest check
4. Final model refit on all cleaned data and saved with `joblib` together with the feature list and the threshold

### Results

- **ROC-AUC: 0.817** on the test set
- **Calibration:** average predicted risk 0.140 vs. actual positive share 0.139

**Why the cutoff matters.** At the default 50% cutoff, the model flagged only 981 of the 7,069 positive people (recall 0.139) and missed 6,088. The model ranks people reasonably well, but almost nobody reaches 50% when only 14% of people are positive.

**Chosen alert threshold: 0.124**, the value that catches about 80% of positive cases on the training folds. On the test set:

| | Value |
|---|---|
| Recall | 0.807 (5,703 caught, 1,366 missed) |
| Precision | 0.29 |
| False alarms | 13,929 |
| Share of people flagged | 38.7% |

Training recall (80%) and test recall (80.7%) agree, which suggests the threshold is not overfitted.

**The trade-off at other cutoffs (test set):**

| Cutoff | Recall | Precision | Flagged |
|---|---|---|---|
| 0.10 | 0.86 | 0.27 | 45% |
| 0.124 (chosen) | 0.81 | 0.29 | 39% |
| 0.15 | 0.75 | 0.31 | 33% |
| 0.20 | 0.64 | 0.36 | 25% |
| 0.25 | 0.53 | 0.39 | 19% |
| 0.30 | 0.43 | 0.43 | 14% |

### Findings

- **Probabilities and alerts are separate decisions.** The model produces a calibrated risk; the cutoff is a policy choice about how many missed cases are acceptable versus how many unnecessary tests.
- **Precision is modest.** About 3 in 10 flagged people are positive, roughly twice the base rate of 14%. That is reasonable for a cheap first-pass screen, but the output must say "consider a blood test", never "you have diabetes".
- **No lab values needed.** ROC-AUC 0.817 from eight self-reported answers is close to the cross-validated score of the Pima model that used Glucose directly.

### Limitations

- BRFSS is self-reported survey data from US adults (2015), so results may not carry over to other populations.
- A single linear model with 8 features. A gradient-boosted model has not been tried yet.
- ROC-AUC of about 0.82 means useful for ranking groups, not reliable for any one person.
- The saved model file is git-ignored; run `python src/train.py` to create it.

## Roadmap

1. ~~BRFSS model~~ done (logistic regression; gradient boosting is a possible later comparison)
2. ~~Calibrated risk + alert threshold~~ done
3. ~~Save the model with `joblib` and write `predict(person)`~~ done
4. **FastAPI backend** exposing the prediction.
5. **Web form** (HTML/CSS/JS) that calls the API and shows the risk result with a clear disclaimer.

## Project structure

```
diabetesPrediction/
├── data/                 # datasets (large files are git-ignored)
├── src/
│   ├── day1_pima_baseline.py   # Pima baseline experiments
│   ├── train.py                # BRFSS model, threshold, saves the model
│   └── predict.py              # predict(person) -> risk, high_risk, message
├── models/               # saved model (git-ignored, created by train.py)
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

**BRFSS model:** download `diabetes_binary_health_indicators_BRFSS2015.csv` from Kaggle into `data/`, then:

```bash
python src/train.py      # trains, picks the threshold, saves models/diabetes_model.joblib
python src/predict.py    # scores two made-up example people
```

`train.py` must be run first, because the model file is not stored in the repository.

## Related work

*TODO: add one or two papers on Pima/BRFSS diabetes prediction and compare their reported accuracy or AUC with the numbers above.*

## Author

Built by Anuj Vishwakarma as a first-year IT student project.