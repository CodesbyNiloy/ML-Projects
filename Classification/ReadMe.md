# ❤️‍🩹 Disease Risk Classification (Logistic Regression Baseline)

A binary classification baseline that predicts whether an individual is **at risk of disease** (`disease_risk`) from health and lifestyle indicators. The project establishes a baseline and documents an important result: a plain logistic regression collapses to the majority class on this data.

---

## 📌 Problem Statement

Classify individuals as **0 (no disease)** or **1 (at risk)** using demographic, lifestyle, and clinical features.

---

## 📊 Dataset

- **File:** `health_lifestyle_dataset.csv`
- **Size:** 100,000 records
- **Nature:** synthetically generated health and lifestyle data (not real patient data)

| Role | Columns |
|---|---|
| Features (13) | `age`, `bmi`, `daily_steps`, `sleep_hours`, `water_intake_l`, `calories_consumed`, `smoker`, `alcohol`, `resting_hr`, `systolic_bp`, `diastolic_bp`, `cholesterol`, `family_history` |
| Target | `disease_risk` (binary) |
| Not used | `id`, `gender` |

**Class balance (test set):** 15,036 negatives vs 4,964 positives, so about 75% / 25%.

---

## 🔄 Pipeline

| Step | Detail |
|---|---|
| 1. Load | Read CSV with pandas |
| 2. Features / target | 13 numeric features, `disease_risk` as target |
| 3. Split | 80/20 stratified train-test (`random_state=42`), 80,000 train / 20,000 test |
| 4. Scale | `StandardScaler` fit on train only |
| 5. Model | `LogisticRegression(max_iter=1000)` |
| 6. Evaluate | Accuracy, precision, recall, F1, confusion matrix, classification report |
| 7. Inference | Example prediction for a custom profile |

---

## 📈 Results

| Metric | Test |
|---|---|
| Accuracy | 0.7518 |
| Precision (class 1) | 0.00 |
| Recall (class 1) | 0.00 |
| F1 (class 1) | 0.00 |

**Confusion matrix**

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 15,036 | 0 |
| **Actual 1** | 4,964 | 0 |

### Interpretation

- The model predicts **class 0 for every sample**. Its 75.18% accuracy is exactly the share of negatives in the test set, so it is not learning anything beyond the class prior.
- Precision, recall, and F1 for the "at risk" class are all zero, which makes the model useless for the actual goal of identifying at-risk individuals.
- This is a clear case of why **accuracy alone is misleading on imbalanced data**.
- The example profile (age 45, BMI 27, 5,000 steps, 8 h sleep, 3 L water, 5,000 kcal, non-smoker, cholesterol 210, family history) is predicted as 0, which is consistent with the model always predicting the majority class.
- Since the data is synthetic, the features may carry little or no signal for the target. A linear model that cannot beat the majority baseline, even with class weighting, would suggest this.

---

## 🗂️ Project Structure

```
.
├── classification.ipynb            # Full notebook
├── health_lifestyle_dataset.csv    # Dataset
└── README.md
```

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Jupyter

---

