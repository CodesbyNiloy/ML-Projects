# 🩺 Cholesterol Prediction from Lifestyle Habits (Linear Regression Baseline)

A baseline regression project that tests whether **daily-habit features** (steps, sleep, water, calories, smoking, alcohol) can predict **cholesterol** levels. The main finding: on this dataset they cannot, which is a useful result in its own right.

---

## 📌 Problem Statement

Estimate an individual's cholesterol (mg/dL) from lifestyle features using a simple linear regression, and evaluate how much signal those features carry.

---

## 📊 Dataset

- **File:** `health_lifestyle_dataset.csv`
- **Size:** 100,000 records
- **Nature:** synthetically generated health and lifestyle data (not real patient data)

| Role | Columns |
|---|---|
| Features used | `daily_steps`, `sleep_hours`, `water_intake_l`, `calories_consumed`, `smoker`, `alcohol` |
| Target | `cholesterol` |
| Other columns (not used) | `id`, `age`, `gender`, `bmi`, `resting_hr`, `systolic_bp`, `diastolic_bp`, `family_history`, `disease_risk` |

---

## 🔄 Pipeline

| Step | Detail |
|---|---|
| 1. Load | Read CSV with pandas |
| 2. Feature selection | 6 lifestyle features, `cholesterol` as target |
| 3. Split | 80/20 train-test (`random_state=42`) |
| 4. Model | `sklearn.linear_model.LinearRegression` |
| 5. Evaluation | MAE, RMSE, R² on the test set |
| 6. Inference | Example prediction for a single custom profile |

---

## 📈 Results

| Metric | Test |
|---|---|
| MAE | 37.59 |
| RMSE | 43.33 |
| R² | ≈ 0.00002 |

**Intercept:** 224.22

### Interpretation

- R² is effectively zero, so the model does no better than predicting the mean cholesterol (~224) for everyone.
- All coefficients are near zero (e.g. `alcohol` ≈ -0.40, `sleep_hours` ≈ 0.08), meaning none of the selected features show a linear relationship with cholesterol in this dataset.
- The example prediction (5,000 steps, 7 h sleep, 3 L water, 2,200 kcal, non-smoker, alcohol = 2) returns ~223.7, which is essentially the baseline mean.
- This is consistent with synthetic data where the target was generated independently of these features. Treat the model as a **baseline and a negative result**, not a usable predictor.

---

## 🗂️ Project Structure

```
.
├── cholestrol_prediction_model.ipynb   # Full notebook
├── health_lifestyle_dataset.csv        # Dataset
└── README.md
```

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Jupyter

---

