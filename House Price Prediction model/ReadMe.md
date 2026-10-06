# 🏠 House Price Prediction (RFE + OLS Linear Regression)

A regression case study that predicts **house prices** from property attributes. It uses **Recursive Feature Elimination (RFE)** to shortlist features, then refines the model with **statsmodels OLS**, p-values, **VIF** checks, and residual analysis.

---

## 📌 Problem Statement

Identify which property attributes drive house prices and build an interpretable linear model to predict them.

---

## 📊 Dataset

- **File:** `Housing.csv`
- **Size:** 545 rows (436 train / 109 test after an 80/20 split)

| Type | Columns |
|---|---|
| Target | `price` |
| Numeric | `area`, `bedrooms`, `bathrooms`, `stories`, `parking` |
| Binary (yes/no) | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` |
| Categorical | `furnishingstatus` (furnished / semi-furnished / unfurnished) |

---

## 🔄 Pipeline

| Step | Detail |
|---|---|
| 1. Encode | Mapped yes/no columns to 1/0; one-hot encoded `furnishingstatus` with `drop_first=True` |
| 2. Split | 80/20 train-test (`random_state=100`) |
| 3. Scale | `MinMaxScaler` fit on train only, applied to `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `price` |
| 4. Feature selection | RFE with `LinearRegression`, keeping 10 of 13 features |
| 5. OLS model | `statsmodels.OLS` for coefficients, p-values, and confidence intervals |
| 6. Refine | Dropped `bedrooms` (p = 0.115, insignificant given other variables) and refit |
| 7. Multicollinearity | VIF check on final features |
| 8. Residuals | Histogram of error terms on training data |
| 9. Predict | Applied train-fitted scaler to the test set and predicted with the final model |

### RFE outcome

| Dropped by RFE | Kept (10) |
|---|---|
| `basement`, `semi-furnished`, `unfurnished` | `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea` |

`bedrooms` was then removed manually, leaving **9 predictors**.

---

## 📈 Results

### Final model (training set, n = 436)

| Metric | Value |
|---|---|
| R² | 0.663 |
| Adj. R² | 0.656 |
| F-statistic | 93.23 (p ≈ 6e-95) |
| AIC / BIC | -807.2 / -766.5 |

### Coefficients (on min-max scaled data)

| Feature | Coef | p-value |
|---|---|---|
| `bathrooms` | 0.316 | < 0.001 |
| `area` | 0.308 | < 0.001 |
| `stories` | 0.105 | < 0.001 |
| `hotwaterheating` | 0.080 | < 0.001 |
| `airconditioning` | 0.077 | < 0.001 |
| `parking` | 0.069 | < 0.001 |
| `prefarea` | 0.058 | < 0.001 |
| `mainroad` | 0.053 | < 0.001 |
| `guestroom` | 0.046 | < 0.001 |

Bathrooms and area are by far the strongest drivers; every retained feature is statistically significant.

### Multicollinearity (VIF)

All VIFs are below 5 (highest: `mainroad` 4.30, `area` 4.22), so multicollinearity is not a concern.

### Diagnostics

- Residuals are right-skewed (skew 0.88, kurtosis 6.09, Jarque-Bera p < 0.001), so the normality assumption is only approximately met.
- Durbin-Watson is 2.10, indicating no meaningful autocorrelation.

---

## 🗂️ Project Structure

```
.
├── House_Price_Prediction_model.ipynb   # Full notebook
├── Housing.csv                          # Dataset
└── README.md
```

---


## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · statsmodels · Matplotlib · seaborn

---
