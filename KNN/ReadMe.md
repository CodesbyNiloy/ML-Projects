# 🎯 K-Nearest Neighbors Classification

A binary classification project using **K-Nearest Neighbors (KNN)** on an anonymized dataset ("Classified Data") with 10 hidden-name features. It covers feature standardization, exploratory pair plots, baseline modelling with k=1, and choosing k using 10-fold cross-validation.

---

## 📌 Problem Statement

Given 10 anonymized numeric features, predict a binary `TARGET CLASS` (0 or 1) for new observations using distance-based classification.

---

## 📊 Dataset

- **File:** `Classified Data` (CSV, first column used as index)
- **Size:** ~1,000 rows (the 30% test split contains 300 samples)
- **Features (10):** `WTT`, `PTI`, `EQW`, `SBI`, `LQE`, `QWG`, `FDJ`, `PJF`, `HQE`, `NXJ` (real names hidden)
- **Target:** `TARGET CLASS` (balanced: roughly 50% / 50%)

---

## 🔄 Pipeline

| Step | Detail |
|---|---|
| 1. Load | Read CSV with `index_col=0` |
| 2. Standardize | `StandardScaler` on all features (KNN is distance-based, so scale matters) |
| 3. EDA | Seaborn pairplot colored by `TARGET CLASS` |
| 4. Split | 70/30 train-test |
| 5. Baseline | `KNeighborsClassifier(n_neighbors=1)` |
| 6. Choose k | 10-fold cross-validation for k = 1 to 39, plotted as accuracy vs k |
| 7. Final model | `KNeighborsClassifier(n_neighbors=23)` |

---

## 📈 Results

Evaluated on the 300-sample test set:

| Model | Accuracy | Precision (0 / 1) | Recall (0 / 1) | F1 (0 / 1) |
|---|---|---|---|---|
| KNN, k = 1 | 0.90 | 0.91 / 0.89 | 0.87 / 0.92 | 0.89 / 0.90 |
| **KNN, k = 23** | **0.95** | 0.96 / 0.93 | 0.92 / 0.97 | 0.94 / 0.95 |

**Confusion matrix, k = 23**

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 132 | 11 |
| **Actual 1** | 5 | 152 |

Increasing k from 1 to 23 raised accuracy by about 5 points. A larger k smooths the decision boundary and reduces sensitivity to noisy neighbours, which is why k=1 overfits relative to k=23. Cross-validation showed performance levelling off beyond k ≈ 23.

> The split uses no `random_state`, so exact numbers vary slightly between runs (e.g. k=1 gave 0.91 on one run and 0.90 on another).

---

## 🗂️ Project Structure

```
.
├── KNN.ipynb          # Full notebook
├── Classified Data    # Dataset (CSV)
└── README.md

```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · seaborn · Matplotlib



