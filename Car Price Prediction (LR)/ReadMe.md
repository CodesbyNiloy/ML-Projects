# 🚗 Car Price Prediction (Linear Regression + Streamlit)

An end-to-end machine learning project that predicts a vehicle's **MSRP** from its specs and exposes the model through an interactive **Streamlit** web app. The workflow covers data cleaning, EDA, feature importance, model training, evaluation, and deployment.

---

## 📌 Problem Statement

1. Understand which vehicle attributes drive car prices.
2. Build a model that can predict MSRP for a given configuration.

---

## 📊 Dataset

- **File:** `car_data.xlsx`
- **Size:** 1,610 rows, 17 raw columns
- **Coverage:** 7 makes (Aston Martin, Audi, BMW, Bentley, Ford, Mercedes-Benz, Nissan), model years 2023–2024

| Type | Columns used |
|---|---|
| Target | `MSRP` |
| Numeric | `Horsepower_No`, `Torque_No` (parsed from raw `Horsepower` / `Torque` strings) |
| Categorical | `Make`, `Body Size`, `Body Style`, `Engine Aspiration`, `Drivetrain`, `Transmission` |

---

## 🔄 Pipeline

| Step | What happens |
|---|---|
| 1. Load | Read `car_data.xlsx` with pandas |
| 2. Missing values | Dropped `Invoice Price`, `Cylinders`, `Highway Fuel Economy` (high null counts). Imputed `Horsepower` with the Ford mean (5 rows) and `Torque` with the overall mean (27 rows) |
| 3. Type cleaning | Stripped `$` and `,` from `MSRP`; extracted numeric horsepower and torque |
| 4. EDA | Pairplots, categorical bar plots, distributions, box/swarm plots, correlation heatmap |
| 5. Feature engineering | Dropped `index`, `Model`, `Year`, `Trim`, `Used/New Price`, raw `Horsepower`/`Torque`; one-hot encoded all categoricals (36 features) |
| 6. Feature importance | Decision tree (`entropy`, `max_depth=10`) importances, top 27 saved to Excel |
| 7. Split | 80/20 hold-out (1,288 train / 322 test, `random_state=15`) |
| 8. Model | `sklearn.linear_model.LinearRegression` |
| 9. Persist | Model saved to `linear_model.pkl`; predictions and importances exported to Excel |
| 10. Deploy | Streamlit app with sidebar inputs, feature-importance chart, and price prediction |

---

## 📈 Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.896 | 0.920 |
| RMSE | $17,422 | $16,535 |
| MAE | $10,599 | $11,090 |

Train and test scores are close, so there is no sign of overfitting on this split.

### Top features (decision-tree importance)

| Rank | Feature | Score |
|---|---|---|
| 1 | Horsepower | 0.247 |
| 2 | Make: Ford | 0.124 |
| 3 | Torque | 0.123 |
| 4 | Engine Aspiration: Turbocharged | 0.087 |
| 5 | Body Size: Large | 0.043 |

---

## 🖥️ Streamlit App

Inputs (sidebar): horsepower, torque, make, body size, body style, engine aspiration, drivetrain, transmission.
Outputs: interactive feature-importance bar chart (Plotly) and the predicted MSRP on button click.

```bash
streamlit run Regr_model_cars.py
```

---

## 🗂️ Project Structure

```
.
├── Car_Price_Prediction__LR_.ipynb   # Full analysis + model training
├── Regr_model_cars.py                # Streamlit app
├── car_data.xlsx                     # Raw dataset
├── linear_model.pkl                  # Trained model
├── feature_importance.xlsx           # Exported feature importances
├── data_with_pred.xlsx               # Dataset with model predictions
├── Pic 1.png                         # App sidebar image
├── Pic 2.png                         # App banner image
└── README.md

```

---

## ⚙️ Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pandas numpy seaborn matplotlib scikit-learn plotly streamlit openpyxl pillow
```

Run the notebook to retrain and regenerate `linear_model.pkl`, `feature_importance.xlsx`, and `data_with_pred.xlsx`, then launch the app.

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · seaborn · Matplotlib · Plotly · Streamlit

---

