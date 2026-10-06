# 🚖 Uber Ride Data Analysis (EDA + Booking Value Model)

Exploratory data analysis of 150,000 Uber bookings in the Delhi-NCR region, followed by a Random Forest baseline to predict **Booking Value**. The analysis covers data cleaning, distribution and relationship plots, and a first modelling pass.

---

## 📌 Objectives

1. Understand booking behaviour: status mix, vehicle types, payment methods, and ride values.
2. Examine the relationship between ride distance and booking value.
3. Build a baseline model to predict booking value.

---

## 📊 Dataset

- **File:** `uber data.csv`
- **Size:** 150,000 rows × 21 columns

| Group | Columns |
|---|---|
| Booking | `Date`, `Time`, `Booking ID`, `Booking Status`, `Customer ID`, `Vehicle Type` |
| Location | `Pickup Location`, `Drop Location` |
| Timing | `Avg VTAT` (vehicle arrival time), `Avg CTAT` (customer trip time) |
| Cancellations / incomplete | `Cancelled Rides by Customer`, `Reason for cancelling by Customer`, `Cancelled Rides by Driver`, `Driver Cancellation Reason`, `Incomplete Rides`, `Incomplete Rides Reason` |
| Ride outcome | `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`, `Payment Method` |

### Missing data overview

| Column | Non-null rows |
|---|---|
| `Avg VTAT` | 139,500 |
| `Avg CTAT` | 102,000 |
| `Booking Value`, `Ride Distance` | 102,000 |
| `Driver Ratings` | 93,000 |
| Customer-cancelled rides | 10,500 |
| Driver-cancelled rides | 27,000 |
| Incomplete rides | 9,000 |

Much of the missingness is structural: ride-outcome fields exist only for bookings that reached that stage, and cancellation fields only for cancelled bookings.

---

## 🔄 Workflow

| Step | Detail |
|---|---|
| 1. Clean | Stripped whitespace from headers; removed stray quotes from `Booking ID` and `Customer ID` |
| 2. Impute | Median fill for numeric columns (`Avg VTAT`, `Avg CTAT`, `Booking Value`, `Ride Distance`, ratings); `"Not Applicable"` for reason columns; mode for `Payment Method` |
| 3. Feature engineering | Combined `Date` + `Time` into a single `datetime` column |
| 4. EDA | Count plots (vehicle type, payment method), booking status distribution, vehicle popularity, booking value histogram with KDE, distance vs value regression plot, correlation heatmap |
| 5. Encode | `LabelEncoder` on all object columns |
| 6. Model | 80/20 split (`random_state=42`), `RandomForestRegressor(n_estimators=100)` targeting `Booking Value` |

---

## 🗂️ Project Structure

```
.
├── Uber_Data_Analysis__EDA_.ipynb   # Full notebook
├── uber data.csv                    # Dataset
└── README.md
```

---


## 🛠️ Tech Stack

Python · pandas · NumPy · seaborn · Matplotlib · scikit-learn

---

