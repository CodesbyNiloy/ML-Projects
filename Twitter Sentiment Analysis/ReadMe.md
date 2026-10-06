# 🐦 Twitter Sentiment Analysis (NLP + TF-IDF + Classical ML)

A multi-class text classification project that labels tweets as **Positive**, **Negative**, **Neutral**, or **Irrelevant**. It covers text cleaning with NLTK, TF-IDF feature extraction, and a comparison of five classical ML models.

---

## 📌 Problem Statement

Given a tweet about a topic (e.g. a game or brand), classify its sentiment into one of four classes.

---

## 📊 Dataset

- **File:** `twitter_data.csv` (no header row)
- **Size:** 74,682 tweets
- **Columns:** `ID`, `Topic`, `Sentiment`, `Text`
- **Missing values:** 686 rows with empty `Text` (handled by returning an empty string in cleaning)

| Class | Count |
|---|---|
| Negative | 22,542 |
| Positive | 20,832 |
| Neutral | 18,318 |
| Irrelevant | 12,990 |

---

## 🔄 Pipeline

| Step | Detail |
|---|---|
| 1. Load | Read CSV with explicit column names |
| 2. Clean text | Lowercase, HTML unescape, remove URLs / mentions / `#` / punctuation, tokenize (NLTK), remove stopwords and 1-char tokens, lemmatize (WordNet) |
| 3. Encode labels | `LabelEncoder` → Irrelevant=0, Negative=1, Neutral=2, Positive=3 |
| 4. Vectorize | `TfidfVectorizer` (max 10,000 features, unigrams + bigrams) |
| 5. Train & compare | Logistic Regression, Naive Bayes, Decision Tree, Random Forest (200 trees), Linear SVM |
| 6. Evaluate | Accuracy, classification report, confusion matrices, one-vs-rest ROC curves (models with `predict_proba`) |

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 0.7178 |
| Logistic Regression | 0.8008 |
| Linear SVM | 0.8549 |
| Decision Tree | 0.9541 |
| Random Forest | 0.9541 |

Random Forest was selected as the final model.


### Per-class F1 (Random Forest, same caveat)

| Class | F1 |
|---|---|
| Negative | 0.97 |
| Irrelevant | 0.96 |
| Neutral | 0.96 |
| Positive | 0.93 |

---

## 🗂️ Project Structure

```
.
├── Twitter_Sentiment_Analysis.ipynb   # Full notebook
├── twitter_data.csv                   # Dataset
└── README.md
```

---

## ⚙️ Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        

pip install pandas numpy scikit-learn nltk seaborn matplotlib jupyter
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

jupyter notebook Twitter_Sentiment_Analysis.ipynb

```

---

## 🛠️ Tech Stack

Python · pandas · NumPy · NLTK · scikit-learn · seaborn · Matplotlib

---

