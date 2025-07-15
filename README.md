# 🌟 Sentiment & Star Rating Prediction for E-Commerce Reviews

This project builds a machine learning pipeline to predict star ratings (1 to 5) from user-written product reviews on a women's clothing e-commerce platform. It uses advanced text preprocessing, TF-IDF vectorization, and machine learning models (LightGBM and XGBoost) to classify reviews based on their sentiment.

---

## 📁 Dataset

- Source: [Womens Clothing E-Commerce Reviews.csv]
- Features used:
  - `Review Text`: User-written textual review
  - `Rating`: Actual star rating given by the user (1–5)

---

## 🛠️ Techniques Used

### 🧹 Preprocessing
- Lowercasing, punctuation & digit removal
- HTML tag & URL removal
- Stopword removal and lemmatization using NLTK

### 📊 Feature Engineering
- TF-IDF vectorization with bigrams
  - `max_features=10000`
  - `ngram_range=(1, 2)`
  - `min_df=3`, `max_df=0.8`

### ⚖️ Data Balancing
- Resampled each rating class to 300 samples (5-class balance)

### 🧠 Models Trained
- **LightGBM**
  - Tuned with: `num_leaves`, `n_estimators`, `learning_rate`
- **XGBoost**
  - Tuned with: `max_depth`, `subsample`, `colsample_bytree`, `learning_rate`
  - Label encoded target variable

---

## 📈 Evaluation

- **Metrics**:
  - Accuracy
  - F1-Score
  - Confusion Matrix (normalized)
- **Accuracy Scores**:
  - LightGBM: ~45%
  - XGBoost: ~49.3%

---

## 📦 Output Files

- `lightgbm_confusion_matrix.png`
- `xgboost_confusion_matrix.png`
- `model_accuracy_comparison.png`
- `rating_distributions.png`

---

## 📈 Results

| Model     | Accuracy |
|-----------|----------|
| LightGBM  | ~45.0%   |
| XGBoost   | ~49.3%   |

> ⚠️ Note: The **accuracy is modest (~49%)** because **consecutive ratings (like 1 & 2, or 3 & 4, or 4 & 5) are highly subjective** and often indistinguishable based solely on text. For example, a user may leave a positive review but still rate 3 stars instead of 5 — introducing natural label noise.

---

## 📌 Key Insights

- **Subjective Bias**: Reviews often don’t directly reflect star ratings — a review that sounds positive may get 3 stars, creating inherent label noise.
- **Balanced Dataset** and **text preprocessing** significantly improve model performance.
- **XGBoost** performed best with nearly 50% accuracy on 5-class prediction (above baseline ~20%).

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python rating_prediction_balanced.py
