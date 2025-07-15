# rating_prediction_balanced.py

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import lightgbm as lgb
import xgboost as xgb

# 🧠 Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 📥 Load dataset
df = pd.read_csv("/Users/chaitanyaadityasinghchouhan/Desktop/meesho projects/sentimental analysis/Womens Clothing E-Commerce Reviews.csv")
df = df[['Review Text', 'Rating']].dropna()
df['Rating'] = df['Rating'].astype(int)

# 🧹 Enhanced Text Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'<.*?>', '', text)     # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    tokens = text.split()  # Use NLTK tokenizer
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

df['Cleaned_Review'] = df['Review Text'].apply(preprocess_text)

# ⚖️ Balance the Classes
min_class_size = 300
df_bal = df[df['Rating'].isin([1, 2, 3, 4, 5])].copy()

balanced = []
for rating in range(1, 6):
    rating_df = df_bal[df_bal['Rating'] == rating]
    if len(rating_df) < min_class_size:
        # Use original data if insufficient samples
        balanced.append(rating_df)
    else:
        rating_sample = resample(rating_df, replace=True, n_samples=min_class_size, random_state=42)
        balanced.append(rating_sample)

df_balanced = pd.concat(balanced)
X = df_balanced['Cleaned_Review']
y = df_balanced['Rating']

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔤 TF-IDF Vectorization with improved parameters
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # Include bigrams
    min_df=3,             # Ignore rare words
    max_df=0.8            # Ignore too common words
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🎯 Enhanced Evaluation Function
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🔍 Model: {name}")
    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix with percentages
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues', 
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5]
    )
    plt.title(f'{name} - Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_confusion_matrix.png", dpi=300)
    plt.show()
    
    return acc

# ✅ LightGBM with hyperparameter tuning
print("\n🚀 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train_vec, y_train)
acc_lgb = evaluate_model(lgb_model, "LightGBM", X_test_vec, y_test)

# ✅ XGBoost with label encoding
print("\n🚀 Training XGBoost...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train_vec, y_train_enc)

# Convert predictions back to original labels
y_pred_xgb = le.inverse_transform(xgb_model.predict(X_test_vec))
acc_xgb = evaluate_model(xgb_model, "XGBoost", X_test_vec, y_test_enc)

# 💾 Save predictions with probabilities
xgb_probs = xgb_model.predict_proba(X_test_vec)
output = pd.DataFrame({
    "Review": X_test.reset_index(drop=True),
    "Cleaned_Review": X_test.apply(preprocess_text).reset_index(drop=True),
    "Actual_Rating": y_test.reset_index(drop=True),
    "Predicted_Rating": y_pred_xgb,
})

# Add probability columns for each rating
for i, rating in enumerate(le.classes_):
    output[f"Prob_{rating}"] = xgb_probs[:, i]

output.to_csv("Rating_Predictions.csv", index=False)
print("✅ Predictions saved to 'Rating_Predictions.csv'")

# 📊 Model Accuracy Comparison
plt.figure(figsize=(10, 6))
models = ['LightGBM', 'XGBoost']
accuracies = [acc_lgb, acc_xgb]

ax = sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title("Model Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, acc in enumerate(accuracies):
    ax.text(i, acc + 0.02, f'{acc:.4f}', 
            ha='center', 
            fontsize=12,
            weight='bold')

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()

# 📈 Rating Distribution Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Actual_Rating', data=output, palette='Set2')
plt.title('Actual Rating Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='Predicted_Rating', data=output, palette='Set2')
plt.title('Predicted Rating Distribution')

plt.tight_layout()
plt.savefig("rating_distributions.png", dpi=300)
plt.show()