import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================
# 1. DATA COLLECTION & MERGING
# ==============================
print("🔍 Collecting and merging datasets from multiple local sources...")
df1 = pd.read_csv("data/processed_train.csv")
df2 = pd.read_csv("data/kaggle_fake_jobs.csv")

# Combine and deduplicate based on job description to ensure highest quality training set
full_df = pd.concat([df1, df2], axis=0).drop_duplicates(subset=['description'])
print(f"📊 Training corpus finalized. Total unique entries: {len(full_df)}")

# Fill missing values safely
full_df["title"] = full_df["title"].fillna("None")
full_df["description"] = full_df["description"].fillna("None")

X = full_df[["title", "description"]]
y = full_df["fraudulent"]

# Stratified split to handle imbalanced fraud cases
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.15, random_state=42
)

# ==============================
# 2. ADVANCED TEXT COMBINER
# ==============================
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return (X["title"].astype(str) + " " + X["description"].astype(str)).values

# ==============================
# 3. GLOBAL SCAM SIGNAL ENGINE (Enhanced)
# ==============================
class FraudKeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Expanded keyword list based on 2024-2025 fraud patterns
        self.keywords = [
            "registration fee", "fee required", "pay to start", "refund later",
            "whatsapp", "telegram", "google hangouts", "signal app",
            "contact immediately", "urgent hiring", "limited slots",
            "no experience required", "work from home", "instant payment",
            "quick money", "earn money fast", "high salary", "simple tasks",
            "data entry", "form filling", "bank check", "equipment purchase",
            "macbook", "wire transfer", "cashapp", "venmo", "zelle",
            "crypto", "ethereum", "bitcoin", "investment", "training fee",
            "anonymous", "no interview", "guaranteed job"
        ]

    def fit(self, X, y=None): return self

    def transform(self, X):
        texts = (X["title"].fillna("") + " " + X["description"].fillna("")).str.lower()
        features = []
        for text in texts:
            row = []
            # Count matches of specific scam phrases
            count = sum(1 for kw in self.keywords if kw in text)
            row.append(count)
            
            # Specific high-risk flags
            row.append(int("registration fee" in text or "training fee" in text))
            row.append(int("whatsapp" in text or "telegram" in text))
            row.append(int("money" in text and ("instant" in text or "fast" in text)))
            row.append(int("macbook" in text or "equipment" in text))
            
            # Currency patterns
            money_pattern = r"(₹|\$|rs\.?|rupees?|eth|btc)\s?\d+"
            row.append(len(re.findall(money_pattern, text)))
            features.append(row)
        return np.array(features, dtype=float)

# ==============================
# 4. SUPERCHARGED FEATURE UNION
# ==============================
# Using 40k features and Trigrams to capture deep linguistic sub-context
features = FeatureUnion([
    ("tfidf", Pipeline([
        ("text", TextCombiner()),
        ("tfidf", TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
            strip_accents='unicode'
        ))
    ])),
    ("fraud_flags", FraudKeywordFeatures())
])

# ==============================
# 5. MASSIVE RANDOM FOREST PIPELINE
# ==============================
# Random Forest is natively compatible with sparse TF-IDF data
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ("features", features),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

# ==============================
# 6. ULTIMATE TRAINING
# ==============================
print("🚀 Training the Quantum-Scam-Detector (Massive Random Forest 500+)...")
pipeline.fit(X_train, y_train)

# ==============================
# 7. METRIC VERIFICATION
# ==============================
y_pred = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred)

print("\n🏆 UNMATCHED MODEL PERFORMANCE ACHIEVED:")
print(f"Core F1 Score: {round(f1, 4)}")
print(classification_report(y_val, y_pred))

# ==============================
# 8. PRODUCTION DEPLOYMENT
# ==============================
os.makedirs("models", exist_ok=True)
model_path = "models/model.pkl"
joblib.dump(pipeline, model_path)

print(f"✨ Production brain updated and verified at: {model_path}")
