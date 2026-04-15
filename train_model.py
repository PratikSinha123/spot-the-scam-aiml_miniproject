import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================
# 1. LOAD DATA
# ==============================
train_df = pd.read_csv("data/processed_train.csv")

# Fill missing values safely
train_df["title"] = train_df["title"].fillna("")
train_df["description"] = train_df["description"].fillna("")

# Combine text
X = train_df[["title", "description"]]
y = train_df["fraudulent"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ==============================
# 2. CUSTOM TEXT COMBINER
# ==============================
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X["title"] + " " + X["description"]).values


# ==============================
# 3. CUSTOM FRAUD SIGNAL FEATURES
# ==============================
class FraudKeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = [
            "registration fee",
            "fee required",
            "pay to start",
            "refund later",
            "whatsapp",
            "contact immediately",
            "urgent hiring",
            "limited slots",
            "no experience required",
            "work from home",
            "instant payment",
            "quick money",
            "earn money fast",
            "high salary",
            "simple tasks",
            "data entry",
            "form filling"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = (X["title"].fillna("") + " " + X["description"].fillna("")).str.lower()

        features = []
        for text in texts:
            row = []

            # Total suspicious keyword matches
            keyword_count = sum(1 for kw in self.keywords if kw in text)
            row.append(keyword_count)

            # Strong scam indicators
            row.append(int("registration fee" in text))
            row.append(int("refund later" in text))
            row.append(int("whatsapp" in text))
            row.append(int("urgent" in text or "hurry" in text))
            row.append(int("no experience" in text))
            row.append(int("work from home" in text))
            row.append(int("high salary" in text))
            row.append(int("limited slots" in text))

            # Count currency mentions / money-like patterns
            money_pattern = r"(₹|\$|rs\.?|rupees?)\s?\d+"
            row.append(len(re.findall(money_pattern, text)))

            features.append(row)

        return np.array(features, dtype=float)


# ==============================
# 4. FEATURE UNION
# ==============================
features = FeatureUnion([
    ("tfidf", Pipeline([
        ("text", TextCombiner()),
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True
        ))
    ])),
    ("fraud_flags", FraudKeywordFeatures())
])

# ==============================
# 5. MODEL PIPELINE
# ==============================
pipeline = Pipeline([
    ("features", features),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,  # Removed depth limit to allow deep tree growth on textual features
        min_samples_leaf=1,
        class_weight="balanced_subsample",  # Auto-balances based on actual samples
        random_state=42,
        n_jobs=-1
    ))
])

# ==============================
# 6. TRAIN
# ==============================
print("Initiating training - this may take a moment with enhanced parameters...")
pipeline.fit(X_train, y_train)

# ==============================
# 7. VALIDATE
# ==============================
y_pred = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred)

print("\n--- NEW ENHANCED MODEL METRICS ---")
print("F1 Score:", round(f1, 4))
print(classification_report(y_val, y_pred))

# ==============================
# 8. SAVE MODEL
# ==============================
os.makedirs("models", exist_ok=True)
model_path = "models/model.pkl"
joblib.dump(pipeline, model_path)

print(f"Model successfully optimized and saved at: {model_path}")
