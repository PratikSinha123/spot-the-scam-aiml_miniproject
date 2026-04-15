import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler

# ==============================
# 1. ULTIMATE GLOBAL DATA MERGING
# ==============================
print("🌐 Harvesting data from all available internet-derived sources...")
sources = ["data/processed_train.csv", "data/kaggle_fake_jobs.csv", "data/extra_jobs.csv"]
dfs = []
for src in sources:
    if os.path.exists(src):
        dfs.append(pd.read_csv(src))
        print(f" [+] Loaded: {src}")

full_df = pd.concat(dfs, axis=0).drop_duplicates(subset=['description'])
print(f"📊 Global Training Matrix constructed. Unique high-fidelity samples: {len(full_df)}")

# Fill missing values
full_df = full_df.dropna(subset=['description', 'fraudulent']) # Remove rows without key data
full_df["title"] = full_df["title"].fillna("None Specified")
X = full_df[["title", "description"]]
y = full_df["fraudulent"]

# ==============================
# 2. DATA SEGMENTATION & NEURAL BALANCING
# ==============================
# Key ML Rule: SPLIT before AUGMENT to avoid data leakage
X_train_raw, X_val, y_train_raw, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("🧠 Initiating Neural Balancing on Training Set (Scientific Split)...")
ros = RandomOverSampler(sampling_strategy=0.8, random_state=42) # Balance to 80%
X_train, y_train = ros.fit_resample(X_train_raw, y_train_raw)
print(f" ✅ Scientific Training Matrix: {len(X_train)} nodes")
print(f" ✅ Validation Matrix Ready: {len(X_val)} unique nodes")

# ==============================
# 3. SUPREME FEATURE EXTRACTION (40k Nodes)
# ==============================
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return (X["title"].astype(str) + " " + X["description"].astype(str)).values

class FraudKeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = ["fee", "whatsapp", "telegram", "hangouts", "instant pay", "limited slots", "urgent hiring", "no experience", "registration", "equipment purchase", "macbook", "crypto", "bitcoin", "ethereum", "zelle", "cashapp", "venmo"]
    def fit(self, X, y=None): return self
    def transform(self, X):
        texts = (X["title"].fillna("") + " " + X["description"].fillna("")).str.lower()
        features = [[sum(1 for kw in self.keywords if kw in t), int("whatsapp" in t), int("telegram" in t)] for t in texts]
        return np.array(features, dtype=float)

features = FeatureUnion([
    ("word_tfidf", Pipeline([
        ("text", TextCombiner()),
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True))
    ])),
    ("char_tfidf", Pipeline([
        ("text", TextCombiner()),
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True))
    ])),
    ("flags", FraudKeywordFeatures())
])

# ==============================
# 4. TRIPLE-ENSEMBLE VOTING CLASSIFIER (Supreme Brain)
# ==============================
from sklearn.tree import DecisionTreeClassifier
voter = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)),
        ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ],
    voting='soft',
    n_jobs=-1
)

supreme_pipeline = Pipeline([
    ("features", features),
    ("clf", voter)
])

# ==============================
# 5. EXECUTE TRAINING
# ==============================
print("🛡️ Training the SUPREME-SCAM-DETECTOR ENSEMBLE (Hybrid Synergy)...")
supreme_pipeline.fit(X_train, y_train)

# ==============================
# 6. VERIFICATION
# ==============================
y_pred = supreme_pipeline.predict(X_val)
print("\n🔥 SUPREME MODEL METRICS ACHIEVED:")
print(classification_report(y_val, y_pred))

# ==============================
# 7. PRODUCTION UPDATE
# ==============================
model_path = "models/model.pkl"
joblib.dump(supreme_pipeline, model_path)
print(f"💎 Model updated to 'SUPREME' status at: {model_path}")
print(f"   Final Accuracy: {round(supreme_pipeline.score(X_val, y_val)*100, 2)}%")
