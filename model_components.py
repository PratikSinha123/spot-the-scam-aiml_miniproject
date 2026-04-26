import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TEXT_COLUMNS = (
    "title",
    "description",
    "company_profile",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "salary_range",
)


class TextCombiner(BaseEstimator, TransformerMixin):
    """Combine available job-post fields into one text document."""

    def __init__(self, text_columns=TEXT_COLUMNS):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return pd.Series(X).fillna("").astype(str).values

        parts = []
        for column in self.text_columns:
            if column in X.columns:
                parts.append(X[column].fillna("").astype(str))

        if not parts:
            return pd.Series([""] * len(X)).values

        combined = parts[0]
        for part in parts[1:]:
            combined = combined + " " + part
        return combined.str.replace(r"\s+", " ", regex=True).str.strip().values


class FraudKeywordFeatures(BaseEstimator, TransformerMixin):
    """Small numeric feature block for explicit scam signals."""

    def __init__(self):
        self.keywords = [
            "registration fee",
            "fee required",
            "training fee",
            "pay to start",
            "refund later",
            "whatsapp",
            "telegram",
            "google hangouts",
            "hangouts",
            "signal app",
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
            "form filling",
            "bank check",
            "equipment purchase",
            "macbook",
            "wire transfer",
            "cashapp",
            "venmo",
            "zelle",
            "crypto",
            "ethereum",
            "bitcoin",
            "investment",
            "anonymous",
            "no interview",
            "guaranteed job",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = pd.Series(TextCombiner().transform(X)).fillna("").astype(str).str.lower()
        features = []

        for text in texts:
            keyword_count = sum(1 for keyword in self.keywords if keyword in text)
            money_mentions = len(re.findall(r"(₹|\$|rs\.?|rupees?|usd|eth|btc)\s?\d+", text))
            features.append(
                [
                    keyword_count,
                    int("registration fee" in text or "training fee" in text),
                    int("whatsapp" in text or "telegram" in text or "signal app" in text),
                    int("urgent" in text or "hurry" in text or "limited slots" in text),
                    int("no experience" in text),
                    int("work from home" in text),
                    int("high salary" in text or "quick money" in text),
                    int("equipment" in text or "macbook" in text),
                    int("crypto" in text or "bitcoin" in text or "ethereum" in text),
                    money_mentions,
                    int(bool(re.search(r"[\w\.-]+@[\w\.-]+", text))),
                    int(bool(re.search(r"https?://|#url_", text))),
                    np.log1p(len(text)),
                ]
            )

        return np.array(features, dtype=float)


class AdvancedLinguisticFeatures(FraudKeywordFeatures):
    pass


class DeepHeuristicFlags(FraudKeywordFeatures):
    pass
