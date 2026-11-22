"""
TechnoHome Customer Support Ticket Router — TF-IDF + Logistic Regression Baseline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from models.model_utils import Prediction


class TFIDFBaseline:
    """TF-IDF + Logistic Regression baseline classifier.

    Uses an sklearn Pipeline wrapping:
      • TfidfVectorizer  (max_features=10000, ngram_range=(1,2), sublinear_tf)
      • LogisticRegression (multinomial, C=1.0, max_iter=1000)
    """

    def __init__(self, max_features=None, ngram_range=None, C=None,
                 max_iter=None, random_state=None):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features or config.TFIDF_MAX_FEATURES,
                ngram_range=ngram_range or config.TFIDF_NGRAM_RANGE,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=max_iter or config.LOGREG_MAX_ITER,
                C=C or config.LOGREG_C,
                solver="lbfgs",
                random_state=random_state or config.RANDOM_SEED,
            )),
        ])

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, texts, labels):
        """Fit the pipeline on *texts* (list[str]) and *labels* (list[str]).

        Labels are expected to be department name strings (e.g. ``'billing_payments'``).
        """
        self.pipeline.fit(texts, labels)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, text):
        """Predict a single text.

        Returns
        -------
        Prediction(label, confidence)
        """
        probs = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        label = self.pipeline.classes_[idx]
        confidence = float(probs[idx])
        return Prediction(label, confidence)

    def predict_raw(self, text):
        """Predict returning (label, confidence) tuple."""
        probs = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        label = self.pipeline.classes_[idx]
        confidence = float(probs[idx])
        return label, confidence

    def predict_batch(self, texts):
        """Predict a list of texts.

        Returns
        -------
        list[tuple[str, float]]
            A list of (label, confidence) pairs.
        """
        probs = self.pipeline.predict_proba(texts)
        indices = np.argmax(probs, axis=1)
        results = []
        for i, idx in enumerate(indices):
            label = self.pipeline.classes_[idx]
            confidence = float(probs[i][idx])
            results.append((label, confidence))
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path=None):
        """Serialize the sklearn pipeline to *path* using joblib."""
        if path is None:
            path = str(config.MODEL_DIR / "tfidf_logreg.joblib")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path=None):
        """Load a previously saved pipeline from *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "tfidf_logreg.joblib")
        self.pipeline = joblib.load(path)


# Alias for app.py compatibility
TFIDFClassifier = TFIDFBaseline
