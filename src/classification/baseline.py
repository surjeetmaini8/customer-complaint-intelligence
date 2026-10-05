"""
Baseline complaint classifier: TF-IDF -> Logistic Regression.

This is the PRIMARY classifier used by the running pipeline (fast, fully
local, trains in seconds on the synthetic dataset, and gives calibrated
probabilities for confidence scoring). See transformer.py for the optional
heavier alternative and why it is not the default in this environment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

from src.config import settings, get_logger

logger = get_logger(__name__)


class BaselineClassifier:
    """One instance handles a single target column (e.g. 'category' or 'subcategory')."""

    def __init__(self, target_name: str = "category"):
        self.target_name = target_name
        cfg = settings.get("classification", {})
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("tfidf_max_features", 20000),
            ngram_range=tuple(cfg.get("tfidf_ngram_range", [1, 2])),
            min_df=2,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.label_classes_: List[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------ training
    def fit(self, texts: List[str], labels: List[str]) -> Dict:
        cfg = settings.get("classification", {})
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=cfg.get("test_size", 0.15),
            random_state=settings.get("data_generation.random_seed", 42),
            stratify=labels,
        )
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        self.model.fit(X_train, y_train)
        self.label_classes_ = list(self.model.classes_)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred)
        logger.info("[%s] baseline trained. acc=%.3f macro_f1=%.3f", self.target_name,
                    metrics["accuracy"], metrics["macro_f1"])
        return metrics

    def _compute_metrics(self, y_true, y_pred) -> Dict:
        acc = accuracy_score(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=self.label_classes_).tolist()
        return {
            "accuracy": acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "macro_f1": f1_macro,
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "weighted_f1": f1_w,
            "classification_report": report,
            "confusion_matrix": cm,
            "labels": self.label_classes_,
        }

    # ------------------------------------------------------------ inference
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict(self, texts: List[str], top_k: int = 3) -> List[Dict]:
        probs = self.predict_proba(texts)
        results = []
        for row in probs:
            order = np.argsort(row)[::-1]
            top = order[:top_k]
            best_idx = top[0]
            alternatives = [
                {"label": self.label_classes_[i], "confidence": float(row[i])}
                for i in top[1:]
            ]
            results.append({
                "label": self.label_classes_[best_idx],
                "confidence": float(row[best_idx]),
                "alternatives": alternatives,
            })
        return results

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / f"{self.target_name}_vectorizer.joblib")
        joblib.dump(self.model, directory / f"{self.target_name}_model.joblib")
        with open(directory / f"{self.target_name}_labels.json", "w") as f:
            json.dump(self.label_classes_, f)
        logger.info("Saved baseline classifier (%s) to %s", self.target_name, directory)

    @classmethod
    def load(cls, directory: Path, target_name: str = "category") -> "BaselineClassifier":
        obj = cls(target_name=target_name)
        obj.vectorizer = joblib.load(directory / f"{target_name}_vectorizer.joblib")
        obj.model = joblib.load(directory / f"{target_name}_model.joblib")
        with open(directory / f"{target_name}_labels.json") as f:
            obj.label_classes_ = json.load(f)
        obj.is_fitted = True
        return obj

    @staticmethod
    def exists(directory: Path, target_name: str = "category") -> bool:
        return (directory / f"{target_name}_model.joblib").exists()
