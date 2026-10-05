from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from src.config import settings, get_logger
from src.severity.features import build_severity_features

logger = get_logger(__name__)

FEATURE_ORDER = ["sentiment", "emotion", "financial_impact", "fraud_risk", "frequency", "unresolved"]


def _level_from_score(score: float) -> str:
    cfg = settings.get("severity.thresholds", {})
    if score >= cfg.get("high", 0.78):
        return "CRITICAL"
    if score >= cfg.get("medium", 0.55):
        return "HIGH"
    if score >= cfg.get("low", 0.30):
        return "MEDIUM"
    return "LOW"


def _build_reasons(features: Dict[str, float], category: str, subcategory: str, amount: float) -> List[str]:
    reasons = []
    if features["fraud_risk"] >= 0.8:
        reasons.append("Potential unauthorized transaction or fraud/security risk")
    elif features["fraud_risk"] > 0:
        reasons.append(f"High-impact issue type ({subcategory})")
    if features["amount_critical"]:
        reasons.append(f"Very high financial impact (amount ≈ {amount:,.0f})")
    elif features["amount_high"]:
        reasons.append(f"Significant financial impact (amount ≈ {amount:,.0f})")
    if features["sentiment"] >= 0.9:
        reasons.append("Strong negative sentiment expressed")
    if features["emotion"] >= 0.8:
        reasons.append("Customer emotion indicates anger/strong frustration")
    if features["frequency"] >= 0.5:
        reasons.append("Part of a high-frequency cluster of similar recent complaints")
    if features["unresolved"] >= 0.9:
        reasons.append("Complaint remains unresolved/escalated")
    if not reasons:
        reasons.append("No strong severity signals detected; routine complaint")
    return reasons


class RuleBasedSeverityEngine:
    def __init__(self):
        self.weights = settings.get("severity.weights", {})

    def score(self, features: Dict[str, float]) -> float:
        total_weight = sum(self.weights.values()) or 1.0
        score = sum(features.get(k, 0.0) * w for k, w in self.weights.items()) / total_weight
        return float(min(max(score, 0.0), 1.0))


class MLSeverityEngine:
    """Optional XGBoost severity classifier trained on rule-derived features + labels."""

    def __init__(self):
        self.model = None
        self.label_classes_: List[str] = []

    def fit(self, X: np.ndarray, y: List[str]):
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.label_classes_ = list(self.label_encoder.classes_)
        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
        )
        self.model.fit(X, y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "severity_xgb.joblib")
        joblib.dump(self.label_encoder, directory / "severity_label_encoder.joblib")

    @classmethod
    def load(cls, directory: Path) -> "MLSeverityEngine":
        obj = cls()
        obj.model = joblib.load(directory / "severity_xgb.joblib")
        obj.label_encoder = joblib.load(directory / "severity_label_encoder.joblib")
        obj.label_classes_ = list(obj.label_encoder.classes_)
        return obj

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "severity_xgb.joblib").exists()


class SeverityService:
    def __init__(self):
        self.model_type = settings.get("severity.model_type", "rule_based")
        self.confidence_threshold = settings.get("severity.confidence_threshold", 0.70)
        self.rule_engine = RuleBasedSeverityEngine()
        self.ml_engine = None
        if self.model_type == "xgboost":
            severity_dir = settings.path("paths.severity_dir")
            if MLSeverityEngine.exists(severity_dir):
                self.ml_engine = MLSeverityEngine.load(severity_dir)
                logger.info("Loaded XGBoost severity model")
            else:
                logger.warning("severity.model_type=xgboost but no trained model found; using rule-based engine")

    def assess(self, category: str, subcategory: str, sentiment: str, emotion: str,
               amount: float, recent_similar_count: int = 0, resolution_status: str = "open") -> Dict:
        features = build_severity_features(category, subcategory, sentiment, emotion,
                                            amount, recent_similar_count, resolution_status)
        rule_score = self.rule_engine.score(features)
        rule_level = _level_from_score(rule_score)
        reasons = _build_reasons(features, category, subcategory, amount)

        if self.ml_engine is not None:
            X = np.array([[features[k] for k in FEATURE_ORDER]])
            probs = self.ml_engine.predict_proba(X)[0]
            best_idx = int(np.argmax(probs))
            level = self.ml_engine.label_classes_[best_idx]
            confidence = float(probs[best_idx])
            score = rule_score  # keep the interpretable rule score alongside ML label
        else:
            level = rule_level
            # confidence derived from distance to nearest threshold boundary
            confidence = min(0.6 + abs(rule_score - 0.5), 0.97)
            score = rule_score

        return {
            "severity": level,
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "features": features,
            "needs_human_review": confidence < self.confidence_threshold,
            "engine": "xgboost" if self.ml_engine is not None else "rule_based",
        }


@lru_cache(maxsize=1)
def get_severity_service() -> SeverityService:
    return SeverityService()
