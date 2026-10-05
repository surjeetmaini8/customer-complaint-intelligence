"""Feature engineering for the severity engine (shared by rule-based and
optional ML-based severity models)."""
from __future__ import annotations

from typing import Dict

from src.config import settings
from src.severity.rules import FRAUD_CATEGORIES, FRAUD_SUBCATEGORIES, HIGH_IMPACT_SUBCATEGORIES


def build_severity_features(
    category: str,
    subcategory: str,
    sentiment: str,
    emotion: str,
    amount: float,
    recent_similar_count: int = 0,
    resolution_status: str = "open",
) -> Dict[str, float]:
    cfg = settings.get("severity", {})
    high_amount = cfg.get("high_amount", 5000)
    critical_amount = cfg.get("critical_amount", 20000)

    financial_impact = min(amount / critical_amount, 1.0) if critical_amount else 0.0

    fraud_risk = 1.0 if (category in FRAUD_CATEGORIES or subcategory in FRAUD_SUBCATEGORIES) else (
        0.4 if subcategory in HIGH_IMPACT_SUBCATEGORIES else 0.0
    )

    sentiment_score = {"Negative": 1.0, "Neutral": 0.4, "Positive": 0.0}.get(sentiment, 0.4)
    emotion_score = {"Angry": 1.0, "Frustrated": 0.8, "Disappointed": 0.6,
                      "Confused": 0.4, "Neutral": 0.2, "Satisfied": 0.0}.get(emotion, 0.4)

    frequency_score = min(recent_similar_count / 20.0, 1.0)
    unresolved_score = 1.0 if resolution_status in ("open", "escalated") else (
        0.5 if resolution_status == "in_progress" else 0.0
    )

    return {
        "sentiment": sentiment_score,
        "emotion": emotion_score,
        "financial_impact": financial_impact,
        "fraud_risk": fraud_risk,
        "frequency": frequency_score,
        "unresolved": unresolved_score,
        "amount_high": 1.0 if amount >= high_amount else 0.0,
        "amount_critical": 1.0 if amount >= critical_amount else 0.0,
    }
