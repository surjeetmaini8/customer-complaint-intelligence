"""Aggregate metrics computed from the database, used by the dashboard's
Executive Overview page and the /stats API endpoint."""
from __future__ import annotations

from typing import Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import Complaint, EmergingIssue

SEVERITY_SCORE_MAP = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}


def get_summary_stats(db: Session) -> Dict:
    total = db.query(func.count(Complaint.id)).scalar() or 0
    critical = db.query(func.count(Complaint.id)).filter(Complaint.severity == "CRITICAL").scalar() or 0
    high = db.query(func.count(Complaint.id)).filter(Complaint.severity == "HIGH").scalar() or 0
    negative = db.query(func.count(Complaint.id)).filter(Complaint.sentiment == "Negative").scalar() or 0
    active_issues = db.query(func.count(EmergingIssue.id)).filter(EmergingIssue.status == "active").scalar() or 0

    severities = db.query(Complaint.severity).all()
    scores = [SEVERITY_SCORE_MAP.get(s[0], 0.0) for s in severities if s[0]]
    avg_severity = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_complaints": total,
        "critical_complaints": critical,
        "high_severity_complaints": high,
        "negative_sentiment_complaints": negative,
        "active_emerging_issues": active_issues,
        "average_severity_score": round(avg_severity, 3),
    }
