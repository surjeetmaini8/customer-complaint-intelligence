"""Distribution/aggregation helpers backing dashboard charts."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import Complaint


def _distribution(db: Session, column) -> Dict[str, int]:
    rows = db.query(column, func.count(Complaint.id)).group_by(column).all()
    return {str(k) if k is not None else "Unknown": v for k, v in rows}


def category_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.category)


def severity_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.severity)


def sentiment_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.sentiment)


def emotion_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.emotion)


def source_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.source)


def region_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.region)


def platform_distribution(db: Session) -> Dict[str, int]:
    return _distribution(db, Complaint.platform)


def daily_volume(db: Session) -> List[Dict]:
    rows = db.query(Complaint.created_at).all()
    if not rows:
        return []
    ts = pd.to_datetime([r[0] for r in rows])
    counts = pd.Series(1, index=ts).resample("D").sum()
    return [{"date": str(d.date()), "count": int(c)} for d, c in counts.items()]


def hourly_volume(db: Session) -> List[Dict]:
    rows = db.query(Complaint.created_at).all()
    if not rows:
        return []
    ts = pd.to_datetime([r[0] for r in rows])
    counts = pd.Series(1, index=ts).resample("h").sum()
    return [{"datetime": str(d), "count": int(c)} for d, c in counts.items()]
