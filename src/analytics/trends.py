"""Trend/time-series helpers, mainly wrapping the anomaly detector for
dashboard consumption."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.anomaly.detector import detect_emerging_issues


def compute_emerging_issues_from_df(df: pd.DataFrame) -> List[Dict]:
    return detect_emerging_issues(df)


def volume_trend_for_subset(df: pd.DataFrame, freq: str = "D") -> List[Dict]:
    if df.empty:
        return []
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df["created_at"])
    counts = pd.Series(1, index=ts).resample(freq).sum()
    return [{"period": str(idx), "count": int(v)} for idx, v in counts.items()]
