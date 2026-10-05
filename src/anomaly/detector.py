"""
Emerging issue detection: finds sudden spikes in complaint volume for a
given (category, subcategory) [optionally platform/region] combination
using a rolling-baseline + z-score approach.

This is what powers the "🚨 OTP Verification Failure, +714%" style alerts.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings, get_logger

logger = get_logger(__name__)


def _bucketize(df: pd.DataFrame, time_bucket: str) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"])
    return ts.dt.floor(time_bucket)


def _severity_for_increase(increase_pct: float, z_score: float) -> str:
    if increase_pct >= 400 or z_score >= 5:
        return "CRITICAL"
    if increase_pct >= 200 or z_score >= 3.5:
        return "HIGH"
    if increase_pct >= 100 or z_score >= 2.5:
        return "MEDIUM"
    return "LOW"


def detect_emerging_issues(df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> List[Dict]:
    """
    df must contain: timestamp, category, subcategory, complaint_id,
    and ideally platform/region.

    For each (category, subcategory) group, builds an hourly time series,
    computes a rolling baseline (mean) and rolling std, and flags the most
    recent bucket if it's a statistically significant spike above baseline
    (z-score) AND exceeds a minimum volume / percent-increase threshold
    (to avoid flagging normal noise on very low-volume issues).
    """
    cfg = settings.get("anomaly_detection", {})
    time_bucket = cfg.get("time_bucket", "H")
    rolling_window = cfg.get("rolling_window", 24)
    z_threshold = cfg.get("z_score_threshold", 2.5)
    min_volume = cfg.get("min_volume_for_alert", 20)
    min_increase_pct = cfg.get("increase_percent_threshold", 100)

    group_cols = group_cols or ["category", "subcategory"]
    df = df.copy()
    df["_bucket"] = _bucketize(df, time_bucket)

    issues = []
    for group_key, group in df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        series = group.groupby("_bucket").size().sort_index()
        if len(series) < 3:
            continue

        # reindex to a continuous hourly range so gaps count as zero volume
        full_index = pd.date_range(series.index.min(), series.index.max(), freq=time_bucket)
        series = series.reindex(full_index, fill_value=0)

        # Baseline = rolling mean/std computed using PRIOR buckets only (shifted),
        # so each point is compared against what came before it - this lets us
        # scan the entire timeline for spikes rather than only the most recent
        # bucket (useful both for live monitoring and for retrospective/demo
        # analysis over a historical dataset).
        rolling_mean = series.shift(1).rolling(window=rolling_window, min_periods=3).mean()
        rolling_std = series.shift(1).rolling(window=rolling_window, min_periods=3).std().fillna(0)
        safe_std = rolling_std.replace(0, 1.0)

        z_scores = (series - rolling_mean) / safe_std
        increase_pcts = np.where(rolling_mean > 0, (series - rolling_mean) / rolling_mean * 100,
                                  np.where(series > 0, 100.0, 0.0))

        candidate_mask = (series >= min_volume) & ((z_scores >= z_threshold) | (increase_pcts >= min_increase_pct))
        candidate_mask = candidate_mask.fillna(False)
        if not candidate_mask.any():
            continue

        # pick the single most anomalous bucket (peak of the spike) for this group
        candidate_z = z_scores.where(candidate_mask, -np.inf)
        peak_idx = candidate_z.idxmax()

        current_volume = series.loc[peak_idx]
        baseline = rolling_mean.loc[peak_idx] if not np.isnan(rolling_mean.loc[peak_idx]) else series.mean()
        z_score = z_scores.loc[peak_idx]
        increase_pct = increase_pcts[series.index.get_loc(peak_idx)]

        window_end = peak_idx
        window_start = window_end - pd.Timedelta(hours=rolling_window)
        recent_complaints = group[group["_bucket"] == window_end]

        record = dict(zip(group_cols, group_key))
        title_parts = [str(v) for v in group_key if v]
        title = " - ".join(title_parts) if title_parts else "Unknown issue"

        platform = recent_complaints["platform"].mode().iloc[0] if "platform" in recent_complaints and len(recent_complaints) else None
        region = recent_complaints["region"].mode().iloc[0] if "region" in recent_complaints and len(recent_complaints) else None

        issues.append({
            "issue_id": f"ISSUE-{uuid.uuid4().hex[:8].upper()}",
            "title": title,
            **record,
            "current_volume": int(current_volume),
            "baseline_volume": float(round(baseline, 1)),
            "increase_percent": float(round(increase_pct, 1)),
            "anomaly_score": float(round(min(z_score / 6.0, 1.0), 3)),
            "severity": _severity_for_increase(increase_pct, z_score),
            "platform": platform,
            "region": region,
            "first_detected_at": window_end.to_pydatetime(),
            "window_start": window_start.to_pydatetime(),
            "window_end": window_end.to_pydatetime(),
            "representative_complaint_ids": recent_complaints["complaint_id"].head(5).tolist(),
        })

    issues.sort(key=lambda x: x["anomaly_score"], reverse=True)
    logger.info("Detected %d emerging issues", len(issues))
    return issues
