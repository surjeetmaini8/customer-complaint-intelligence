"""
Complaint clustering using HDBSCAN (density-based, finds natural cluster
count and marks outliers as noise -1), with a KMeans fallback if HDBSCAN
is unavailable or produces degenerate results.

Each cluster is summarized with: dominant category/subcategory/sentiment,
average severity, top TF-IDF keywords, representative complaints, time
range, and an auto-generated human-readable title.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import settings, get_logger

logger = get_logger(__name__)

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

SEVERITY_SCORE_MAP = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}


def run_clustering(embeddings: np.ndarray) -> np.ndarray:
    cfg = settings.get("clustering", {})
    method = cfg.get("method", "hdbscan")

    if method == "hdbscan" and HDBSCAN_AVAILABLE:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=cfg.get("min_cluster_size", 15),
            min_samples=cfg.get("min_samples", 5),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            logger.info("HDBSCAN produced %d clusters (%d noise points)", n_clusters, int((labels == -1).sum()))
            return labels
        logger.info("HDBSCAN produced too few clusters (%d); falling back to KMeans", n_clusters)

    from sklearn.cluster import KMeans
    k = cfg.get("kmeans_fallback_k", 25)
    k = min(k, max(len(embeddings) // 10, 2))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    logger.info("KMeans produced %d clusters", k)
    return labels


def _top_keywords(texts: List[str], top_n: int = 8) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(texts) < 2:
        return []
    try:
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english", min_df=1)
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        order = scores.argsort()[::-1][:top_n]
        return [terms[i] for i in order]
    except ValueError:
        return []


def _auto_title(category: str, subcategory: str, keywords: List[str], platform: str = None) -> str:
    base = subcategory.capitalize() if subcategory else category
    if platform:
        return f"{platform} {base}"
    return base


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray, text_col: str = "complaint_text") -> List[Dict]:
    df = df.copy()
    df["_cluster_label"] = labels
    summaries = []
    for cluster_label, group in df.groupby("_cluster_label"):
        if cluster_label == -1:
            continue   # noise points are not a "cluster"
        dominant_category = group["category"].mode().iloc[0] if "category" in group else None
        dominant_subcategory = group["subcategory"].mode().iloc[0] if "subcategory" in group else None
        dominant_sentiment = group["sentiment"].mode().iloc[0] if "sentiment" in group else None
        avg_severity = group["severity"].map(SEVERITY_SCORE_MAP).mean() if "severity" in group else None
        keywords = _top_keywords(group[text_col].tolist())
        representative_ids = group["complaint_id"].head(5).tolist() if "complaint_id" in group else []
        dominant_platform = group["platform"].mode().iloc[0] if "platform" in group else None

        time_start = pd.to_datetime(group["timestamp"]).min() if "timestamp" in group else None
        time_end = pd.to_datetime(group["timestamp"]).max() if "timestamp" in group else None

        title = _auto_title(dominant_category, dominant_subcategory, keywords, dominant_platform)

        summaries.append({
            "cluster_label": int(cluster_label),
            "title": title,
            "size": int(len(group)),
            "dominant_category": dominant_category,
            "dominant_subcategory": dominant_subcategory,
            "dominant_sentiment": dominant_sentiment,
            "avg_severity_score": float(avg_severity) if avg_severity is not None else None,
            "keywords": keywords,
            "representative_ids": representative_ids,
            "time_start": time_start,
            "time_end": time_end,
        })
    summaries.sort(key=lambda x: x["size"], reverse=True)
    return summaries
