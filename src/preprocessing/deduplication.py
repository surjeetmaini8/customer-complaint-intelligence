"""
Deduplication: exact, normalized-text, and optional semantic (embedding-based)
duplicate detection.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_NORM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_for_dedup(text: str) -> str:
    text = text.lower()
    text = _NORM_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_exact_duplicates(df: pd.DataFrame, text_col: str = "complaint_text") -> pd.Series:
    """Returns a boolean Series marking rows that are exact-text duplicates
    of an earlier row (first occurrence kept as non-duplicate)."""
    hashes = df[text_col].fillna("").apply(_hash_text)
    return hashes.duplicated(keep="first")


def find_normalized_duplicates(df: pd.DataFrame, text_col: str = "complaint_text") -> pd.Series:
    normalized = df[text_col].fillna("").apply(_normalize_for_dedup)
    return normalized.duplicated(keep="first")


def find_semantic_duplicates(embeddings: np.ndarray, threshold: float = 0.97) -> np.ndarray:
    """Optional embedding-based near-duplicate detection using cosine similarity
    against previously-seen rows (O(n^2) - fine for moderate batch sizes;
    for large-scale use, this should run per-cluster instead of globally)."""
    n = embeddings.shape[0]
    is_dup = np.zeros(n, dtype=bool)
    if n == 0:
        return is_dup
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    seen = []
    for i in range(n):
        if seen:
            sims = norm[i] @ np.array(seen).T
            if sims.max() >= threshold:
                is_dup[i] = True
                continue
        seen.append(norm[i])
    return is_dup


def deduplicate_dataframe(df: pd.DataFrame, text_col: str = "complaint_text",
                           embeddings: Optional[np.ndarray] = None,
                           semantic_threshold: float = 0.97) -> pd.DataFrame:
    """Adds `is_duplicate` and `duplicate_type` columns. Does not drop rows -
    the caller decides whether to filter, since duplicates can still carry
    useful frequency signal for severity/emerging-issue detection."""
    df = df.copy()
    exact = find_exact_duplicates(df, text_col)
    normalized = find_normalized_duplicates(df, text_col)
    df["is_duplicate"] = exact | normalized
    df["duplicate_type"] = np.where(exact, "exact", np.where(normalized, "normalized", "none"))

    if embeddings is not None:
        semantic = find_semantic_duplicates(embeddings, semantic_threshold)
        newly = semantic & ~df["is_duplicate"].values
        df.loc[newly, "duplicate_type"] = "semantic"
        df["is_duplicate"] = df["is_duplicate"].values | semantic
    return df
