"""
Lightweight language detection.

Design: English-first, but structured so a proper multilingual detector
(e.g. fasttext / langdetect) can be swapped in later without touching
call sites - just replace `detect_language`.
"""
from __future__ import annotations

import re

_ASCII_LETTERS_RE = re.compile(r"[A-Za-z]")
_COMMON_EN_WORDS = {
    "the", "and", "is", "was", "my", "to", "for", "on", "in", "please",
    "order", "payment", "account", "app", "issue", "not", "with",
}


def detect_language(text: str) -> str:
    """Returns an ISO-639-1-ish code. Currently supports reliable English
    detection; anything that doesn't look like Latin-script text is
    tagged 'unknown' so downstream code can decide how to handle it."""
    if not text or not text.strip():
        return "unknown"

    letters = _ASCII_LETTERS_RE.findall(text)
    ratio = len(letters) / max(len(text), 1)
    if ratio < 0.4:
        return "unknown"

    tokens = set(re.findall(r"[a-zA-Z']+", text.lower()))
    overlap = len(tokens & _COMMON_EN_WORDS)
    if ratio >= 0.6 or overlap >= 1:
        return "en"
    return "unknown"


def is_english(text: str) -> bool:
    return detect_language(text) == "en"
