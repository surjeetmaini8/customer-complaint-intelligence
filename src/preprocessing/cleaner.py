"""Text cleaning utilities. Deliberately conservative: transformer/NLP models
benefit from natural language, so we normalize noise without stripping
stopwords or over-aggressively altering meaning."""
from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")   # 4+ repeats of same char -> collapse
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_PUNCT_RE = re.compile(r"([!?.]){3,}")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def normalize_urls(text: str) -> str:
    return _URL_RE.sub("[URL]", text)


def collapse_repeated_chars(text: str) -> str:
    return _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * 3, text)


def collapse_repeated_punctuation(text: str) -> str:
    return _MULTI_PUNCT_RE.sub(lambda m: m.group(1) * 2, text)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline used before any NLP model sees the text."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = normalize_urls(text)
    text = collapse_repeated_chars(text)
    text = collapse_repeated_punctuation(text)
    text = normalize_whitespace(text)
    return text
