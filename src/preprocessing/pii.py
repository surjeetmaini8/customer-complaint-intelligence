"""
PII detection and masking (regex-based - deterministic, no external model
required, runs identically in any environment).

Masks: phone numbers, emails, card-like numbers, order IDs, customer IDs,
and best-effort street addresses. Masked text is what gets sent to the LLM.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b")),
    ("ORDER_ID", re.compile(r"\bORD\d{6,}\b", re.IGNORECASE)),
    ("CUSTOMER_ID", re.compile(r"\bCUST\d{4,}\b", re.IGNORECASE)),
]

_ADDRESS_HINT_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.,'\s]{3,40}\b(street|st\.|road|rd\.|avenue|ave\.|lane|block|sector|colony)\b",
    re.IGNORECASE,
)


def detect_pii(text: str) -> List[Dict]:
    """Returns list of {text, label, start, end} for detected PII spans."""
    matches = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            matches.append({"text": m.group(), "label": label, "start": m.start(), "end": m.end()})
    for m in _ADDRESS_HINT_RE.finditer(text):
        matches.append({"text": m.group(), "label": "ADDRESS", "start": m.start(), "end": m.end()})
    matches.sort(key=lambda x: x["start"])
    return matches


def mask_pii(text: str) -> str:
    """Replaces detected PII spans with [LABEL] placeholders, longest/earliest first
    to avoid overlapping replacement corruption."""
    if not text:
        return text
    matches = detect_pii(text)
    if not matches:
        return text
    # resolve overlaps: keep first (earliest, then longest) non-overlapping spans
    resolved = []
    last_end = -1
    for m in sorted(matches, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if m["start"] >= last_end:
            resolved.append(m)
            last_end = m["end"]

    out = []
    cursor = 0
    for m in resolved:
        out.append(text[cursor:m["start"]])
        out.append(f"[{m['label']}]")
        cursor = m["end"]
    out.append(text[cursor:])
    return "".join(out)
