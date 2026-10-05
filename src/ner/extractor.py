"""
Named entity extraction for complaint text.

Extracts: ORDER_ID, PRODUCT, LOCATION, DATE, COMPANY, AMOUNT,
PAYMENT_METHOD, PHONE, EMAIL.

Backend selection:
  - If spaCy + a pretrained pipeline are available, use it for generic
    entities (PERSON/ORG/GPE/DATE) and merge with regex rules for the
    domain-specific entities pretrained NER cannot reliably catch
    (order IDs, payment methods, amounts).
  - Otherwise, use regex/rule-based extraction only (fully offline,
    zero dependency risk) - this is the default in this environment.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

from src.config import get_logger

logger = get_logger(__name__)

try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        _NLP = None
except Exception:
    SPACY_AVAILABLE = False
    _NLP = None

PRODUCTS = ["Prime Wallet", "QuickPay", "ShopEase App", "FastCart", "PayNow Card",
            "CloudStore Subscription", "InstaDeliver", "SecureBank App", "MegaMart Online", "TravelGo"]
PAYMENT_METHODS = ["UPI", "credit card", "debit card", "net banking", "wallet balance", "COD",
                    "wallet", "netbanking"]
REGIONS = ["North India", "South India", "East India", "West India", "Central India"]

_RULES = [
    ("ORDER_ID", re.compile(r"\bORD\d{6,}\b", re.IGNORECASE)),
    ("CUSTOMER_ID", re.compile(r"\bCUST\d{4,}\b", re.IGNORECASE)),
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10}\b")),
    ("AMOUNT", re.compile(r"(?:Rs\.?|INR|₹)\s?[\d,]+(?:\.\d+)?", re.IGNORECASE)),
    ("DATE", re.compile(
        r"\b\d{1,2}\s+(days|hours|weeks|months)\b|"
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", re.IGNORECASE)),
]


def _regex_entities(text: str) -> List[Dict]:
    ents = []
    for label, pattern in _RULES:
        for m in pattern.finditer(text):
            ents.append({"text": m.group(), "label": label, "start": m.start(), "end": m.end()})

    for product in PRODUCTS:
        for m in re.finditer(re.escape(product), text, re.IGNORECASE):
            ents.append({"text": m.group(), "label": "PRODUCT", "start": m.start(), "end": m.end()})

    for method in PAYMENT_METHODS:
        for m in re.finditer(r"\b" + re.escape(method) + r"\b", text, re.IGNORECASE):
            ents.append({"text": m.group(), "label": "PAYMENT_METHOD", "start": m.start(), "end": m.end()})

    for region in REGIONS:
        for m in re.finditer(re.escape(region), text, re.IGNORECASE):
            ents.append({"text": m.group(), "label": "LOCATION", "start": m.start(), "end": m.end()})

    return ents


def _spacy_entities(text: str) -> List[Dict]:
    label_map = {"ORG": "COMPANY", "GPE": "LOCATION", "LOC": "LOCATION", "DATE": "DATE", "MONEY": "AMOUNT"}
    doc = _NLP(text)
    ents = []
    for ent in doc.ents:
        mapped = label_map.get(ent.label_)
        if mapped:
            ents.append({"text": ent.text, "label": mapped, "start": ent.start_char, "end": ent.end_char})
    return ents


def _dedupe_overlaps(entities: List[Dict]) -> List[Dict]:
    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    result = []
    last_end = -1
    for e in entities:
        if e["start"] >= last_end:
            result.append(e)
            last_end = e["end"]
    return sorted(result, key=lambda e: e["start"])


class NERExtractor:
    def __init__(self):
        self.backend = "spacy+rules" if SPACY_AVAILABLE else "rules_only"
        logger.info("NERExtractor backend: %s", self.backend)

    def extract(self, text: str) -> List[Dict]:
        if not text:
            return []
        entities = _regex_entities(text)
        if SPACY_AVAILABLE:
            try:
                entities += _spacy_entities(text)
            except Exception as e:
                logger.warning("spaCy NER failed (%s), continuing with regex-only entities", e)
        return _dedupe_overlaps(entities)

    def extract_batch(self, texts: List[str]) -> List[List[Dict]]:
        return [self.extract(t) for t in texts]


@lru_cache(maxsize=1)
def get_ner_extractor() -> NERExtractor:
    return NERExtractor()
