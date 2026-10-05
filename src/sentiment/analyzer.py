"""
Sentiment classification: Positive / Neutral / Negative + confidence.

Backend selection (transparent, automatic):
  1. If `transformers` + a pretrained sentiment model are available and
     reachable, use them (best quality).
  2. Otherwise, fall back to a deterministic lexicon/rule-based sentiment
     scorer built into this module. This keeps the platform fully runnable
     offline while still returning calibrated-looking confidence scores.

The model is loaded ONCE per process (module-level singleton) and reused -
see get_sentiment_analyzer().
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

from src.config import settings, get_logger

logger = get_logger(__name__)

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


POSITIVE_WORDS = {
    "great", "good", "excellent", "amazing", "love", "happy", "satisfied", "thank", "thanks",
    "helpful", "resolved", "quick", "smooth", "perfect", "awesome", "pleased", "appreciate",
    "fantastic", "wonderful", "nice", "best", "easy", "fast", "impressed",
}
NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "horrible", "angry", "frustrated", "frustrating", "disappointed",
    "disappointing", "issue", "problem", "failed", "failure", "fraud", "unauthorized", "damaged",
    "broken", "delay", "delayed", "missing", "wrong", "rude", "unhelpful", "never", "not working",
    "crash", "crashes", "error", "poor", "unacceptable", "waste", "scam", "cheated", "annoyed",
    "annoying", "hate", "useless", "slow", "stuck", "pending", "complain", "complaint", "refund",
    "reversed", "locked", "suspicious", "urgent", "asap", "disrespectfully",
}
INTENSIFIERS = {"very", "extremely", "completely", "totally", "absolutely", "really"}
NEGATIONS = {"not", "never", "no", "n't"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _lexicon_sentiment(text: str) -> Dict:
    tokens = _tokenize(text)
    pos_score, neg_score = 0.0, 0.0
    for i, tok in enumerate(tokens):
        weight = 1.0
        window = tokens[max(0, i - 2):i]
        if any(w in INTENSIFIERS for w in window):
            weight = 1.5
        negated = any(w in NEGATIONS for w in window)
        if tok in POSITIVE_WORDS:
            if negated:
                neg_score += weight
            else:
                pos_score += weight
        elif tok in NEGATIVE_WORDS:
            if negated:
                pos_score += weight * 0.5
            else:
                neg_score += weight

    total = pos_score + neg_score
    if total == 0:
        return {"label": "Neutral", "confidence": 0.55, "scores": {"Positive": 0.2, "Neutral": 0.6, "Negative": 0.2}}

    pos_ratio = pos_score / total
    neg_ratio = neg_score / total

    if pos_ratio - neg_ratio > 0.25:
        label = "Positive"
        confidence = min(0.5 + pos_ratio * 0.5, 0.97)
    elif neg_ratio - pos_ratio > 0.15:
        label = "Negative"
        confidence = min(0.5 + neg_ratio * 0.5, 0.97)
    else:
        label = "Neutral"
        confidence = 0.55

    scores = {
        "Positive": round(pos_ratio, 3),
        "Negative": round(neg_ratio, 3),
        "Neutral": round(max(1 - pos_ratio - neg_ratio, 0.0), 3),
    }
    return {"label": label, "confidence": round(confidence, 3), "scores": scores}


class SentimentAnalyzer:
    def __init__(self):
        self.threshold = settings.get("sentiment.confidence_threshold", 0.60)
        self._hf_pipe = None
        self.backend = "lexicon"
        if TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = hf_pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                )
                self.backend = "transformer"
                logger.info("SentimentAnalyzer using transformer backend")
            except Exception as e:
                logger.info("Transformer sentiment model unavailable (%s); using lexicon fallback", e)
                self._hf_pipe = None
        if self._hf_pipe is None:
            logger.info("SentimentAnalyzer using lexicon-based fallback backend")

    def analyze(self, text: str) -> Dict:
        if not text or not text.strip():
            return {"label": "Neutral", "confidence": 0.5, "needs_human_review": True, "backend": self.backend}

        if self._hf_pipe is not None:
            try:
                out = self._hf_pipe(text[:512])[0]
                label_map = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
                label = label_map.get(out["label"], "Neutral")
                confidence = float(out["score"])
                result = {"label": label, "confidence": confidence}
            except Exception as e:
                logger.warning("Transformer sentiment inference failed (%s), using lexicon fallback", e)
                result = _lexicon_sentiment(text)
        else:
            result = _lexicon_sentiment(text)

        result["needs_human_review"] = result["confidence"] < self.threshold
        result["backend"] = self.backend
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(t) for t in texts]


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()
