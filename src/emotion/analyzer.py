"""
Emotion classification into the project taxonomy:
Angry, Frustrated, Confused, Disappointed, Satisfied, Neutral.

Backend selection mirrors sentiment/analyzer.py:
  1. transformer pretrained emotion model, if reachable, with its native
     labels MAPPED into our taxonomy (mapping documented below).
  2. deterministic lexicon fallback otherwise.

LABEL MAPPING (used only when the transformer backend is active - the
common `j-hartmann/emotion-english-distilroberta-base` label set):
    anger    -> Angry
    disgust  -> Frustrated
    fear     -> Confused
    joy      -> Satisfied
    neutral  -> Neutral
    sadness  -> Disappointed
    surprise -> Confused
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

HF_LABEL_MAP = {
    "anger": "Angry",
    "disgust": "Frustrated",
    "fear": "Confused",
    "joy": "Satisfied",
    "neutral": "Neutral",
    "sadness": "Disappointed",
    "surprise": "Confused",
}

EMOTION_LEXICON = {
    "Angry": {"angry", "furious", "outrageous", "unacceptable", "rude", "scam", "fraud",
              "unauthorized", "cheated", "disrespectfully", "hate"},
    "Frustrated": {"frustrated", "frustrating", "annoyed", "annoying", "again", "still",
                   "keep", "keeps", "repeatedly", "multiple", "tired"},
    "Confused": {"confusing", "confused", "unclear", "not sure", "understand", "explain",
                 "why", "how", "what"},
    "Disappointed": {"disappointed", "disappointing", "expected", "hoped", "let down",
                      "sad", "unfortunately"},
    "Satisfied": {"thank", "thanks", "great", "good", "happy", "satisfied", "appreciate",
                  "resolved", "helpful", "pleased"},
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _lexicon_emotion(text: str) -> Dict:
    tokens = set(_tokenize(text))
    scores = {label: 0 for label in EMOTION_LEXICON}
    for label, words in EMOTION_LEXICON.items():
        scores[label] = len(tokens & words)

    total = sum(scores.values())
    if total == 0:
        return {"label": "Neutral", "confidence": 0.5,
                "scores": {k: 0.0 for k in EMOTION_LEXICON}}

    best_label = max(scores, key=scores.get)
    confidence = min(0.5 + (scores[best_label] / total) * 0.45, 0.95)
    norm_scores = {k: round(v / total, 3) for k, v in scores.items()}
    return {"label": best_label, "confidence": round(confidence, 3), "scores": norm_scores}


class EmotionAnalyzer:
    def __init__(self):
        self.threshold = settings.get("emotion.confidence_threshold", 0.55)
        self._hf_pipe = None
        self.backend = "lexicon"
        if TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = hf_pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                )
                self.backend = "transformer"
                logger.info("EmotionAnalyzer using transformer backend")
            except Exception as e:
                logger.info("Transformer emotion model unavailable (%s); using lexicon fallback", e)
                self._hf_pipe = None
        if self._hf_pipe is None:
            logger.info("EmotionAnalyzer using lexicon-based fallback backend")

    def analyze(self, text: str) -> Dict:
        if not text or not text.strip():
            return {"label": "Neutral", "confidence": 0.5, "needs_human_review": True, "backend": self.backend}

        if self._hf_pipe is not None:
            try:
                raw = self._hf_pipe(text[:512])[0]
                best = max(raw, key=lambda x: x["score"])
                mapped = HF_LABEL_MAP.get(best["label"].lower(), "Neutral")
                result = {"label": mapped, "confidence": float(best["score"])}
            except Exception as e:
                logger.warning("Transformer emotion inference failed (%s), using lexicon fallback", e)
                result = _lexicon_emotion(text)
        else:
            result = _lexicon_emotion(text)

        result["needs_human_review"] = result["confidence"] < self.threshold
        result["backend"] = self.backend
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(t) for t in texts]


@lru_cache(maxsize=1)
def get_emotion_analyzer() -> EmotionAnalyzer:
    return EmotionAnalyzer()
