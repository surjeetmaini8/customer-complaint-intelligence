"""
Unified classification inference API used by the rest of the system.

Loads the configured backend (baseline TF-IDF+LogReg by default, or the
optional transformer classifier if configured & available) ONCE and caches
it in memory, so repeated calls do not reload models from disk.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from src.config import settings, get_logger
from src.classification.baseline import BaselineClassifier
from src.classification.taxonomy import subcategories

logger = get_logger(__name__)


class ClassificationService:
    def __init__(self):
        self.model_type = settings.get("classification.model_type", "tfidf_logreg")
        self.confidence_threshold = settings.get("classification.confidence_threshold", 0.70)
        self.top_k = settings.get("classification.top_k_alternatives", 3)
        self._category_model = None
        self._subcategory_models: Dict[str, BaselineClassifier] = {}
        self._load()

    def _load(self):
        cat_dir = settings.path("paths.classification_dir")
        try:
            if self.model_type == "transformer":
                from src.classification.transformer import TransformerClassifier
                self._category_model = TransformerClassifier.load(cat_dir, "category")
                logger.info("Loaded transformer category classifier")
            else:
                self._category_model = BaselineClassifier.load(cat_dir, "category")
                logger.info("Loaded baseline (TF-IDF+LogReg) category classifier")
        except Exception as e:
            logger.warning("No trained category classifier found (%s). "
                            "Run scripts/train_models.py first. Predictions will be unavailable.", e)
            self._category_model = None

        # per-category subcategory classifiers (optional, best-effort)
        for cat in ["Payment", "Delivery", "Refund", "Account", "Product Quality",
                    "Fraud", "Technical Issue", "Subscription", "Customer Service", "Other"]:
            safe_name = f"sub_{cat.replace(' ', '_')}"
            if BaselineClassifier.exists(cat_dir, safe_name):
                try:
                    self._subcategory_models[cat] = BaselineClassifier.load(cat_dir, safe_name)
                except Exception:
                    pass

    @property
    def is_ready(self) -> bool:
        return self._category_model is not None

    def predict_category(self, text: str) -> Dict:
        if not self.is_ready:
            return {"label": "Other", "confidence": 0.0, "alternatives": [],
                    "needs_human_review": True, "error": "model_not_trained"}
        result = self._category_model.predict([text], top_k=self.top_k)[0]
        result["needs_human_review"] = result["confidence"] < self.confidence_threshold
        return result

    def predict_subcategory(self, text: str, category: str) -> Dict:
        model = self._subcategory_models.get(category)
        if model is None:
            subs = subcategories(category)
            fallback = subs[0] if subs else "miscellaneous"
            return {"label": fallback, "confidence": 0.0, "alternatives": [],
                    "needs_human_review": True, "error": "no_subcategory_model"}
        result = model.predict([text], top_k=min(self.top_k, len(model.label_classes_)))[0]
        result["needs_human_review"] = result["confidence"] < self.confidence_threshold
        return result

    def predict(self, text: str) -> Dict:
        category_result = self.predict_category(text)
        subcategory_result = self.predict_subcategory(text, category_result["label"])
        return {
            "category": category_result["label"],
            "category_confidence": category_result["confidence"],
            "category_alternatives": category_result["alternatives"],
            "subcategory": subcategory_result["label"],
            "subcategory_confidence": subcategory_result["confidence"],
            "subcategory_alternatives": subcategory_result["alternatives"],
            "needs_human_review": category_result.get("needs_human_review", False)
            or subcategory_result.get("needs_human_review", False),
        }


@lru_cache(maxsize=1)
def get_classification_service() -> ClassificationService:
    return ClassificationService()
