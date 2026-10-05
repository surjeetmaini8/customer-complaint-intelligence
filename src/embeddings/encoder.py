"""
Text embedding layer.

Backend selection:
  1. sentence-transformers (config: embeddings.model_name), if the package
     and model weights are reachable - produces high quality semantic
     embeddings.
  2. Fallback: TF-IDF + TruncatedSVD (LSA), fit locally on the complaint
     corpus. This requires no external downloads at all and still gives a
     genuinely useful semantic-ish embedding space for similarity search,
     clustering, and anomaly detection - the architecture (FAISS index,
     cosine similarity, clustering) is identical either way.

Whichever backend is used, the encoder is persisted so encoding is
consistent between index-build time and query time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np

from src.config import settings, get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False


class EmbeddingEncoder:
    def __init__(self):
        self.model_name = settings.get("embeddings.model_name")
        self.dimension = settings.get("embeddings.dimension", 200)
        self.svd_components = settings.get("embeddings.svd_components", 200)
        self.backend = None
        self._st_model = None
        self._vectorizer = None
        self._svd = None

    # ------------------------------------------------------------ fit (fallback path)
    def fit_fallback(self, texts: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self._vectorizer.fit_transform(texts)
        n_components = min(self.svd_components, X.shape[1] - 1, X.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(n_components, 2), random_state=42)
        self._svd.fit(X)
        self.dimension = self._svd.n_components
        self.backend = "tfidf_svd"
        logger.info("Fitted TF-IDF+SVD embedding fallback (dim=%d) on %d documents",
                    self.dimension, len(texts))

    def _try_load_sentence_transformer(self) -> bool:
        if not ST_AVAILABLE:
            return False
        try:
            self._st_model = SentenceTransformer(self.model_name)
            self.backend = "sentence_transformer"
            self.dimension = self._st_model.get_sentence_embedding_dimension()
            logger.info("Loaded sentence-transformers model '%s' (dim=%d)", self.model_name, self.dimension)
            return True
        except Exception as e:
            logger.info("sentence-transformers model unavailable (%s); using TF-IDF+SVD fallback", e)
            return False

    def fit(self, texts: List[str]) -> None:
        if self._try_load_sentence_transformer():
            return
        self.fit_fallback(texts)

    # ------------------------------------------------------------ encode
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sentence_transformer":
            return np.asarray(self._st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        if self.backend == "tfidf_svd":
            X = self._vectorizer.transform(texts)
            emb = self._svd.transform(X)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return emb / norms
        raise RuntimeError("EmbeddingEncoder is not fitted/loaded. Call fit() or load() first.")

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"backend": self.backend, "model_name": self.model_name, "dimension": self.dimension}
        with open(directory / "encoder_meta.json", "w") as f:
            json.dump(meta, f)
        if self.backend == "tfidf_svd":
            joblib.dump(self._vectorizer, directory / "tfidf_vectorizer.joblib")
            joblib.dump(self._svd, directory / "svd.joblib")

    @classmethod
    def load(cls, directory: Path) -> "EmbeddingEncoder":
        obj = cls()
        with open(directory / "encoder_meta.json") as f:
            meta = json.load(f)
        obj.backend = meta["backend"]
        obj.model_name = meta["model_name"]
        obj.dimension = meta["dimension"]
        if obj.backend == "tfidf_svd":
            obj._vectorizer = joblib.load(directory / "tfidf_vectorizer.joblib")
            obj._svd = joblib.load(directory / "svd.joblib")
        elif obj.backend == "sentence_transformer":
            if not obj._try_load_sentence_transformer():
                raise RuntimeError("Saved encoder used sentence-transformers but it is unavailable now.")
        return obj

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "encoder_meta.json").exists()
