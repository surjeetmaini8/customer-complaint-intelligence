"""
Retriever: encodes a query and returns top-k relevant knowledge-base
chunks with metadata, using a shared embedding encoder so knowledge and
complaint embeddings live in a comparable space (both fit at index-build
time - see scripts/build_index.py).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from src.config import settings, get_logger
from src.embeddings.encoder import EmbeddingEncoder
from src.rag.vector_store import KnowledgeVectorStore

logger = get_logger(__name__)


class KnowledgeRetriever:
    def __init__(self):
        self.encoder: Optional[EmbeddingEncoder] = None
        self.store: Optional[KnowledgeVectorStore] = None
        self._load()

    def _load(self):
        faiss_dir = settings.path("paths.faiss_dir")
        try:
            self.encoder = EmbeddingEncoder.load(faiss_dir / "knowledge_encoder")
            self.store = KnowledgeVectorStore.load(faiss_dir)
            logger.info("Loaded knowledge retriever (%d chunks)", len(self.store.chunks))
        except Exception as e:
            logger.warning("Knowledge retriever not built yet (%s). Run scripts/build_index.py.", e)
            self.encoder = None
            self.store = None

    @property
    def is_ready(self) -> bool:
        return self.encoder is not None and self.store is not None

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 min_similarity: Optional[float] = None,
                 category: Optional[str] = None) -> List[Dict]:
        if not self.is_ready:
            return []
        cfg = settings.get("rag", {})
        top_k = top_k or cfg.get("top_k", 4)
        min_similarity = min_similarity if min_similarity is not None else cfg.get("min_similarity", 0.15)

        query_emb = self.encoder.encode([query])[0]
        results = self.store.search(query_emb, top_k=top_k * 3)
        if category:
            category_matches = [r for r in results if r.get("category") == category]
            if category_matches:
                results = category_matches
        filtered = [r for r in results if r["similarity"] >= min_similarity][:top_k]
        return filtered


@lru_cache(maxsize=1)
def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()
