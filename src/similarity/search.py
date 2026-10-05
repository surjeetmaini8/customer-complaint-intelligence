"""Vector similarity search with FAISS when available and a NumPy fallback otherwise."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from src.config import settings, get_logger
logger = get_logger(__name__)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

class ComplaintVectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.id_map: List[str] = []
        self._vectors = np.empty((0, dimension), dtype="float32")
        self.index = faiss.IndexFlatIP(dimension) if FAISS_AVAILABLE else None

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else len(self.id_map)

    def add(self, embeddings: np.ndarray, complaint_ids: List[str]) -> None:
        if embeddings.shape[0] != len(complaint_ids):
            raise ValueError("Number of embeddings must match complaint IDs")
        values = embeddings.astype("float32")
        if self.index is not None:
            self.index.add(values)
        else:
            self._vectors = np.vstack([self._vectors, values])
        self.id_map.extend(complaint_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.ntotal == 0:
            return []
        top_k = min(top_k, self.ntotal)
        query = query_embedding.astype("float32").reshape(-1)
        if self.index is not None:
            scores, indices = self.index.search(query.reshape(1, -1), top_k)
            pairs = zip(scores[0], indices[0])
        else:
            scores = self._vectors @ query
            indices = np.argsort(scores)[::-1][:top_k]
            pairs = ((scores[i], i) for i in indices)
        return [{"complaint_id": self.id_map[int(idx)], "similarity": float(score)} for score, idx in pairs if int(idx) >= 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "complaints_id_map.json", "w") as f:
            json.dump(self.id_map, f)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "complaints.index"))
            backend = "faiss"
        else:
            np.save(directory / "complaints_vectors.npy", self._vectors)
            backend = "numpy"
        with open(directory / "complaints_index_meta.json", "w") as f:
            json.dump({"backend": backend, "dimension": self.dimension}, f)
        logger.info("Saved %s complaint index (%d vectors)", backend, self.ntotal)

    @classmethod
    def load(cls, directory: Path) -> "ComplaintVectorIndex":
        with open(directory / "complaints_id_map.json") as f:
            ids = json.load(f)
        meta_path = directory / "complaints_index_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            dimension, backend = int(meta["dimension"]), meta["backend"]
        else:
            if not FAISS_AVAILABLE:
                raise RuntimeError("Legacy FAISS index found but faiss-cpu is not installed")
            backend = "faiss"
            dimension = faiss.read_index(str(directory / "complaints.index")).d
        obj = cls(dimension)
        obj.id_map = ids
        if backend == "faiss" and FAISS_AVAILABLE and (directory / "complaints.index").exists():
            obj.index = faiss.read_index(str(directory / "complaints.index"))
        elif (directory / "complaints_vectors.npy").exists():
            obj.index = None
            obj._vectors = np.load(directory / "complaints_vectors.npy")
        else:
            raise RuntimeError("Saved complaint index cannot be loaded with installed dependencies")
        return obj

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "complaints.index").exists() or (directory / "complaints_vectors.npy").exists()

def similar_complaints_for_text(text: str, encoder, index: ComplaintVectorIndex, top_k: Optional[int] = None, min_similarity: Optional[float] = None) -> List[Dict]:
    cfg = settings.get("similarity", {})
    top_k = top_k or cfg.get("top_k", 10)
    min_similarity = min_similarity if min_similarity is not None else cfg.get("min_similarity", 0.35)
    embedding = encoder.encode([text])[0]
    results = index.search(embedding, top_k=top_k * 2)
    return [r for r in results if r["similarity"] >= min_similarity][:top_k]
