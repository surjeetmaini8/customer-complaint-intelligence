"""Knowledge-base vector store with FAISS when available and NumPy fallback."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from src.config import get_logger
logger = get_logger(__name__)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

class KnowledgeVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunks: List[Dict] = []
        self._vectors = np.empty((0, dimension), dtype="float32")
        self.index = faiss.IndexFlatIP(dimension) if FAISS_AVAILABLE else None
    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else len(self.chunks)
    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match chunks")
        values = embeddings.astype("float32")
        if self.index is not None:
            self.index.add(values)
        else:
            self._vectors = np.vstack([self._vectors, values])
        self.chunks.extend(chunks)
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict]:
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
        results = []
        for score, idx in pairs:
            idx = int(idx)
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["similarity"] = float(score)
            results.append(chunk)
        return results
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "knowledge_chunks.json", "w") as f:
            json.dump(self.chunks, f)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "knowledge.index"))
            backend = "faiss"
        else:
            np.save(directory / "knowledge_vectors.npy", self._vectors)
            backend = "numpy"
        with open(directory / "knowledge_index_meta.json", "w") as f:
            json.dump({"backend": backend, "dimension": self.dimension}, f)
        logger.info("Saved %s knowledge index (%d chunks)", backend, self.ntotal)
    @classmethod
    def load(cls, directory: Path) -> "KnowledgeVectorStore":
        with open(directory / "knowledge_chunks.json") as f:
            chunks = json.load(f)
        meta_path = directory / "knowledge_index_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            dimension, backend = int(meta["dimension"]), meta["backend"]
        else:
            if not FAISS_AVAILABLE:
                raise RuntimeError("Legacy FAISS index found but faiss-cpu is not installed")
            backend = "faiss"
            dimension = faiss.read_index(str(directory / "knowledge.index")).d
        obj = cls(dimension)
        obj.chunks = chunks
        if backend == "faiss" and FAISS_AVAILABLE and (directory / "knowledge.index").exists():
            obj.index = faiss.read_index(str(directory / "knowledge.index"))
        elif (directory / "knowledge_vectors.npy").exists():
            obj.index = None
            obj._vectors = np.load(directory / "knowledge_vectors.npy")
        else:
            raise RuntimeError("Saved knowledge index cannot be loaded with installed dependencies")
        return obj
    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "knowledge.index").exists() or (directory / "knowledge_vectors.npy").exists()
