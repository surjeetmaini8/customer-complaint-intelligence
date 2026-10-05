"""
Builds:
  1. The complaint FAISS similarity index (+ embedding encoder) from the
     generated complaints dataset.
  2. The knowledge-base vector index (+ chunking) from data/knowledge_base/.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import settings, get_logger
from src.embeddings.encoder import EmbeddingEncoder
from src.similarity.search import ComplaintVectorIndex
from src.rag.ingestion import load_knowledge_base, chunk_documents
from src.rag.vector_store import KnowledgeVectorStore
from src.database.database import init_db, get_session
from src.database import repository as repo

logger = get_logger(__name__)


def build_complaint_index():
    csv_path = settings.path("paths.complaints_csv")
    if not csv_path.exists():
        logger.error("No dataset found at %s. Run scripts/generate_data.py first.", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["complaint_text"] = df["complaint_text"].fillna("")

    logger.info("Fitting embedding encoder on %d complaints...", len(df))
    encoder = EmbeddingEncoder()
    encoder.fit(df["complaint_text"].tolist())

    embeddings = encoder.encode(df["complaint_text"].tolist())
    index = ComplaintVectorIndex(dimension=embeddings.shape[1])
    index.add(embeddings, df["complaint_id"].tolist())

    faiss_dir = settings.path("paths.faiss_dir")
    index.save(faiss_dir)
    encoder.save(faiss_dir / "complaint_encoder")
    logger.info("Complaint vector index built: %d vectors, backend=%s", index.ntotal, encoder.backend)
    return encoder, embeddings, df


def build_knowledge_index():
    docs = load_knowledge_base()
    if not docs:
        logger.warning("No knowledge base documents found; skipping knowledge index build.")
        return

    chunks = chunk_documents(docs)
    texts = [c["chunk_text"] for c in chunks]

    logger.info("Fitting embedding encoder on %d knowledge chunks...", len(chunks))
    encoder = EmbeddingEncoder()
    encoder.fit(texts)
    embeddings = encoder.encode(texts)

    store = KnowledgeVectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)

    faiss_dir = settings.path("paths.faiss_dir")
    store.save(faiss_dir)
    encoder.save(faiss_dir / "knowledge_encoder")
    logger.info("Knowledge vector index built: %d chunks, backend=%s", store.ntotal, encoder.backend)

    # also persist document metadata to the DB for the dashboard's Knowledge page
    init_db()
    with get_session() as db:
        for doc in docs:
            repo.upsert_knowledge_document(db, {
                "document_id": doc["document_id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "category": doc.get("category"),
                "tags": doc.get("tags"),
                "content": doc["content"],
                "file_path": doc["file_path"],
                "indexed": True,
            })
    logger.info("Persisted %d knowledge documents to the database", len(docs))


def main():
    logger.info("=== Building complaint similarity index ===")
    build_complaint_index()
    logger.info("=== Building knowledge base RAG index ===")
    build_knowledge_index()
    print("Index build complete.")


if __name__ == "__main__":
    main()
