"""
Knowledge-base document ingestion: parses markdown files with YAML
frontmatter metadata, and chunks their content for embedding/retrieval.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml

from src.config import settings, get_logger

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_markdown_document(path: Path) -> Dict:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {
            "document_id": path.stem, "title": path.stem, "doc_type": "unknown",
            "category": None, "tags": None, "created_date": None,
            "content": raw.strip(), "file_path": str(path),
        }
    frontmatter_raw, content = match.groups()
    meta = yaml.safe_load(frontmatter_raw) or {}
    return {
        "document_id": meta.get("document_id", path.stem),
        "title": meta.get("title", path.stem),
        "doc_type": meta.get("doc_type", "unknown"),
        "category": meta.get("category"),
        "tags": meta.get("tags"),
        "created_date": meta.get("created_date"),
        "content": content.strip(),
        "file_path": str(path),
    }


def load_knowledge_base(directory: Path = None) -> List[Dict]:
    directory = directory or settings.path("paths.knowledge_base")
    docs = []
    for path in sorted(directory.glob("*.md")):
        try:
            docs.append(parse_markdown_document(path))
        except Exception as e:
            logger.warning("Failed to parse knowledge document %s: %s", path, e)
    logger.info("Loaded %d knowledge base documents from %s", len(docs), directory)
    return docs


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Simple word-count based chunking with overlap - good enough for the
    short SOP/FAQ/incident documents in this demo knowledge base."""
    cfg = settings.get("rag", {})
    chunk_size = chunk_size or cfg.get("chunk_size", 400)
    overlap = overlap or cfg.get("chunk_overlap", 60)

    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """Returns a flat list of chunk records: {document_id, title, doc_type,
    category, chunk_index, chunk_text, ...metadata}."""
    chunks = []
    for doc in docs:
        text_chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "document_id": doc["document_id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "category": doc.get("category"),
                "tags": doc.get("tags"),
                "chunk_index": i,
                "chunk_text": chunk,
            })
    logger.info("Chunked %d documents into %d chunks", len(docs), len(chunks))
    return chunks
