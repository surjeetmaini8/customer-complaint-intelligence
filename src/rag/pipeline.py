"""Orchestrates the RAG retrieval step used by root-cause analysis."""
from __future__ import annotations

from typing import Dict, List, Optional

from src.rag.retriever import get_knowledge_retriever


def retrieve_evidence(query: str, category: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]:
    retriever = get_knowledge_retriever()
    if not retriever.is_ready:
        return []
    return retriever.retrieve(query, top_k=top_k, category=category)
