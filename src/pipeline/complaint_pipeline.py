"""
Central end-to-end complaint processing pipeline.

process_complaint(text, ...) runs: preprocessing -> PII masking ->
classification -> sentiment -> emotion -> NER -> severity -> embedding ->
similar complaints -> cluster lookup -> emerging issue check -> RAG
retrieval -> LLM root-cause analysis -> one structured JSON-compatible
result.

This is the ONE place that wires every component together - the API and
scripts/process_complaints.py both call this module rather than
re-implementing orchestration logic.
"""
from __future__ import annotations

import datetime as dt
import uuid
from functools import lru_cache
from typing import Dict, List, Optional

from src.config import settings, get_logger
from src.preprocessing.cleaner import clean_text
from src.preprocessing.pii import mask_pii
from src.preprocessing.language import detect_language
from src.classification.inference import get_classification_service
from src.sentiment.analyzer import get_sentiment_analyzer
from src.emotion.analyzer import get_emotion_analyzer
from src.ner.extractor import get_ner_extractor
from src.severity.predictor import get_severity_service
from src.embeddings.encoder import EmbeddingEncoder
from src.similarity.search import ComplaintVectorIndex, similar_complaints_for_text
from src.rag.pipeline import retrieve_evidence
from src.llm.reasoning import analyze_root_cause
from src.database.database import get_session
from src.database import repository as repo

logger = get_logger(__name__)


class ComplaintPipelineResources:
    """Loads all heavy resources (models/indexes) ONCE and shares them
    across pipeline invocations."""

    def __init__(self):
        self.classification = get_classification_service()
        self.sentiment = get_sentiment_analyzer()
        self.emotion = get_emotion_analyzer()
        self.ner = get_ner_extractor()
        self.severity = get_severity_service()

        faiss_dir = settings.path("paths.faiss_dir")
        self.encoder: Optional[EmbeddingEncoder] = None
        self.vector_index: Optional[ComplaintVectorIndex] = None
        try:
            self.encoder = EmbeddingEncoder.load(faiss_dir / "complaint_encoder")
            self.vector_index = ComplaintVectorIndex.load(faiss_dir)
            logger.info("Loaded complaint embedding encoder + vector index (%d vectors)",
                        self.vector_index.ntotal)
        except Exception as e:
            logger.warning("Complaint similarity index not available (%s). "
                            "Run scripts/build_index.py first.", e)

        self.clusters_by_id: Dict[str, int] = {}   # populated by scripts/process_complaints.py

    @property
    def similarity_ready(self) -> bool:
        return self.encoder is not None and self.vector_index is not None


@lru_cache(maxsize=1)
def get_pipeline_resources() -> ComplaintPipelineResources:
    return ComplaintPipelineResources()


def _find_emerging_issue_for(db, category: str, subcategory: str):
    issues = repo.list_emerging_issues(db, status="active")
    for issue in issues:
        if issue.category == category and issue.subcategory == subcategory:
            return {
                "issue_id": issue.issue_id, "title": issue.title,
                "current_volume": issue.current_volume, "baseline_volume": issue.baseline_volume,
                "increase_percent": issue.increase_percent, "platform": issue.platform,
                "region": issue.region, "severity": issue.severity,
            }
    return None


def process_complaint(
    text: str,
    complaint_id: Optional[str] = None,
    source: str = "api",
    metadata: Optional[Dict] = None,
    persist: bool = True,
    run_llm: bool = True,
) -> Dict:
    """Runs the full pipeline on a single complaint and returns one
    structured, JSON-serializable result dict. If `persist=True`, writes
    the complaint + predictions + entities + analysis to the database."""
    metadata = metadata or {}
    resources = get_pipeline_resources()
    complaint_id = complaint_id or f"CMP-{uuid.uuid4().hex[:10].upper()}"

    # 1-2. preprocessing + PII masking
    processed_text = clean_text(text)
    masked_text = mask_pii(processed_text)
    language = detect_language(processed_text)

    # 3. classification
    classification = resources.classification.predict(processed_text)

    # 4-5. sentiment + emotion
    sentiment = resources.sentiment.analyze(processed_text)
    emotion = resources.emotion.analyze(processed_text)

    # 6. NER
    entities = resources.ner.extract(processed_text)

    # 7. severity
    amount = float(metadata.get("amount", 0.0) or 0.0)
    resolution_status = metadata.get("resolution_status", "open")
    severity = resources.severity.assess(
        category=classification["category"], subcategory=classification["subcategory"],
        sentiment=sentiment["label"], emotion=emotion["label"],
        amount=amount, recent_similar_count=0, resolution_status=resolution_status,
    )

    # 8-9. embedding + similar complaints
    similar_complaints: List[Dict] = []
    if resources.similarity_ready:
        similar_complaints = similar_complaints_for_text(processed_text, resources.encoder, resources.vector_index)

    # 10. cluster assignment (best-effort lookup against DB-stored clusters)
    cluster_info = None

    # 11-12. emerging issue check + RAG retrieval
    emerging_issue = None
    retrieved_docs = retrieve_evidence(processed_text, category=classification["category"])

    with get_session() as db:
        emerging_issue = _find_emerging_issue_for(db, classification["category"], classification["subcategory"])

        # 13. LLM root-cause analysis (only if there's enough context to be useful)
        root_cause_result = None
        if run_llm:
            root_cause_result = analyze_root_cause(
                complaint_text=masked_text,
                category=classification["category"], subcategory=classification["subcategory"],
                severity=severity["severity"], similar_count=len(similar_complaints),
                retrieved_docs=retrieved_docs, emerging_issue=emerging_issue,
            )

        # 14. persistence
        if persist:
            complaint_data = {
                "complaint_id": complaint_id,
                "source": source,
                "original_text": text if settings.retain_original_text else masked_text,
                "processed_text": processed_text,
                "masked_text": masked_text,
                "language": language,
                "category": classification["category"],
                "subcategory": classification["subcategory"],
                "sentiment": sentiment["label"],
                "emotion": emotion["label"],
                "severity": severity["severity"],
                "product": metadata.get("product"),
                "region": metadata.get("region"),
                "platform": metadata.get("platform"),
                "amount": amount,
                "order_id": metadata.get("order_id"),
                "customer_id": metadata.get("customer_id"),
                "resolution_status": resolution_status,
                "processed_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            }
            if "timestamp" in metadata and metadata["timestamp"]:
                complaint_data["created_at"] = metadata["timestamp"]
            repo.upsert_complaint(db, complaint_data)

            repo.add_prediction(db, complaint_id, "category", classification["category"],
                                 classification["category_confidence"], classification["category_alternatives"],
                                 classification["needs_human_review"])
            repo.add_prediction(db, complaint_id, "subcategory", classification["subcategory"],
                                 classification["subcategory_confidence"], classification["subcategory_alternatives"])
            repo.add_prediction(db, complaint_id, "sentiment", sentiment["label"], sentiment["confidence"])
            repo.add_prediction(db, complaint_id, "emotion", emotion["label"], emotion["confidence"])
            repo.add_prediction(db, complaint_id, "severity", severity["severity"], severity["confidence"])

            if entities:
                repo.add_entities(db, complaint_id, entities)

            if root_cause_result:
                repo.add_analysis_result(db, {
                    "complaint_id": complaint_id,
                    "issue_summary": root_cause_result["issue_summary"],
                    "root_cause": root_cause_result["root_cause"],
                    "confidence": root_cause_result["confidence"],
                    "alternative_causes_json": _to_json(root_cause_result["alternative_causes"]),
                    "recommended_actions_json": _to_json(root_cause_result["recommended_actions"]),
                    "evidence_json": _to_json(root_cause_result["evidence"]),
                    "insufficient_evidence": root_cause_result["insufficient_evidence"],
                    "needs_human_review": root_cause_result["needs_human_review"],
                    "llm_mode": root_cause_result["llm_mode"],
                })

    needs_human_review = (
        classification["needs_human_review"] or severity["needs_human_review"]
        or (root_cause_result["needs_human_review"] if root_cause_result else False)
    )

    return {
        "complaint_id": complaint_id,
        "original_text": text,
        "processed_text": processed_text,
        "masked_text": masked_text,
        "language": language,
        "category": {
            "label": classification["category"], "confidence": classification["category_confidence"],
            "alternatives": classification["category_alternatives"],
        },
        "subcategory": {
            "label": classification["subcategory"], "confidence": classification["subcategory_confidence"],
            "alternatives": classification["subcategory_alternatives"],
        },
        "sentiment": sentiment,
        "emotion": emotion,
        "entities": entities,
        "severity": severity,
        "similar_complaints": similar_complaints,
        "cluster": cluster_info,
        "emerging_issue": emerging_issue,
        "retrieved_evidence": retrieved_docs,
        "root_cause_analysis": root_cause_result,
        "needs_human_review": needs_human_review,
    }


def _to_json(obj) -> str:
    import json
    return json.dumps(obj)
