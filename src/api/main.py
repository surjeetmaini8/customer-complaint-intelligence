"""
FastAPI backend exposing the complaint intelligence platform.

Run with:
    python -m uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import datetime as dt
import json
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Header
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.config import settings, get_logger
from src.database.database import init_db, get_session_dependency
from src.database import repository as repo
from src.pipeline.complaint_pipeline import process_complaint
from src.rag.pipeline import retrieve_evidence
from src.analytics import metrics as analytics_metrics
from src.analytics import aggregations

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Complaint Intelligence & Root-Cause Analysis Platform",
    description="Transforms raw customer complaints into structured, prioritized, "
                "evidence-backed business intelligence. All data is synthetic demo data.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("FastAPI app started")


def get_db():
    yield from get_session_dependency()


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Optional API-key protection for deployments outside a trusted local network."""
    if not settings.api_require_key:
        return
    if not settings.api_key:
        raise HTTPException(503, "API key protection is enabled but API_KEY is not configured")
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")


# ============================================================== schemas
class ComplaintCreateRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=20000)
    source: str = Field(default="api", min_length=1, max_length=32)
    product: Optional[str] = None
    region: Optional[str] = None
    platform: Optional[str] = None
    amount: Optional[float] = 0.0
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    resolution_status: Optional[str] = "open"
    run_llm: bool = True

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("text must contain at least 3 non-whitespace characters")
        return value


class FeedbackRequest(BaseModel):
    complaint_id: str
    is_correct: bool
    incorrect_fields: Optional[List[str]] = None
    comment: Optional[str] = None


# ============================================================== helpers
def _complaint_to_dict(c) -> dict:
    return {
        "complaint_id": c.complaint_id, "source": c.source, "original_text": c.original_text if settings.get("api.expose_original_text", True) else None,
        "processed_text": c.processed_text, "category": c.category, "subcategory": c.subcategory,
        "sentiment": c.sentiment, "emotion": c.emotion, "severity": c.severity,
        "product": c.product, "region": c.region, "platform": c.platform, "amount": c.amount,
        "order_id": c.order_id, "customer_id": c.customer_id,
        "resolution_status": c.resolution_status, "cluster_id": c.cluster_id,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def _issue_to_dict(i) -> dict:
    return {
        "issue_id": i.issue_id, "title": i.title, "category": i.category, "subcategory": i.subcategory,
        "current_volume": i.current_volume, "baseline_volume": i.baseline_volume,
        "increase_percent": i.increase_percent, "anomaly_score": i.anomaly_score,
        "severity": i.severity, "platform": i.platform, "region": i.region,
        "first_detected_at": i.first_detected_at.isoformat() if i.first_detected_at else None,
        "representative_complaint_ids": json.loads(i.representative_complaint_ids_json or "[]"),
        "status": i.status,
    }


def _cluster_to_dict(c) -> dict:
    return {
        "cluster_label": c.cluster_label, "title": c.title, "size": c.size,
        "dominant_category": c.dominant_category, "dominant_subcategory": c.dominant_subcategory,
        "dominant_sentiment": c.dominant_sentiment, "avg_severity_score": c.avg_severity_score,
        "keywords": json.loads(c.keywords_json or "[]"),
        "representative_ids": json.loads(c.representative_ids_json or "[]"),
        "time_start": c.time_start.isoformat() if c.time_start else None,
        "time_end": c.time_end.isoformat() if c.time_end else None,
    }


def _analysis_to_dict(a) -> Optional[dict]:
    if a is None:
        return None
    return {
        "issue_summary": a.issue_summary, "root_cause": a.root_cause, "confidence": a.confidence,
        "alternative_causes": json.loads(a.alternative_causes_json or "[]"),
        "recommended_actions": json.loads(a.recommended_actions_json or "[]"),
        "evidence": json.loads(a.evidence_json or "[]"),
        "insufficient_evidence": a.insufficient_evidence, "needs_human_review": a.needs_human_review,
        "llm_mode": a.llm_mode,
    }


# ============================================================== endpoints
@app.get("/health")
def health():
    from src.llm.client import get_llm_client
    return {
        "status": "ok",
        "llm_mode": get_llm_client().mode,
        "database": settings.database_url,
    }


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    return analytics_metrics.get_summary_stats(db)


@app.post("/complaints")
def create_complaint(payload: ComplaintCreateRequest, _: None = Depends(require_api_key)):
    metadata = payload.model_dump(exclude={"text", "source", "run_llm"})
    result = process_complaint(payload.text, source=payload.source, metadata=metadata,
                                persist=True, run_llm=payload.run_llm)
    return result


@app.get("/complaints")
def list_complaints(
    category: Optional[str] = None, severity: Optional[str] = None,
    sentiment: Optional[str] = None, source: Optional[str] = None, region: Optional[str] = None,
    search: Optional[str] = None, limit: int = Query(50, le=500), offset: int = 0,
    db: Session = Depends(get_db),
):
    complaints = repo.list_complaints(db, category=category, severity=severity, sentiment=sentiment,
                                       source=source, region=region, search=search,
                                       limit=limit, offset=offset)
    total = repo.count_complaints(
        db, category=category, severity=severity, sentiment=sentiment,
        source=source, region=region, search=search
    )
    return {"total": total, "items": [_complaint_to_dict(c) for c in complaints]}


@app.get("/complaints/{complaint_id}")
def get_complaint(complaint_id: str, db: Session = Depends(get_db)):
    c = repo.get_complaint(db, complaint_id)
    if c is None:
        raise HTTPException(404, "Complaint not found")
    entities = [{"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                for e in repo.get_entities(db, complaint_id)]
    analysis = repo.get_latest_analysis(db, complaint_id=complaint_id)
    result = _complaint_to_dict(c)
    result["entities"] = entities
    result["root_cause_analysis"] = _analysis_to_dict(analysis)
    if c.cluster_id is not None:
        cluster = repo.get_cluster(db, c.cluster_id)
        result["cluster"] = _cluster_to_dict(cluster) if cluster else None
    return result


@app.post("/complaints/analyze")
def analyze_complaint(payload: ComplaintCreateRequest, _: None = Depends(require_api_key)):
    """Analyze a complaint without writing it to the database."""
    metadata = payload.model_dump(exclude={"text", "source", "run_llm"})
    result = process_complaint(payload.text, source=payload.source, metadata=metadata,
                                persist=False, run_llm=payload.run_llm)
    return result


@app.get("/complaints/{complaint_id}/similar")
def get_similar_complaints(complaint_id: str, top_k: int = 10, db: Session = Depends(get_db)):
    from src.pipeline.complaint_pipeline import get_pipeline_resources
    from src.similarity.search import similar_complaints_for_text

    c = repo.get_complaint(db, complaint_id)
    if c is None:
        raise HTTPException(404, "Complaint not found")
    resources = get_pipeline_resources()
    if not resources.similarity_ready:
        raise HTTPException(503, "Similarity index not built. Run scripts/build_index.py.")
    results = similar_complaints_for_text(c.processed_text or c.original_text, resources.encoder,
                                           resources.vector_index, top_k=top_k)
    enriched = []
    for r in results:
        if r["complaint_id"] == complaint_id:
            continue
        sc = repo.get_complaint(db, r["complaint_id"])
        if sc:
            enriched.append({**_complaint_to_dict(sc), "similarity": r["similarity"]})
    return {"complaint_id": complaint_id, "similar_complaints": enriched}


@app.get("/issues")
@app.get("/issues/emerging")
def list_emerging_issues(status: Optional[str] = "active", db: Session = Depends(get_db)):
    issues = repo.list_emerging_issues(db, status=status)
    return {"items": [_issue_to_dict(i) for i in issues]}


@app.get("/issues/{issue_id}")
def get_issue(issue_id: str, db: Session = Depends(get_db)):
    issue = repo.get_emerging_issue(db, issue_id)
    if issue is None:
        raise HTTPException(404, "Issue not found")
    analysis = repo.get_latest_analysis(db, issue_id=issue_id)
    rep_ids = json.loads(issue.representative_complaint_ids_json or "[]")
    representative_complaints = [_complaint_to_dict(repo.get_complaint(db, cid))
                                  for cid in rep_ids if repo.get_complaint(db, cid)]
    result = _issue_to_dict(issue)
    result["root_cause_analysis"] = _analysis_to_dict(analysis)
    result["representative_complaints"] = representative_complaints
    return result


@app.get("/clusters")
def list_clusters(db: Session = Depends(get_db)):
    clusters = repo.list_clusters(db)
    return {"items": [_cluster_to_dict(c) for c in clusters]}


@app.get("/clusters/{cluster_label}")
def get_cluster(cluster_label: int, db: Session = Depends(get_db)):
    cluster = repo.get_cluster(db, cluster_label)
    if cluster is None:
        raise HTTPException(404, "Cluster not found")
    return _cluster_to_dict(cluster)


@app.get("/knowledge/search")
def search_knowledge(q: str, top_k: int = 5, category: Optional[str] = None):
    results = retrieve_evidence(q, category=category, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/knowledge/documents")
def list_knowledge_documents(db: Session = Depends(get_db)):
    docs = repo.list_knowledge_documents(db)
    return {"items": [
        {"document_id": d.document_id, "title": d.title, "doc_type": d.doc_type,
         "category": d.category, "tags": d.tags, "created_date": str(d.created_date)}
        for d in docs
    ]}


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    f = repo.add_feedback(db, payload.complaint_id, payload.is_correct,
                           payload.incorrect_fields, payload.comment)
    return {"status": "recorded", "feedback_id": f.id}


@app.post("/rebuild-index")
def rebuild_index():
    import subprocess
    try:
        subprocess.run(["python3", "scripts/build_index.py"], cwd=str(settings.root), check=True)
        return {"status": "rebuilt"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Index rebuild failed: {e}")


@app.post("/process-all")
def process_all(limit: Optional[int] = None):
    import subprocess
    cmd = ["python3", "scripts/process_complaints.py"]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        subprocess.run(cmd, cwd=str(settings.root), check=True)
        return {"status": "processed"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Processing failed: {e}")


@app.get("/analytics/distributions")
def get_distributions(db: Session = Depends(get_db)):
    return {
        "category": aggregations.category_distribution(db),
        "severity": aggregations.severity_distribution(db),
        "sentiment": aggregations.sentiment_distribution(db),
        "emotion": aggregations.emotion_distribution(db),
        "source": aggregations.source_distribution(db),
        "region": aggregations.region_distribution(db),
        "platform": aggregations.platform_distribution(db),
    }


@app.get("/analytics/volume/daily")
def get_daily_volume(db: Session = Depends(get_db)):
    return {"items": aggregations.daily_volume(db)}
