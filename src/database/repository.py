"""
Repository layer: all reads/writes to the database go through here so the
API, dashboard, and pipeline share one consistent data-access contract.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import (
    Complaint, Prediction, EntityRecord, Cluster, EmergingIssue,
    KnowledgeDocument, AnalysisResult, HumanFeedback,
)


# ---------------------------------------------------------------- complaints
def upsert_complaint(db: Session, data: Dict[str, Any]) -> Complaint:
    existing = db.query(Complaint).filter_by(complaint_id=data["complaint_id"]).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    c = Complaint(**data)
    db.add(c)
    db.flush()
    return c


def get_complaint(db: Session, complaint_id: str) -> Optional[Complaint]:
    return db.query(Complaint).filter_by(complaint_id=complaint_id).first()


def list_complaints(
    db: Session,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    region: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[dt.datetime] = None,
    end_date: Optional[dt.datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Complaint]:
    q = db.query(Complaint)
    if category:
        q = q.filter(Complaint.category == category)
    if severity:
        q = q.filter(Complaint.severity == severity)
    if sentiment:
        q = q.filter(Complaint.sentiment == sentiment)
    if source:
        q = q.filter(Complaint.source == source)
    if region:
        q = q.filter(Complaint.region == region)
    if search:
        like = f"%{search}%"
        q = q.filter(Complaint.original_text.ilike(like))
    if start_date:
        q = q.filter(Complaint.created_at >= start_date)
    if end_date:
        q = q.filter(Complaint.created_at <= end_date)
    return q.order_by(Complaint.created_at.desc()).offset(offset).limit(limit).all()


def count_complaints(db: Session, **filters) -> int:
    """Count complaints using the same filter semantics as list_complaints."""
    q = db.query(func.count(Complaint.id))
    exact_fields = ("category", "severity", "sentiment", "source", "region")
    for field in exact_fields:
        val = filters.get(field)
        if val:
            q = q.filter(getattr(Complaint, field) == val)
    search = filters.get("search")
    if search:
        q = q.filter(Complaint.original_text.ilike(f"%{search}%"))
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    if start_date:
        q = q.filter(Complaint.created_at >= start_date)
    if end_date:
        q = q.filter(Complaint.created_at <= end_date)
    return q.scalar() or 0


# ---------------------------------------------------------------- predictions
def add_prediction(db: Session, complaint_id: str, task: str, label: str,
                    confidence: float, alternatives: Optional[List[dict]] = None,
                    needs_human_review: bool = False) -> Prediction:
    p = Prediction(
        complaint_id=complaint_id, task=task, label=label, confidence=confidence,
        alternatives_json=json.dumps(alternatives or []),
        needs_human_review=needs_human_review,
    )
    db.add(p)
    db.flush()
    return p


def get_predictions(db: Session, complaint_id: str) -> List[Prediction]:
    return db.query(Prediction).filter_by(complaint_id=complaint_id).all()


# ---------------------------------------------------------------- entities
def add_entities(db: Session, complaint_id: str, entities: List[dict]) -> None:
    for e in entities:
        db.add(EntityRecord(complaint_id=complaint_id, text=e["text"],
                             label=e["label"], start=e.get("start", 0), end=e.get("end", 0)))
    db.flush()


def get_entities(db: Session, complaint_id: str) -> List[EntityRecord]:
    return db.query(EntityRecord).filter_by(complaint_id=complaint_id).all()


# ---------------------------------------------------------------- clusters
def upsert_cluster(db: Session, data: Dict[str, Any]) -> Cluster:
    existing = db.query(Cluster).filter_by(cluster_label=data["cluster_label"]).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    c = Cluster(**data)
    db.add(c)
    db.flush()
    return c


def list_clusters(db: Session) -> List[Cluster]:
    return db.query(Cluster).order_by(Cluster.size.desc()).all()


def get_cluster(db: Session, cluster_label: int) -> Optional[Cluster]:
    return db.query(Cluster).filter_by(cluster_label=cluster_label).first()


# ---------------------------------------------------------------- emerging issues
def upsert_emerging_issue(db: Session, data: Dict[str, Any]) -> EmergingIssue:
    existing = db.query(EmergingIssue).filter_by(issue_id=data["issue_id"]).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    e = EmergingIssue(**data)
    db.add(e)
    db.flush()
    return e


def list_emerging_issues(db: Session, status: Optional[str] = "active") -> List[EmergingIssue]:
    q = db.query(EmergingIssue)
    if status:
        q = q.filter_by(status=status)
    return q.order_by(EmergingIssue.anomaly_score.desc()).all()


def get_emerging_issue(db: Session, issue_id: str) -> Optional[EmergingIssue]:
    return db.query(EmergingIssue).filter_by(issue_id=issue_id).first()


# ---------------------------------------------------------------- knowledge docs
def upsert_knowledge_document(db: Session, data: Dict[str, Any]) -> KnowledgeDocument:
    existing = db.query(KnowledgeDocument).filter_by(document_id=data["document_id"]).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    d = KnowledgeDocument(**data)
    db.add(d)
    db.flush()
    return d


def list_knowledge_documents(db: Session) -> List[KnowledgeDocument]:
    return db.query(KnowledgeDocument).all()


def get_knowledge_document(db: Session, document_id: str) -> Optional[KnowledgeDocument]:
    return db.query(KnowledgeDocument).filter_by(document_id=document_id).first()


# ---------------------------------------------------------------- analysis results
def add_analysis_result(db: Session, data: Dict[str, Any]) -> AnalysisResult:
    a = AnalysisResult(**data)
    db.add(a)
    db.flush()
    return a


def get_latest_analysis(db: Session, complaint_id: Optional[str] = None,
                         issue_id: Optional[str] = None) -> Optional[AnalysisResult]:
    q = db.query(AnalysisResult)
    if complaint_id:
        q = q.filter_by(complaint_id=complaint_id)
    if issue_id:
        q = q.filter_by(issue_id=issue_id)
    return q.order_by(AnalysisResult.created_at.desc()).first()


# ---------------------------------------------------------------- feedback
def add_feedback(db: Session, complaint_id: str, is_correct: bool,
                  incorrect_fields: Optional[List[str]] = None,
                  comment: Optional[str] = None) -> HumanFeedback:
    f = HumanFeedback(
        complaint_id=complaint_id, is_correct=is_correct,
        incorrect_fields_json=json.dumps(incorrect_fields or []), comment=comment,
    )
    db.add(f)
    db.flush()
    return f
