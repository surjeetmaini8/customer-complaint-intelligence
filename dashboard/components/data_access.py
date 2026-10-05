
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src.config import settings
from src.database.database import init_db, get_session
from src.database import repository as repo
from src.analytics import metrics as analytics_metrics
from src.analytics import aggregations


@st.cache_resource
def ensure_db():
    init_db()
    return True


@st.cache_data(ttl=30)
def load_summary_stats():
    ensure_db()
    with get_session() as db:
        return analytics_metrics.get_summary_stats(db)


@st.cache_data(ttl=30)
def load_distributions():
    ensure_db()
    with get_session() as db:
        return {
            "category": aggregations.category_distribution(db),
            "severity": aggregations.severity_distribution(db),
            "sentiment": aggregations.sentiment_distribution(db),
            "emotion": aggregations.emotion_distribution(db),
            "source": aggregations.source_distribution(db),
            "region": aggregations.region_distribution(db),
            "platform": aggregations.platform_distribution(db),
        }


@st.cache_data(ttl=30)
def load_daily_volume():
    ensure_db()
    with get_session() as db:
        return aggregations.daily_volume(db)


@st.cache_data(ttl=30)
def load_complaints_df(category=None, severity=None, sentiment=None, source=None,
                        region=None, search=None, limit=500):
    ensure_db()
    with get_session() as db:
        rows = repo.list_complaints(db, category=category, severity=severity, sentiment=sentiment,
                                     source=source, region=region, search=search, limit=limit)
        data = [{
            "complaint_id": c.complaint_id, "source": c.source, "original_text": c.original_text,
            "category": c.category, "subcategory": c.subcategory, "sentiment": c.sentiment,
            "emotion": c.emotion, "severity": c.severity, "product": c.product, "region": c.region,
            "platform": c.platform, "amount": c.amount, "order_id": c.order_id,
            "resolution_status": c.resolution_status, "cluster_id": c.cluster_id,
            "created_at": c.created_at,
        } for c in rows]
    return pd.DataFrame(data)


def get_complaint_detail(complaint_id: str):
    ensure_db()
    with get_session() as db:
        c = repo.get_complaint(db, complaint_id)
        if c is None:
            return None
        entities = [{"text": e.text, "label": e.label} for e in repo.get_entities(db, complaint_id)]
        analysis = repo.get_latest_analysis(db, complaint_id=complaint_id)
        cluster = repo.get_cluster(db, c.cluster_id) if c.cluster_id is not None else None
        complaint_obj = SimpleNamespace(
            complaint_id=c.complaint_id, source=c.source, original_text=c.original_text,
            processed_text=c.processed_text, category=c.category, subcategory=c.subcategory,
            sentiment=c.sentiment, emotion=c.emotion, severity=c.severity, product=c.product,
            region=c.region, platform=c.platform, amount=c.amount, order_id=c.order_id,
            customer_id=c.customer_id, resolution_status=c.resolution_status,
            cluster_id=c.cluster_id, created_at=c.created_at,
        )
        cluster_obj = None
        if cluster is not None:
            cluster_obj = SimpleNamespace(title=cluster.title, size=cluster.size, cluster_label=cluster.cluster_label)
        return {
            "complaint": complaint_obj, "entities": entities,
            "analysis": _analysis_dict(analysis), "cluster": cluster_obj,
        }


def _analysis_dict(a):
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


@st.cache_data(ttl=30)
def load_emerging_issues():
    ensure_db()
    with get_session() as db:
        issues = repo.list_emerging_issues(db, status="active")
        return [{
            "issue_id": i.issue_id, "title": i.title, "category": i.category, "subcategory": i.subcategory,
            "current_volume": i.current_volume, "baseline_volume": i.baseline_volume,
            "increase_percent": i.increase_percent, "anomaly_score": i.anomaly_score,
            "severity": i.severity, "platform": i.platform, "region": i.region,
            "first_detected_at": i.first_detected_at,
            "representative_complaint_ids": json.loads(i.representative_complaint_ids_json or "[]"),
        } for i in issues]


def get_issue_detail(issue_id: str):
    ensure_db()
    with get_session() as db:
        issue = repo.get_emerging_issue(db, issue_id)
        if issue is None:
            return None
        analysis = repo.get_latest_analysis(db, issue_id=issue_id)
        rep_ids = json.loads(issue.representative_complaint_ids_json or "[]")
        reps = [repo.get_complaint(db, cid) for cid in rep_ids]
        reps = [r for r in reps if r is not None]
        issue_obj = SimpleNamespace(
            issue_id=issue.issue_id, title=issue.title, category=issue.category,
            subcategory=issue.subcategory, current_volume=issue.current_volume,
            baseline_volume=issue.baseline_volume, increase_percent=issue.increase_percent,
            anomaly_score=issue.anomaly_score, severity=issue.severity, platform=issue.platform,
            region=issue.region, first_detected_at=issue.first_detected_at,
            window_start=issue.window_start, window_end=issue.window_end,
        )
        rep_objs = [SimpleNamespace(complaint_id=r.complaint_id, original_text=r.original_text,
                                     severity=r.severity, platform=r.platform, region=r.region)
                    for r in reps]
        return {"issue": issue_obj, "analysis": _analysis_dict(analysis), "representative_complaints": rep_objs}


@st.cache_data(ttl=30)
def load_clusters():
    ensure_db()
    with get_session() as db:
        clusters = repo.list_clusters(db)
        return [{
            "cluster_label": c.cluster_label, "title": c.title, "size": c.size,
            "dominant_category": c.dominant_category, "dominant_subcategory": c.dominant_subcategory,
            "dominant_sentiment": c.dominant_sentiment, "avg_severity_score": c.avg_severity_score,
            "keywords": json.loads(c.keywords_json or "[]"),
            "representative_ids": json.loads(c.representative_ids_json or "[]"),
        } for c in clusters]


@st.cache_data(ttl=30)
def load_knowledge_documents():
    ensure_db()
    with get_session() as db:
        docs = repo.list_knowledge_documents(db)
        return [{
            "document_id": d.document_id, "title": d.title, "doc_type": d.doc_type,
            "category": d.category, "tags": d.tags, "content": d.content,
        } for d in docs]


def submit_feedback(complaint_id: str, is_correct: bool, incorrect_fields=None, comment=None):
    ensure_db()
    with get_session() as db:
        repo.add_feedback(db, complaint_id, is_correct, incorrect_fields, comment)
