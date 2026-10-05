"""SQLAlchemy ORM models for the complaint intelligence platform."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(32), unique=True, index=True, nullable=False)
    source = Column(String(32), index=True)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    masked_text = Column(Text)
    language = Column(String(8), default="en")

    category = Column(String(64), index=True)
    subcategory = Column(String(128))
    sentiment = Column(String(16), index=True)
    emotion = Column(String(24))
    severity = Column(String(16), index=True)

    product = Column(String(64))
    region = Column(String(64), index=True)
    platform = Column(String(32))
    amount = Column(Float, default=0.0)
    order_id = Column(String(32))
    customer_id = Column(String(32))
    resolution_status = Column(String(24), default="open")

    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(String(32), nullable=True)

    cluster_id = Column(Integer, nullable=True, index=True)
    embedding_index = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)

    predictions = relationship("Prediction", back_populates="complaint", cascade="all, delete-orphan")
    entities = relationship("EntityRecord", back_populates="complaint", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="complaint", cascade="all, delete-orphan")
    feedback = relationship("HumanFeedback", back_populates="complaint", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_complaints_created_source", "created_at", "source"),
    )


class Prediction(Base):
    """Stores raw model prediction outputs (classification/sentiment/emotion/severity)."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(32), ForeignKey("complaints.complaint_id"), index=True)
    task = Column(String(32))          # category | subcategory | sentiment | emotion | severity
    label = Column(String(128))
    confidence = Column(Float)
    alternatives_json = Column(Text)   # JSON-encoded list of alternatives
    needs_human_review = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    complaint = relationship("Complaint", back_populates="predictions")


class EntityRecord(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(32), ForeignKey("complaints.complaint_id"), index=True)
    text = Column(String(256))
    label = Column(String(32))
    start = Column(Integer)
    end = Column(Integer)

    complaint = relationship("Complaint", back_populates="entities")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_label = Column(Integer, unique=True, index=True)
    title = Column(String(256))
    size = Column(Integer)
    dominant_category = Column(String(64))
    dominant_subcategory = Column(String(128))
    dominant_sentiment = Column(String(16))
    avg_severity_score = Column(Float)
    keywords_json = Column(Text)
    representative_ids_json = Column(Text)
    time_start = Column(DateTime, nullable=True)
    time_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class EmergingIssue(Base):
    __tablename__ = "emerging_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(String(32), unique=True, index=True)
    title = Column(String(256))
    category = Column(String(64))
    subcategory = Column(String(128))
    current_volume = Column(Integer)
    baseline_volume = Column(Float)
    increase_percent = Column(Float)
    anomaly_score = Column(Float)
    severity = Column(String(16))
    platform = Column(String(32), nullable=True)
    region = Column(String(64), nullable=True)
    first_detected_at = Column(DateTime)
    window_start = Column(DateTime)
    window_end = Column(DateTime)
    representative_complaint_ids_json = Column(Text)
    status = Column(String(16), default="active")   # active | resolved
    created_at = Column(DateTime, default=utcnow)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(32), unique=True, index=True)
    title = Column(String(256))
    doc_type = Column(String(32))     # faq | sop | incident | bug | troubleshooting
    category = Column(String(64), nullable=True)
    tags = Column(String(256), nullable=True)
    content = Column(Text)
    file_path = Column(String(256))
    created_date = Column(DateTime, default=utcnow)
    indexed = Column(Boolean, default=False)


class AnalysisResult(Base):
    """Stores the full LLM-generated root-cause analysis for a complaint/issue."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(32), ForeignKey("complaints.complaint_id"), nullable=True, index=True)
    issue_id = Column(String(32), nullable=True, index=True)

    issue_summary = Column(Text)
    root_cause = Column(Text)
    confidence = Column(Float)
    alternative_causes_json = Column(Text)
    recommended_actions_json = Column(Text)
    evidence_json = Column(Text)
    insufficient_evidence = Column(Boolean, default=False)
    needs_human_review = Column(Boolean, default=False)
    llm_mode = Column(String(16), default="mock")   # live | mock
    created_at = Column(DateTime, default=utcnow)

    complaint = relationship("Complaint", back_populates="analysis_results")


class HumanFeedback(Base):
    __tablename__ = "human_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(32), ForeignKey("complaints.complaint_id"), index=True)
    is_correct = Column(Boolean)
    incorrect_fields_json = Column(Text)  # e.g. ["category", "severity"]
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    complaint = relationship("Complaint", back_populates="feedback")
