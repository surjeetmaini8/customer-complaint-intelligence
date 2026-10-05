"""Pydantic schemas for all structured AI outputs used across the platform."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Alternative(BaseModel):
    label: str
    confidence: float


class ComplaintPrediction(BaseModel):
    category: str
    category_confidence: float
    subcategory: str
    subcategory_confidence: float
    alternatives: List[Alternative] = Field(default_factory=list)
    needs_human_review: bool = False


class SentimentPrediction(BaseModel):
    label: str
    confidence: float
    needs_human_review: bool = False


class EmotionPrediction(BaseModel):
    label: str
    confidence: float
    needs_human_review: bool = False


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class SeverityPrediction(BaseModel):
    severity: str
    score: float
    confidence: float
    reasons: List[str]
    needs_human_review: bool = False


class SimilarComplaint(BaseModel):
    complaint_id: str
    similarity: float
    category: Optional[str] = None
    subcategory: Optional[str] = None
    severity: Optional[str] = None
    timestamp: Optional[str] = None
    text: Optional[str] = None


class ClusterInfo(BaseModel):
    cluster_label: int
    title: str
    size: int
    dominant_category: Optional[str] = None
    dominant_subcategory: Optional[str] = None
    dominant_sentiment: Optional[str] = None
    avg_severity_score: Optional[float] = None
    keywords: List[str] = Field(default_factory=list)


class EmergingIssue(BaseModel):
    issue_id: str
    title: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    current_volume: int
    baseline_volume: float
    increase_percent: float
    anomaly_score: float
    severity: str
    platform: Optional[str] = None
    region: Optional[str] = None


class RetrievedDocument(BaseModel):
    document_id: str
    title: str
    similarity: float
    chunk_text: str
    doc_type: Optional[str] = None
    category: Optional[str] = None


class Evidence(BaseModel):
    document_id: str
    title: str
    reason: str


class RootCauseAnalysis(BaseModel):
    issue_summary: str
    root_cause: str
    confidence: float
    alternative_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    insufficient_evidence: bool = False
    needs_human_review: bool = False
    llm_mode: str = "mock"   # live | mock


class Recommendation(BaseModel):
    action: str
    priority: str = "medium"   # low | medium | high
    rationale: Optional[str] = None
