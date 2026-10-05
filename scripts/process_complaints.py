"""
Runs the full analysis pipeline over the generated complaint dataset:
classification, sentiment, emotion, NER, severity, embeddings, clustering,
emerging-issue detection, and (for representative complaints/issues only)
RAG + LLM root-cause analysis. Persists everything to the database so the
API and dashboard can serve real, pre-computed results.

Usage:
    python scripts/process_complaints.py
    python scripts/process_complaints.py --limit 2000   # faster dev iteration
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import settings, get_logger
from src.database.database import init_db, get_session
from src.database import repository as repo
from src.preprocessing.cleaner import clean_text
from src.preprocessing.pii import mask_pii
from src.preprocessing.language import detect_language
from src.preprocessing.deduplication import deduplicate_dataframe
from src.classification.inference import get_classification_service
from src.sentiment.analyzer import get_sentiment_analyzer
from src.emotion.analyzer import get_emotion_analyzer
from src.ner.extractor import get_ner_extractor
from src.severity.predictor import get_severity_service
from src.embeddings.encoder import EmbeddingEncoder
from src.clustering.clusterer import run_clustering, summarize_clusters
from src.anomaly.detector import detect_emerging_issues
from src.rag.pipeline import retrieve_evidence
from src.llm.reasoning import analyze_root_cause

logger = get_logger(__name__)


def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing %d complaints...", len(df))
    df = df.copy()
    df["complaint_text"] = df["complaint_text"].fillna("")
    df["processed_text"] = df["complaint_text"].apply(clean_text)
    df["masked_text"] = df["processed_text"].apply(mask_pii)
    df["language"] = df["processed_text"].apply(detect_language)
    df = deduplicate_dataframe(df, text_col="processed_text")

    logger.info("Running classification (predicted, not ground truth)...")
    clf = get_classification_service()
    predicted_categories, predicted_subcats = [], []
    cat_conf, sub_conf = [], []
    for text in df["processed_text"]:
        result = clf.predict(text)
        predicted_categories.append(result["category"])
        predicted_subcats.append(result["subcategory"])
        cat_conf.append(result["category_confidence"])
        sub_conf.append(result["subcategory_confidence"])
    df["pred_category"] = predicted_categories
    df["pred_subcategory"] = predicted_subcats
    df["category_confidence"] = cat_conf
    df["subcategory_confidence"] = sub_conf

    logger.info("Running sentiment analysis...")
    sentiment_analyzer = get_sentiment_analyzer()
    sentiments = sentiment_analyzer.analyze_batch(df["processed_text"].tolist())
    df["pred_sentiment"] = [s["label"] for s in sentiments]
    df["sentiment_confidence"] = [s["confidence"] for s in sentiments]

    logger.info("Running emotion analysis...")
    emotion_analyzer = get_emotion_analyzer()
    emotions = emotion_analyzer.analyze_batch(df["processed_text"].tolist())
    df["pred_emotion"] = [e["label"] for e in emotions]
    df["emotion_confidence"] = [e["confidence"] for e in emotions]

    logger.info("Running NER extraction...")
    ner = get_ner_extractor()
    df["entities"] = ner.extract_batch(df["processed_text"].tolist())

    logger.info("Running severity assessment...")
    severity_service = get_severity_service()
    freq_counts = df.groupby(["pred_category", "pred_subcategory"])["complaint_id"].transform("count")
    severities, sev_scores, sev_conf, sev_reasons = [], [], [], []
    for i, row in df.iterrows():
        result = severity_service.assess(
            category=row["pred_category"], subcategory=row["pred_subcategory"],
            sentiment=row["pred_sentiment"], emotion=row["pred_emotion"],
            amount=row.get("amount", 0.0) or 0.0,
            recent_similar_count=int(freq_counts.loc[i]),
            resolution_status=row.get("resolution_status", "open"),
        )
        severities.append(result["severity"])
        sev_scores.append(result["score"])
        sev_conf.append(result["confidence"])
        sev_reasons.append(result["reasons"])
    df["pred_severity"] = severities
    df["severity_score"] = sev_scores
    df["severity_confidence"] = sev_conf
    df["severity_reasons"] = sev_reasons

    return df


def persist_complaints(df: pd.DataFrame) -> None:
    logger.info("Persisting %d complaints to the database...", len(df))
    with get_session() as db:
        for _, row in df.iterrows():
            repo.upsert_complaint(db, {
                "complaint_id": row["complaint_id"],
                "source": row.get("source"),
                "original_text": row["complaint_text"],
                "processed_text": row["processed_text"],
                "masked_text": row["masked_text"],
                "language": row["language"],
                "category": row["pred_category"],
                "subcategory": row["pred_subcategory"],
                "sentiment": row["pred_sentiment"],
                "emotion": row["pred_emotion"],
                "severity": row["pred_severity"],
                "product": row.get("product"),
                "region": row.get("region"),
                "platform": row.get("platform"),
                "amount": float(row.get("amount", 0.0) or 0.0),
                "order_id": row.get("order_id"),
                "customer_id": row.get("customer_id"),
                "resolution_status": row.get("resolution_status", "open"),
                "is_duplicate": bool(row.get("is_duplicate", False)),
                "created_at": pd.to_datetime(row["timestamp"]).to_pydatetime(),
                "processed_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            })
            repo.add_prediction(db, row["complaint_id"], "category", row["pred_category"],
                                 float(row["category_confidence"]))
            repo.add_prediction(db, row["complaint_id"], "subcategory", row["pred_subcategory"],
                                 float(row["subcategory_confidence"]))
            repo.add_prediction(db, row["complaint_id"], "sentiment", row["pred_sentiment"],
                                 float(row["sentiment_confidence"]))
            repo.add_prediction(db, row["complaint_id"], "emotion", row["pred_emotion"],
                                 float(row["emotion_confidence"]))
            repo.add_prediction(db, row["complaint_id"], "severity", row["pred_severity"],
                                 float(row["severity_confidence"]))
            if row["entities"]:
                repo.add_entities(db, row["complaint_id"], row["entities"])
    logger.info("Persisted complaints, predictions, and entities.")


def run_clustering_stage(df: pd.DataFrame, encoder: EmbeddingEncoder, embeddings: np.ndarray) -> np.ndarray:
    logger.info("Running clustering on %d complaint embeddings...", len(df))
    labels = run_clustering(embeddings)
    cluster_df = df.drop(columns=["category", "subcategory", "sentiment", "severity"], errors="ignore")
    cluster_df = cluster_df.rename(columns={"pred_category": "category", "pred_subcategory": "subcategory",
                                             "pred_sentiment": "sentiment", "pred_severity": "severity"})
    summaries = summarize_clusters(cluster_df, labels, text_col="processed_text")

    with get_session() as db:
        for s in summaries:
            repo.upsert_cluster(db, {
                "cluster_label": s["cluster_label"],
                "title": s["title"],
                "size": s["size"],
                "dominant_category": s["dominant_category"],
                "dominant_subcategory": s["dominant_subcategory"],
                "dominant_sentiment": s["dominant_sentiment"],
                "avg_severity_score": s["avg_severity_score"],
                "keywords_json": json.dumps(s["keywords"]),
                "representative_ids_json": json.dumps(s["representative_ids"]),
                "time_start": s["time_start"].to_pydatetime() if s["time_start"] is not None else None,
                "time_end": s["time_end"].to_pydatetime() if s["time_end"] is not None else None,
            })
        # persist cluster assignment on each complaint
        for cid, label in zip(df["complaint_id"], labels):
            c = repo.get_complaint(db, cid)
            if c is not None:
                c.cluster_id = int(label)
    logger.info("Persisted %d clusters.", len(summaries))
    return labels


def run_emerging_issues_stage(df: pd.DataFrame) -> list:
    logger.info("Running emerging issue / anomaly detection...")
    detect_df = df.drop(columns=["category", "subcategory"], errors="ignore")
    detect_df = detect_df.rename(columns={"pred_category": "category", "pred_subcategory": "subcategory"})
    issues = detect_emerging_issues(detect_df)
    with get_session() as db:
        for issue in issues:
            repo.upsert_emerging_issue(db, {
                "issue_id": issue["issue_id"],
                "title": issue["title"],
                "category": issue.get("category"),
                "subcategory": issue.get("subcategory"),
                "current_volume": issue["current_volume"],
                "baseline_volume": issue["baseline_volume"],
                "increase_percent": issue["increase_percent"],
                "anomaly_score": issue["anomaly_score"],
                "severity": issue["severity"],
                "platform": issue.get("platform"),
                "region": issue.get("region"),
                "first_detected_at": issue["first_detected_at"],
                "window_start": issue["window_start"],
                "window_end": issue["window_end"],
                "representative_complaint_ids_json": json.dumps(issue["representative_complaint_ids"]),
                "status": "active",
            })
    logger.info("Persisted %d emerging issues.", len(issues))
    return issues


def run_root_cause_for_issues(issues: list, df: pd.DataFrame, max_issues: int = 20) -> None:
    logger.info("Running LLM root-cause analysis for top %d emerging issues...", min(len(issues), max_issues))
    with get_session() as db:
        for issue in issues[:max_issues]:
            rep_ids = issue["representative_complaint_ids"]
            if not rep_ids:
                continue
            rep_text_row = df[df["complaint_id"] == rep_ids[0]]
            if rep_text_row.empty:
                continue
            rep_text = rep_text_row.iloc[0]["masked_text"]
            similar_count = issue["current_volume"]

            retrieved_docs = retrieve_evidence(rep_text, category=issue.get("category"))
            result = analyze_root_cause(
                complaint_text=rep_text, category=issue.get("category", ""),
                subcategory=issue.get("subcategory", ""), severity=issue["severity"],
                similar_count=similar_count, retrieved_docs=retrieved_docs,
                emerging_issue=issue,
            )
            repo.add_analysis_result(db, {
                "issue_id": issue["issue_id"],
                "issue_summary": result["issue_summary"],
                "root_cause": result["root_cause"],
                "confidence": result["confidence"],
                "alternative_causes_json": json.dumps(result["alternative_causes"]),
                "recommended_actions_json": json.dumps(result["recommended_actions"]),
                "evidence_json": json.dumps(result["evidence"]),
                "insufficient_evidence": result["insufficient_evidence"],
                "needs_human_review": result["needs_human_review"],
                "llm_mode": result["llm_mode"],
            })
    logger.info("Root-cause analysis complete for emerging issues.")


def run_root_cause_for_top_clusters(labels: np.ndarray, df: pd.DataFrame, max_clusters: int = 15) -> None:
    from collections import Counter
    counts = Counter(l for l in labels if l != -1)
    top_clusters = [c for c, _ in counts.most_common(max_clusters)]
    logger.info("Running LLM root-cause analysis for top %d clusters...", len(top_clusters))

    df = df.copy()
    df["_cluster_label"] = labels
    with get_session() as db:
        for cluster_label in top_clusters:
            group = df[df["_cluster_label"] == cluster_label]
            if group.empty:
                continue
            rep = group.iloc[0]
            retrieved_docs = retrieve_evidence(rep["masked_text"], category=rep["pred_category"])
            result = analyze_root_cause(
                complaint_text=rep["masked_text"], category=rep["pred_category"],
                subcategory=rep["pred_subcategory"], severity=rep["pred_severity"],
                similar_count=len(group), retrieved_docs=retrieved_docs, emerging_issue=None,
            )
            repo.add_analysis_result(db, {
                "complaint_id": rep["complaint_id"],
                "issue_summary": result["issue_summary"],
                "root_cause": result["root_cause"],
                "confidence": result["confidence"],
                "alternative_causes_json": json.dumps(result["alternative_causes"]),
                "recommended_actions_json": json.dumps(result["recommended_actions"]),
                "evidence_json": json.dumps(result["evidence"]),
                "insufficient_evidence": result["insufficient_evidence"],
                "needs_human_review": result["needs_human_review"],
                "llm_mode": result["llm_mode"],
            })
    logger.info("Root-cause analysis complete for top clusters.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (dev/testing)")
    args = parser.parse_args()

    csv_path = settings.path("paths.complaints_csv")
    if not csv_path.exists():
        logger.error("No dataset found at %s. Run scripts/generate_data.py first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    init_db()

    df = annotate_dataframe(df)
    persist_complaints(df)

    faiss_dir = settings.path("paths.faiss_dir")
    try:
        encoder = EmbeddingEncoder.load(faiss_dir / "complaint_encoder")
    except Exception:
        logger.info("No persisted encoder found; fitting a fresh one for clustering.")
        encoder = EmbeddingEncoder()
        encoder.fit(df["processed_text"].tolist())
    embeddings = encoder.encode(df["processed_text"].tolist())

    labels = run_clustering_stage(df, encoder, embeddings)
    issues = run_emerging_issues_stage(df)

    run_root_cause_for_issues(issues, df)
    run_root_cause_for_top_clusters(labels, df)

    print("\n=== Processing complete ===")
    print(f"Processed {len(df)} complaints")
    print(f"Detected {len(issues)} emerging issues")
    print(f"Clusters found: {len(set(l for l in labels if l != -1))}")


if __name__ == "__main__":
    main()
