"""
Root-cause reasoning layer: combines classification, cluster/similar-
complaint statistics, emerging-issue signals, and RAG-retrieved evidence,
then asks the LLM to synthesize a root-cause analysis.

The LLM is NEVER allowed to invent evidence: only document IDs present in
the retrieved-documents list may be cited (enforced both via the prompt
and via a post-hoc validation step here that strips any hallucinated IDs).

If no LLM API key is configured, or a live call fails/returns unusable
JSON, this module falls back to a DETERMINISTIC MOCK analysis built
directly from the retrieved evidence and statistics - it is clearly
labeled `llm_mode: "mock"` and explicitly states that full LLM reasoning
is unavailable, rather than pretending to have produced a real
generative analysis (see Principle 2 & 6 in the project README).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.config import settings, get_logger
from src.llm.client import get_llm_client
from src.rag.prompts import ROOT_CAUSE_SYSTEM_PROMPT, build_root_cause_user_prompt

logger = get_logger(__name__)


def _validate_and_clean(result: Dict, retrieved_docs: List[Dict]) -> Dict:
    """Strips any evidence entries whose document_id was NOT actually
    retrieved, guarding against LLM hallucination of evidence IDs."""
    valid_ids = {d["document_id"] for d in retrieved_docs}
    evidence = result.get("evidence", []) or []
    cleaned_evidence = [e for e in evidence if e.get("document_id") in valid_ids]
    if len(cleaned_evidence) < len(evidence):
        logger.warning("Stripped %d hallucinated evidence citation(s) from LLM output",
                        len(evidence) - len(cleaned_evidence))
    result["evidence"] = cleaned_evidence
    if not cleaned_evidence and not result.get("insufficient_evidence"):
        # no valid evidence survived validation -> be honest about it
        result["insufficient_evidence"] = True
    return result


def _mock_root_cause(
    complaint_text: str, category: str, subcategory: str, severity: str,
    similar_count: int, emerging_issue: Optional[Dict], retrieved_docs: List[Dict],
) -> Dict:
    """Deterministic, evidence-grounded fallback used when no live LLM is
    configured. Explicitly labeled as mock - does not pretend to be a real
    generative analysis."""
    if not retrieved_docs:
        return {
            "issue_summary": f"Complaint categorized as {category} / {subcategory} "
                              f"with {severity} severity. {similar_count} similar complaints found.",
            "root_cause": (
                "LLM-based root-cause analysis is unavailable (no LLM API key configured, "
                "and no relevant knowledge-base evidence was retrieved for this complaint). "
                "A definitive root cause cannot be determined automatically."
            ),
            "confidence": 0.0,
            "alternative_causes": [],
            "recommended_actions": [
                "Route to the relevant operations team for manual investigation.",
                "Configure an LLM_API_KEY to enable full generative root-cause analysis.",
            ],
            "evidence": [],
            "insufficient_evidence": True,
            "llm_mode": "mock",
        }

    top_doc = retrieved_docs[0]
    evidence = [
        {"document_id": d["document_id"], "title": d["title"],
         "reason": f"Retrieved with similarity {d['similarity']:.2f} to the complaint text; "
                    f"document type '{d.get('doc_type')}' matches the complaint category."}
        for d in retrieved_docs[:3]
    ]

    issue_note = ""
    if emerging_issue:
        issue_note = (f" This aligns with a currently detected emerging issue "
                       f"('{emerging_issue['title']}', {emerging_issue['increase_percent']:.0f}% "
                       f"above baseline volume).")

    root_cause = (
        f"[MOCK / deterministic fallback - no live LLM configured] Based on the most relevant "
        f"retrieved document ('{top_doc['title']}', similarity {top_doc['similarity']:.2f}), the "
        f"likely root cause relates to the issue described in that document.{issue_note} This is a "
        f"template-based summary of retrieved evidence, not a full generative analysis - configure "
        f"LLM_API_KEY for richer, free-text root-cause reasoning."
    )

    return {
        "issue_summary": f"{len(retrieved_docs)} relevant knowledge-base document(s) found for this "
                          f"{category} / {subcategory} complaint ({similar_count} similar complaints "
                          f"in the historical dataset).",
        "root_cause": root_cause,
        "confidence": min(0.4 + top_doc["similarity"] * 0.4, 0.75),
        "alternative_causes": [d["title"] for d in retrieved_docs[1:3]],
        "recommended_actions": [
            "Review the linked knowledge-base document(s) for the documented resolution steps.",
            "Escalate per the relevant SOP if complaint volume continues to rise.",
        ],
        "evidence": evidence,
        "insufficient_evidence": top_doc["similarity"] < settings.get("rag.min_similarity", 0.15),
        "llm_mode": "mock",
    }


def analyze_root_cause(
    complaint_text: str,
    category: str,
    subcategory: str,
    severity: str,
    similar_count: int,
    retrieved_docs: List[Dict],
    emerging_issue: Optional[Dict] = None,
) -> Dict:
    client = get_llm_client()
    confidence_threshold = settings.get("llm.root_cause_confidence_threshold", 0.70)

    if client.is_live:
        user_prompt = build_root_cause_user_prompt(
            complaint_text, category, subcategory, severity, similar_count,
            emerging_issue, retrieved_docs,
        )
        result = client.generate_structured(ROOT_CAUSE_SYSTEM_PROMPT, user_prompt)
        if result is not None:
            try:
                result = _validate_and_clean(result, retrieved_docs)
                result["llm_mode"] = "live"
                result["needs_human_review"] = result.get("confidence", 0.0) < confidence_threshold
                return result
            except Exception as e:
                logger.warning("Malformed live LLM result (%s); falling back to mock analysis", e)
        else:
            logger.info("Live LLM call unavailable/failed; falling back to mock analysis")

    result = _mock_root_cause(complaint_text, category, subcategory, severity,
                               similar_count, emerging_issue, retrieved_docs)
    result["needs_human_review"] = result.get("confidence", 0.0) < confidence_threshold
    return result
