"""Prompt templates for the LLM reasoning layer. Kept centralized so prompt
engineering changes happen in one place."""
from __future__ import annotations

from typing import Dict, List

ROOT_CAUSE_SYSTEM_PROMPT = """You are a customer-operations root-cause analysis assistant for an \
e-commerce/fintech platform. You are given a complaint (or complaint cluster), \
statistics about similar complaints, and retrieved internal knowledge-base \
documents (SOPs, incident reports, known bugs, FAQs).

STRICT RULES:
1. Base your root cause and evidence ONLY on the provided context (complaint \
   details, similar-complaint statistics, and retrieved documents). Do not \
   invent facts not present in the context.
2. You may ONLY cite document IDs that appear in the "RETRIEVED DOCUMENTS" \
   section below. Never invent a document ID.
3. If the retrieved documents and statistics do not clearly explain the \
   complaint, set "insufficient_evidence" to true and say so honestly in \
   root_cause rather than guessing confidently.
4. Provide a confidence score between 0 and 1 that honestly reflects how \
   well the evidence supports your conclusion.
5. Always propose at least one concrete, actionable recommended action.
6. Respond with a single JSON object matching this schema exactly:
{
  "issue_summary": string,
  "root_cause": string,
  "confidence": number,
  "alternative_causes": [string, ...],
  "recommended_actions": [string, ...],
  "evidence": [{"document_id": string, "title": string, "reason": string}, ...],
  "insufficient_evidence": boolean
}
"""


def build_root_cause_user_prompt(
    complaint_text: str,
    category: str,
    subcategory: str,
    severity: str,
    similar_count: int,
    emerging_issue: Dict = None,
    retrieved_docs: List[Dict] = None,
) -> str:
    retrieved_docs = retrieved_docs or []

    docs_block = "\n\n".join(
        f"[Document ID: {d['document_id']}] {d['title']} (type={d.get('doc_type')})\n{d['chunk_text']}"
        for d in retrieved_docs
    ) or "(No relevant knowledge-base documents were retrieved.)"

    issue_block = "(No emerging issue / anomaly detected for this complaint's category.)"
    if emerging_issue:
        issue_block = (
            f"Emerging issue detected: '{emerging_issue['title']}'. "
            f"Current volume: {emerging_issue['current_volume']} vs baseline "
            f"{emerging_issue['baseline_volume']:.1f} "
            f"({emerging_issue['increase_percent']:.0f}% increase). "
            f"Platform: {emerging_issue.get('platform')}, Region: {emerging_issue.get('region')}."
        )

    return f"""COMPLAINT:
"{complaint_text}"

CLASSIFICATION: category={category}, subcategory={subcategory}, severity={severity}

SIMILAR COMPLAINT STATISTICS:
{similar_count} similar complaints found in the historical dataset.

{issue_block}

RETRIEVED DOCUMENTS:
{docs_block}

Analyze this complaint and produce the JSON object described in the system prompt."""
