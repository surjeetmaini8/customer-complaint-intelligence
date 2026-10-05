"""Page 2 - Complaint Explorer: search/filter complaints and view full
per-complaint detail (predictions, entities, severity, similar complaints,
cluster, root-cause analysis, evidence, feedback)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.components import data_access as da
from src.classification.taxonomy import categories


def _severity_badge(sev: str) -> str:
    colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
    return f"{colors.get(sev, '⚪')} {sev}"


def render():
    st.title("🔍 Complaint Explorer")

    stats = da.load_summary_stats()
    if stats["total_complaints"] == 0:
        st.warning("No complaints found. Run the pipeline scripts first (see README).")
        return

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        category = c1.selectbox("Category", ["All"] + categories())
        severity = c2.selectbox("Severity", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
        sentiment = c3.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
        dist = da.load_distributions()
        source = c4.selectbox("Source", ["All"] + list(dist["source"].keys()))

        c5, c6 = st.columns(2)
        region = c5.selectbox("Region", ["All"] + list(dist["region"].keys()))
        search = c6.text_input("Search complaint text", "")

    df = da.load_complaints_df(
        category=None if category == "All" else category,
        severity=None if severity == "All" else severity,
        sentiment=None if sentiment == "All" else sentiment,
        source=None if source == "All" else source,
        region=None if region == "All" else region,
        search=search or None,
        limit=500,
    )

    st.caption(f"Showing {len(df)} complaints (max 500 rows)")

    if df.empty:
        st.info("No complaints match the selected filters.")
        return

    display_df = df[["complaint_id", "category", "subcategory", "severity", "sentiment",
                      "emotion", "region", "platform", "created_at"]].copy()
    st.dataframe(display_df, use_container_width=True, height=300)

    st.divider()
    st.subheader("Complaint Detail")
    selected_id = st.selectbox("Select a complaint to inspect", df["complaint_id"].tolist())

    if not selected_id:
        return

    detail = da.get_complaint_detail(selected_id)
    if detail is None:
        st.error("Complaint not found.")
        return

    c = detail["complaint"]

    st.markdown(f"**Complaint ID:** `{c.complaint_id}`  |  **Source:** {c.source}  |  "
                f"**Created:** {c.created_at}")
    st.info(c.original_text)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Category", c.category)
    col1.caption(f"Subcategory: {c.subcategory}")
    col2.metric("Sentiment", c.sentiment)
    col2.caption(f"Emotion: {c.emotion}")
    col3.metric("Severity", _severity_badge(c.severity))
    col4.metric("Amount", f"₹{c.amount:,.0f}" if c.amount else "-")
    col4.caption(f"Status: {c.resolution_status}")

    if detail["entities"]:
        st.markdown("**Extracted Entities**")
        st.dataframe(pd.DataFrame(detail["entities"]), use_container_width=True, height=150)

    if detail["cluster"]:
        cl = detail["cluster"]
        st.markdown(f"**Cluster:** {cl.title} (size: {cl.size}, id: {cl.cluster_label})")

    st.divider()
    st.subheader("🧠 Root-Cause Analysis")
    analysis = detail["analysis"]
    if analysis:
        mode_label = "🟢 Live LLM" if analysis["llm_mode"] == "live" else "🟡 Mock/Deterministic (no LLM API key configured)"
        st.caption(f"Analysis mode: {mode_label}")
        if analysis["needs_human_review"]:
            st.warning("⚠️ Human Review Recommended (low confidence)")
        st.markdown(f"**Summary:** {analysis['issue_summary']}")
        st.markdown(f"**Root Cause:** {analysis['root_cause']}")
        st.progress(min(max(analysis["confidence"], 0.0), 1.0), text=f"Confidence: {analysis['confidence']:.0%}")
        if analysis["alternative_causes"]:
            st.markdown("**Alternative Causes:** " + ", ".join(analysis["alternative_causes"]))
        if analysis["recommended_actions"]:
            st.markdown("**Recommended Actions:**")
            for a in analysis["recommended_actions"]:
                st.markdown(f"- {a}")
        if analysis["evidence"]:
            st.markdown("**Evidence:**")
            for e in analysis["evidence"]:
                st.markdown(f"- `{e['document_id']}` - {e['title']}: {e['reason']}")
        if analysis["insufficient_evidence"]:
            st.info("ℹ️ The system flagged that available evidence was insufficient for a fully confident root cause.")
    else:
        st.info("No root-cause analysis has been generated for this specific complaint yet "
                "(analysis is generated primarily at the cluster/emerging-issue level to control "
                "LLM usage - see README 'Cost Control'). You can run one on demand via the "
                "POST /complaints/analyze API endpoint.")

    st.divider()
    st.subheader("👍 Was this analysis correct?")
    fc1, fc2 = st.columns(2)
    if fc1.button("✅ Correct", key=f"correct_{selected_id}"):
        da.submit_feedback(selected_id, True)
        st.success("Thanks for your feedback!")
    if fc2.button("❌ Incorrect", key=f"incorrect_{selected_id}"):
        st.session_state[f"show_incorrect_{selected_id}"] = True

    if st.session_state.get(f"show_incorrect_{selected_id}"):
        wrong_fields = st.multiselect(
            "What was incorrect?",
            ["category", "severity", "sentiment", "root_cause", "recommendation"],
            key=f"wrong_fields_{selected_id}",
        )
        comment = st.text_area("Additional comments (optional)", key=f"comment_{selected_id}")
        if st.button("Submit feedback", key=f"submit_{selected_id}"):
            da.submit_feedback(selected_id, False, wrong_fields, comment)
            st.success("Feedback recorded. Thank you!")
            st.session_state[f"show_incorrect_{selected_id}"] = False


render()
