"""Page 3 - Emerging Issues (alert-style) + Issue Investigation drill-down,
and a Clusters tab for browsing complaint clusters."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.components import data_access as da

SEVERITY_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}


def _render_issue_card(issue: dict):
    emoji = SEVERITY_EMOJI.get(issue["severity"], "⚪")
    with st.container(border=True):
        st.markdown(f"### {emoji} {issue['title']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Complaints", issue["current_volume"])
        c2.metric("Expected (Baseline)", f"{issue['baseline_volume']:.0f}")
        c3.metric("Increase", f"+{issue['increase_percent']:.0f}%")
        c4.metric("Severity", issue["severity"])
        st.caption(f"First detected: {issue['first_detected_at']}  |  "
                   f"Affected: {issue.get('platform') or 'All platforms'} / {issue.get('region') or 'All regions'}")
        if st.button("🔎 Investigate", key=f"investigate_{issue['issue_id']}"):
            st.session_state["selected_issue_id"] = issue["issue_id"]
            st.session_state["issues_tab"] = "investigate"


def _render_emerging_issues_tab():
    st.subheader("🚨 Emerging Issues")
    st.caption("Sudden statistically-significant spikes in complaint volume, detected via "
               "rolling-baseline z-score analysis (see src/anomaly/detector.py).")
    issues = da.load_emerging_issues()
    if not issues:
        st.success("✅ No emerging issues currently detected.")
        return
    issues_sorted = sorted(issues, key=lambda i: i["anomaly_score"], reverse=True)
    for issue in issues_sorted:
        _render_issue_card(issue)


def _render_investigation_tab():
    st.subheader("🔬 Issue Investigation")
    issues = da.load_emerging_issues()
    if not issues:
        st.info("No emerging issues to investigate.")
        return

    options = {f"{i['title']} ({i['issue_id']})": i["issue_id"] for i in issues}
    default_id = st.session_state.get("selected_issue_id")
    default_label = next((k for k, v in options.items() if v == default_id), list(options.keys())[0])
    selected_label = st.selectbox("Select an issue", list(options.keys()),
                                   index=list(options.keys()).index(default_label))
    issue_id = options[selected_label]

    detail = da.get_issue_detail(issue_id)
    if detail is None:
        st.error("Issue not found.")
        return

    issue = detail["issue"]
    emoji = SEVERITY_EMOJI.get(issue.severity, "⚪")
    st.markdown(f"## {emoji} {issue.title}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Complaint Count", issue.current_volume)
    c2.metric("Growth", f"+{issue.increase_percent:.0f}%")
    c3.metric("Severity", issue.severity)
    c4.metric("Anomaly Score", f"{issue.anomaly_score:.2f}")

    st.markdown(f"**Affected platform:** {issue.platform or 'N/A'}  |  "
                f"**Affected region:** {issue.region or 'N/A'}  |  "
                f"**Window:** {issue.window_start} → {issue.window_end}")

    st.divider()
    st.markdown("### Representative Complaints")
    reps = detail["representative_complaints"]
    if reps:
        rep_df = pd.DataFrame([{
            "complaint_id": r.complaint_id, "text": r.original_text,
            "severity": r.severity, "platform": r.platform, "region": r.region,
        } for r in reps])
        st.dataframe(rep_df, use_container_width=True)
    else:
        st.info("No representative complaints stored.")

    st.divider()
    st.markdown("### 🧠 Root-Cause Analysis")
    analysis = detail["analysis"]
    if analysis:
        mode_label = "🟢 Live LLM" if analysis["llm_mode"] == "live" else "🟡 Mock/Deterministic (no LLM API key configured)"
        st.caption(f"Analysis mode: {mode_label}")
        if analysis["needs_human_review"]:
            st.warning("⚠️ Human Review Recommended")
        else:
            st.success("✅ Human review not required (confidence above threshold)")
        st.markdown(f"**Likely root cause:** {analysis['root_cause']}")
        st.progress(min(max(analysis["confidence"], 0.0), 1.0), text=f"Confidence: {analysis['confidence']:.0%}")
        if analysis["evidence"]:
            st.markdown("**Evidence:**")
            for e in analysis["evidence"]:
                st.markdown(f"- `{e['document_id']}` — {e['title']}")
        if analysis["recommended_actions"]:
            st.markdown("**Recommended action:**")
            for a in analysis["recommended_actions"]:
                st.markdown(f"- {a}")
    else:
        st.info("No root-cause analysis generated yet for this issue.")


def _render_clusters_tab():
    st.subheader("🧬 Complaint Clusters")
    clusters = da.load_clusters()
    if not clusters:
        st.info("No clusters found. Run scripts/process_complaints.py.")
        return
    cluster_df = pd.DataFrame(clusters)[[
        "cluster_label", "title", "size", "dominant_category", "dominant_subcategory",
        "dominant_sentiment", "avg_severity_score",
    ]]
    st.dataframe(cluster_df, use_container_width=True, height=400)

    selected = st.selectbox("Inspect a cluster", cluster_df["cluster_label"].tolist())
    cluster = next(c for c in clusters if c["cluster_label"] == selected)
    st.markdown(f"### {cluster['title']}")
    st.markdown(f"**Size:** {cluster['size']}  |  **Dominant category:** {cluster['dominant_category']} / "
                f"{cluster['dominant_subcategory']}  |  **Dominant sentiment:** {cluster['dominant_sentiment']}")
    st.markdown("**Keywords:** " + ", ".join(cluster["keywords"]))
    st.markdown("**Representative complaint IDs:** " + ", ".join(cluster["representative_ids"]))


def render():
    st.title("🚨 Emerging Issues & Investigation")
    tab1, tab2, tab3 = st.tabs(["Emerging Issues", "Issue Investigation", "Clusters"])
    with tab1:
        _render_emerging_issues_tab()
    with tab2:
        _render_investigation_tab()
    with tab3:
        _render_clusters_tab()


render()
