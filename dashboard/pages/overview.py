"""Page 1 - Executive Overview: KPIs and top-level charts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import data_access as da


def render():
    st.title("📊 Executive Overview")
    st.caption("All data shown is 100% synthetic demo data generated for this project.")

    stats = da.load_summary_stats()
    if stats["total_complaints"] == 0:
        st.warning("No complaints found in the database yet. Run the pipeline first:\n\n"
                   "`python scripts/generate_data.py && python scripts/train_models.py && "
                   "python scripts/build_index.py && python scripts/process_complaints.py`")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Complaints", f"{stats['total_complaints']:,}")
    c2.metric("Critical", f"{stats['critical_complaints']:,}")
    c3.metric("High Severity", f"{stats['high_severity_complaints']:,}")
    c4.metric("Negative Sentiment", f"{stats['negative_sentiment_complaints']:,}")
    c5.metric("Active Emerging Issues", f"{stats['active_emerging_issues']:,}")
    c6.metric("Avg Severity Score", f"{stats['average_severity_score']:.2f}")

    st.divider()

    dist = da.load_distributions()

    col1, col2 = st.columns(2)
    with col1:
        cat_df = pd.DataFrame(list(dist["category"].items()), columns=["Category", "Count"]).sort_values("Count", ascending=True)
        fig = px.bar(cat_df, x="Count", y="Category", orientation="h", title="Complaints by Category",
                     color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        sev_df = pd.DataFrame(list(dist["severity"].items()), columns=["Severity", "Count"])
        sev_df["Severity"] = pd.Categorical(sev_df["Severity"], categories=sev_order, ordered=True)
        sev_df = sev_df.sort_values("Severity")
        colors = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#FF9800", "CRITICAL": "#F44336"}
        fig = px.bar(sev_df, x="Severity", y="Count", title="Complaints by Severity",
                     color="Severity", color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sent_df = pd.DataFrame(list(dist["sentiment"].items()), columns=["Sentiment", "Count"])
        colors = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}
        fig = px.pie(sent_df, names="Sentiment", values="Count", title="Complaints by Sentiment",
                     color="Sentiment", color_discrete_map=colors, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        src_df = pd.DataFrame(list(dist["source"].items()), columns=["Source", "Count"])
        fig = px.pie(src_df, names="Source", values="Count", title="Complaints by Source", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Complaint Volume Over Time")
    volume = da.load_daily_volume()
    if volume:
        vol_df = pd.DataFrame(volume)
        vol_df["date"] = pd.to_datetime(vol_df["date"])
        fig = px.line(vol_df, x="date", y="count", title="Daily Complaint Volume")
        fig.update_traces(line_color="#2563EB")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note the sharp spikes on 2024-02-14 (UPI payment failure incident) and "
                   "2024-03-10 (OTP verification failure incident) - injected synthetic incidents "
                   "used to demonstrate anomaly/emerging-issue detection.")


render()
