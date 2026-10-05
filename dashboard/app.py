
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runpy
import streamlit as st

st.set_page_config(
    page_title="Complaint Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "📊 Executive Overview": "dashboard/pages/overview.py",
    "🔍 Complaint Explorer": "dashboard/pages/complaints.py",
    "🚨 Emerging Issues & Investigation": "dashboard/pages/issues.py",
    "📚 Knowledge & Evidence": "dashboard/pages/knowledge.py",
}

st.sidebar.title("🧭 Navigation")
st.sidebar.caption("Customer Complaint Intelligence Platform")
choice = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown(
    "**About this project**\n\n"
    "All complaint data is 100% synthetic, generated for demonstration and "
    "ML experimentation purposes only.\n\n"
    "Classification/sentiment/emotion/embeddings run locally. LLM root-cause "
    "synthesis uses a live API if `LLM_API_KEY` is configured in `.env`, "
    "otherwise a clearly-labeled deterministic mock fallback is used."
)

runpy.run_path(PAGES[choice], run_name="__page__")
