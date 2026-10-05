"""Page 4 - Knowledge/Evidence: browse the knowledge base and test RAG
retrieval interactively."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components import data_access as da
from src.rag.pipeline import retrieve_evidence


def render():
    st.title("📚 Knowledge Base & RAG Evidence")
    st.caption("All documents below are synthetic, fictional demo content created for this project "
               "(not real internal company documents).")

    tab1, tab2 = st.tabs(["Browse Documents", "Test Retrieval"])

    with tab1:
        docs = da.load_knowledge_documents()
        if not docs:
            st.warning("No knowledge documents found. Run scripts/build_index.py.")
            return

        doc_types = sorted(set(d["doc_type"] for d in docs))
        selected_type = st.selectbox("Filter by document type", ["All"] + doc_types)
        filtered = docs if selected_type == "All" else [d for d in docs if d["doc_type"] == selected_type]

        for d in filtered:
            with st.expander(f"[{d['document_id']}] {d['title']} ({d['doc_type']})"):
                st.caption(f"Category: {d['category'] or 'N/A'}  |  Tags: {d['tags'] or 'N/A'}")
                st.markdown(d["content"])

    with tab2:
        st.markdown("Enter a query to see which knowledge-base chunks the RAG retriever would "
                     "surface as evidence for root-cause analysis.")
        query = st.text_input("Query", "My OTP never arrives when trying to login")
        top_k = st.slider("Top K", 1, 10, 4)
        if st.button("Retrieve"):
            results = retrieve_evidence(query, top_k=top_k)
            if not results:
                st.info("No relevant documents retrieved above the similarity threshold.")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**[{r['document_id']}] {r['title']}**  (similarity: {r['similarity']:.3f})")
                    st.caption(f"Type: {r.get('doc_type')}  |  Category: {r.get('category')}")
                    st.text(r["chunk_text"][:500])


render()
