"""Minimal chat UI with upload + citations (Streamlit placeholder)"""
## 💬 `frontend/streamlit_app.py`

import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 RAG Assistant — Industry-Grade Minimal UI")

with st.sidebar:
    st.header("Ingest Documents")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf and st.button("Ingest PDF"):
        files = {"file": (pdf.name, pdf.getvalue(), pdf.type)}
        r = requests.post(f"{API_URL}/ingest/pdf", files=files, timeout=120)
        st.success(r.json())

    txt = st.file_uploader("Upload Text/MD", type=["txt", "md"])
    if txt and st.button("Ingest Text"):
        files = {"file": (txt.name, txt.getvalue(), txt.type)}
        r = requests.post(f"{API_URL}/ingest/text", files=files, timeout=120)
        st.success(r.json())

st.divider()
query = st.text_input("Ask a question about your documents:")
if st.button("Ask") and query:
    payload = {"query": query}
    r = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
    if r.status_code == 200:
        data = r.json()
        st.subheader("Answer")
        st.write(data["answer"])
        st.subheader("Sources")
        for s in data.get("sources", []):
            with st.expander(s.get("source")) or "(no source)":
                st.write(f"Page: {s.get('page')}")
                st.write(s.get("snippet"))
    else:
        st.error(r.text)

