"""Demo interface: paste or upload a document, see category, confidence and route (FR20).

Run:  streamlit run app/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Router", layout="centered")
st.title("Document Classification & Routing")

uploaded = st.file_uploader("Upload a .txt document", type=["txt"])
text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""
text = st.text_area("...or paste the document text", value=text, height=220)

if st.button("Classify", type="primary", disabled=not text.strip()):
    try:
        resp = requests.post(f"{API_URL}/classify", json={"text": text}, timeout=30)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        st.stop()
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.json().get('detail')}")
        st.stop()
    r = resp.json()
    col1, col2 = st.columns(2)
    col1.metric("Category", r["category"], f"{r['confidence']:.0%} confidence")
    col2.metric("Route to", r["department"])
    if r["needs_review"]:
        st.warning("Low confidence: sent to manual review.")
    top = pd.DataFrame(r["top_k"]).set_index("label")
    st.bar_chart(top["probability"])
    st.caption(f"Model: {r['model_version']}")
