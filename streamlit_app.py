# frontend/streamlit_app.py
import streamlit as st
import requests


st.set_page_config(page_title="Insurance Doc Assistant")
st.title("Insurance Document Assistant")
st.write("Upload an insurance document (PDF). The system will classify it and — if it's a claim/settlement — provide a plain-language explanation.")


api_base = st.text_input("Backend API URL", value="https://your-backend-url.onrender.com")


uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded is not None:
files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
with st.spinner("Analyzing..."):
try:
res = requests.post(f"{api_base}/analyze", files=files, timeout=60)
data = res.json()
except Exception as e:
st.error(f"API call failed: {e}")
data = None


if data:
st.success(f"Document type: {data.get('doc_type')} (confidence: {data.get('confidence'):.2f})")
if data.get('explanation'):
st.subheader("Explanation")
st.write(data.get('explanation'))
st.subheader("Preview (first 2000 chars)")
st.text(data.get('preview', '')[:2000])
