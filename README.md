
# DocClassifier + Claims Explainer — Quickstart


## 1) Prepare model
- Use `train/train.py` to prepare `document_classifier.pkl`. Upload that file to the repository root.


## 2) Set LLM key (optional)
- To get high-quality explanations, set `OPENAI_API_KEY` as an environment variable in your host (Render/Deta/etc.). The code uses OpenAI if the key exists, otherwise a fallback summary.


## 3) Deploy backend
- Option A (Render): connect your GitHub repo and set the `Start Command` to:
`uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
- Option B (Docker): build and run the Dockerfile.


## 4) Deploy frontend
- Option A (Streamlit Cloud or Hugging Face Space): deploy the `frontend/streamlit_app.py` and point the API URL to your backend.
- Option B: Serve static frontend and have it call the backend URL.


## 5) Test
- Visit `https://your-backend-url/docs` for interactive docs.
- Use the Streamlit UI to upload PDFs and view classification + explanations.
