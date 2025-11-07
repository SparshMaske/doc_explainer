# backend/server.py


app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"],
)


# Load classifier if present
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "document_classifier.pkl")
if os.path.exists(MODEL_PATH):
classifier = joblib.load(MODEL_PATH)
else:
classifier = None


@app.get("/")
async def root():
return {"message": "Welcome to the Document Classifier + Explainer API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
# save to temp
temp_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(temp_dir, exist_ok=True)
file_path = os.path.join(temp_dir, file.filename)
contents = await file.read()
with open(file_path, "wb") as f:
f.write(contents)


# extract text
text = ""
try:
text = extract_text_from_pdf(file_path)
except Exception:
try:
text = ocr_image_bytes(contents)
except Exception:
text = ""


if not text or classifier is None:
return JSONResponse({"error": "No text extracted or classifier not available"}, status_code=400)


# classify
pred = classifier.predict([text])[0]
probs = classifier.predict_proba([text])[0]
conf = float(max(probs))


result = {
"filename": file.filename,
"doc_type": pred,
"confidence": conf,
}


# If claim-like document, generate explanation
if any(k in pred.lower() for k in ["claim", "settlement", "explanation", "report"]):
explanation = explain_text(text)
result["explanation"] = explanation


# optionally return a short extract
result["preview"] = text[:5000]
return result


if __name__ == "__main__":
uvicorn.run("backend.server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
