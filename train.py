# train/train.py
import os
import joblib
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# Example dataset list: replace with real paths & labels
DATA = [
("/path/to/data/claim1.pdf", "Claim"),
("/path/to/data/inspection1.pdf", "Inspection Report"),
("/path/to/data/invoice1.pdf", "Invoice"),
]


texts, labels = [], []
for fpath, lbl in DATA:
if not os.path.exists(fpath):
print("missing", fpath)
continue
with pdfplumber.open(fpath) as pdf:
text = "".join([p.extract_text() or "" for p in pdf.pages])
if not text.strip():
print("No text in", fpath)
continue
texts.append(text)
labels.append(lbl)


clf = make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1,2)), LogisticRegression(max_iter=1000))
clf.fit(texts, labels)
joblib.dump(clf, os.path.join(os.path.dirname(__file__), "..", "document_classifier.pkl"))
print("Saved model to document_classifier.pkl")
