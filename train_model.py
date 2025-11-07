# train_model.py
"""
Quick training script to create a simple document_classifier.pkl
Uses TF-IDF + LogisticRegression on tiny sample texts for demo/testing.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# --- 1) Example training data (small, for demo) ---
texts = [
    # Claim forms / settlement letters
    "Claim form: insured vehicle collided with a tree. Policy number included.",
    "We regret to inform you that your claim was partially approved due to deductible.",
    "Settlement letter: approved amount is $1,200 after deductible and depreciation.",
    # Invoices
    "Invoice: Parts replaced - bumper, headlight. Total due $450.00. Supplier invoice.",
    "Mechanic invoice for car repair, labour 3 hours, parts and tax included.",
    # Inspection reports
    "Inspection report: roof damage from hail. Photos attached, measurements noted.",
    "Property inspection: observed water infiltration in basement and mold.",
    # Misc / other
    "Customer inquiry about policy coverage and premium payment options.",
    "Third party report: damage assessment and estimated repair costs."
]

labels = [
    "Claim Settlement",   # for first
    "Claim Settlement",
    "Claim Settlement",
    "Invoice",
    "Invoice",
    "Inspection Report",
    "Inspection Report",
    "Other",
    "Other"
]

# --- 2) Train model pipeline ---
print("Training model on sample data...")
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_features=5000),
    LogisticRegression(max_iter=1000)
)
model.fit(texts, labels)

# --- 3) Save model ---
out_path = "document_classifier.pkl"
joblib.dump(model, out_path)
print(f"Saved trained model to: {out_path}")
