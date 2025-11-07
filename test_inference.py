# test_inference.py
import joblib

model = joblib.load("document_classifier.pkl")

samples = [
    "This is a settlement letter stating the insurer approved part of the claim after deductible.",
    "Repair invoice from garage with labour and parts listed, total $350.",
    "Inspection: Found roof damage and missing shingles after storm."
]

for s in samples:
    pred = model.predict([s])[0]
    probs = model.predict_proba([s])[0]
    print("Text:", s)
    print("Predicted:", pred, "Confidence:", round(max(probs), 3))
    print("-" * 60)
