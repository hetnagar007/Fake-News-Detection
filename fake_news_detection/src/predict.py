
import sys
from pathlib import Path
import joblib
import re

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "fake_news_pipeline.pkl"

def clean_text(s):
    s = str(s)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def predict_text(text):
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    proba = None
    try:
        proba = model.predict_proba([cleaned])[0].max()
    except:
        proba = None
    label = "Real" if int(pred)==1 else "Fake"
    return label, proba

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a headline to classify.")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    label, proba = predict_text(text)
    if proba is not None:
        print(f"Prediction: {label} (confidence: {proba:.2f})")
    else:
        print(f"Prediction: {label}")
