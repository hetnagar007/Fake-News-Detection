
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import load_data
import os

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_and_save():
    df = load_data()
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODELS_DIR / "fake_news_pipeline.pkl")
    print("Saved model to", MODELS_DIR / "fake_news_pipeline.pkl")

if __name__ == "__main__":
    train_and_save()
