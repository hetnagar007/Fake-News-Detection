
import pandas as pd
import re
from pathlib import Path
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

def clean_text(s):
    s = str(s)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    stop = set(stopwords.words("english"))
    tokens = [w for w in s.split() if w not in stop]
    return " ".join(tokens)

def load_data(path=None):
    base = Path(__file__).resolve().parent.parent
    if path is None:
        path = base / "data" / "news.csv"
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    df = df.dropna(subset=["text","label"])
    df["text"] = df["text"].apply(clean_text)
    return df
