
import streamlit as st
import joblib, re
from pathlib import Path

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "fake_news_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def clean_text(s):
    s = str(s)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

try:
    model = load_model()
except Exception as e:
    st.warning("Model not found or failed to load. Please run 'python src/model_training.py' to train the model.")
    st.stop()

st.title("📰 Fake News Detection — Modern UI")
st.markdown("Detect whether a news headline or short article is likely **REAL** or **FAKE**.")

col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_area("Enter a news headline or short article:", height=200, placeholder="Type or paste text here...")
    if st.button("Use Example (Real)"):
        user_input = "Government announces new healthcare initiative to support citizens"
        st.rerun()
    if st.button("Use Example (Fake)"):
        user_input = "Secret alien base discovered under city mall, sources claim"
        st.rerun()
with col2:
    st.markdown("### How it works")
    st.markdown("• TF-IDF vectorization\n• Logistic Regression classifier\n• Trained on headline-style data")

st.markdown("---")

if st.button("Detect"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            try:
                pred = model.predict([cleaned])[0]
                try:
                    proba = model.predict_proba([cleaned])[0].max()
                except:
                    proba = None
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                st.stop()

            if int(pred) == 1:
                st.success("✅ Likely REAL NEWS")
                st.metric("Confidence", f"{(proba*100):.1f}%" if proba is not None else "N/A")
            else:
                st.error("🚨 Likely FAKE NEWS")
                st.metric("Confidence", f"{(proba*100):.1f}%" if proba is not None else "N/A")

st.markdown('---')
st.caption("Built with TF-IDF, Logistic Regression, and Streamlit.")
