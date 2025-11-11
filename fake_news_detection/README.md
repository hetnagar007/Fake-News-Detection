
# Fake News Detection — Modern Streamlit GUI (Package)
## Quick start
1. Install dependencies:
   pip install -r requirements.txt
2. Train model (creates models/fake_news_pipeline.pkl):
   python src/model_training.py
3. Run Streamlit app:
   streamlit run app.py
4. Test CLI predictor:
   python src/predict.py "Your headline here"
