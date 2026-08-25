import streamlit as st
import joblib
from preprocessing import clean_text

# Load model
model = joblib.load("models/fake_news_model.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.divider()

# Input
st.subheader("🔎 Enter News Article")

news = st.text_area(
    "Paste your news article below:",
    height=250,
    placeholder="Paste the complete news article here..."
)

# Button
if st.button("🔍 Check News", use_container_width=True):

    if not news.strip():
        st.warning("⚠️ Please enter a news article first.")

    else:
        # Preprocess
        cleaned_news = clean_text(news)

        # Prediction
        prediction = model.predict([cleaned_news])[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned_news])[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        st.divider()

        # Result
        if prediction == "FAKE":
            st.error("🔴 FAKE NEWS")
        else:
            st.success("🟢 REAL NEWS")

        # Confidence
        if confidence is not None:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

st.divider()

st.caption("⚠️ This system is an ML-based prediction and should not be treated as a definitive fact-check.")