import streamlit as st
import joblib
from preprocessing import clean_text


# =========================================================
# LOAD MODEL
# =========================================================

import joblib
model = joblib.load("models/fake_news_model.pkl")
print(type(model))
print(hasattr(model, "predict_proba"))


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# SESSION STATE
# =========================================================

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #b8b8b8;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 10px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.real-card {
    background: rgba(0, 180, 80, 0.12);
    border: 1px solid rgba(0, 200, 100, 0.4);
}

.fake-card {
    background: rgba(255, 50, 50, 0.12);
    border: 1px solid rgba(255, 70, 70, 0.4);
}

.result-title {
    font-size: 34px;
    font-weight: 800;
}

.confidence {
    font-size: 20px;
    font-weight: 600;
}

.info-card {
    padding: 18px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 10px;
}

.small-text {
    color: #aaaaaa;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Enter News Article</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste a complete news article below and let the AI model analyze it."
)


# =========================================================
# SAMPLE NEWS
# =========================================================

sample_news = """
India and China agreed to continue dialogue aimed at finding
a fair and mutually acceptable settlement to their long-standing
border dispute. Both sides emphasized continued communication
and efforts to maintain peace along the border.
""".strip()


col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Load Sample News", use_container_width=True):
        st.session_state.news_text = sample_news
        st.rerun()

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.news_text = ""
        st.rerun()


# =========================================================
# TEXT AREA
# =========================================================

news = st.text_area(
    "Paste your news article below:",
    value=st.session_state.news_text,
    height=250,
    placeholder="Paste the complete news article here..."
)


# =========================================================
# ARTICLE STATISTICS
# =========================================================

word_count = len(news.split())
character_count = len(news)

col1, col2 = st.columns(2)

with col1:
    st.caption(f"📝 Words: **{word_count}**")

with col2:
    st.caption(f"🔤 Characters: **{character_count}**")


# =========================================================
# CHECK NEWS BUTTON
# =========================================================

check_news = st.button(
    "🔍 Check News",
    use_container_width=True,
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if check_news:

    if not news.strip():

        st.warning("⚠️ Please enter a news article first.")

    elif word_count < 10:

        st.warning(
            "⚠️ Please enter a longer news article "
            "for a better prediction."
        )

    else:

        # Save current article
        st.session_state.news_text = news

        # -------------------------------------------------
        # LOADING
        # -------------------------------------------------

        with st.spinner("🤖 AI is analyzing the news article..."):

            # Preprocess
            cleaned_news = clean_text(news)

            # Prediction
            prediction = model.predict([cleaned_news])[0]

            # Probability
            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    [cleaned_news]
                )[0]

                confidence = max(probabilities) * 100


        st.divider()

        # =================================================
        # RESULT
        # =================================================

        if prediction == "FAKE":

            st.markdown(
                """
                <div class="result-card fake-card">
                    <div class="result-title">
                        🔴 FAKE NEWS
                    </div>
                    <div>
                        The AI model predicts that this article
                        may contain unreliable information.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-card real-card">
                    <div class="result-title">
                        🟢 REAL NEWS
                    </div>
                    <div>
                        The AI model predicts that this article
                        appears consistent with real news.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # CONFIDENCE
        # =================================================

        if confidence is not None:

            st.subheader("📊 Prediction Confidence")

            st.metric(
                "AI Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(int(confidence), 100)
            )


        # =================================================
        # AI ANALYSIS
        # =================================================

        st.divider()

        st.subheader("🤖 AI Analysis")


        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Word Count",
                word_count
            )

        with col2:
            st.metric(
                "Characters",
                character_count
            )

        with col3:

            if confidence is not None:
                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )
            else:
                st.metric(
                    "Confidence",
                    "N/A"
                )


        # =================================================
        # EXPLANATION
        # =================================================

        st.subheader("💡 Analysis Explanation")

        if confidence is not None:

            if confidence >= 90:

                explanation = (
                    "The model has a high confidence in this "
                    "classification based on the text patterns "
                    "identified during prediction."
                )

            elif confidence >= 70:

                explanation = (
                    "The model has moderate-to-high confidence "
                    "in this classification. Some uncertainty "
                    "may still exist."
                )

            else:

                explanation = (
                    "The model has relatively low confidence. "
                    "The article may require verification from "
                    "trusted sources."
                )

        else:

            explanation = (
                "The model does not provide probability scores, "
                "so a confidence value cannot be displayed."
            )


        st.info("💡 " + explanation)


        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            "⚠️ This is an ML-based prediction. "
            "It should not be treated as a definitive "
            "fact-check. Always verify important news "
            "using reliable sources."
        )


        # =================================================
        # SAVE HISTORY
        # =================================================

        result_text = "🔴 FAKE" if prediction == "FAKE" else "🟢 REAL"

        history_item = {
            "result": result_text,
            "confidence": (
                f"{confidence:.1f}%"
                if confidence is not None
                else "N/A"
            ),
            "words": word_count,
            "preview": news[:80].replace("\n", " ")
        }

        st.session_state.history.insert(
            0,
            history_item
        )

        # Keep only last 5 results
        st.session_state.history = (
            st.session_state.history[:5]
        )


# =========================================================
# RECENT CHECKS
# =========================================================

if st.session_state.history:

    st.divider()

    st.subheader("📜 Recent Checks")

    for i, item in enumerate(
        st.session_state.history,
        start=1
    ):

        st.markdown(
            f"""
            <div class="info-card">
                <b>{i}. {item['result']}</b>
                &nbsp;&nbsp; | &nbsp;&nbsp;
                Confidence: <b>{item['confidence']}</b>
                &nbsp;&nbsp; | &nbsp;&nbsp;
                Words: <b>{item['words']}</b>
                <br>
                <span class="small-text">
                    {item['preview']}...
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ABOUT SYSTEM
# =========================================================

st.divider()

st.subheader("ℹ️ About This System")

st.write(
    """
    This AI-powered Fake News Detection System uses
    Natural Language Processing and Machine Learning
    techniques to analyze news articles and classify
    them as potentially real or fake.
    """
)

st.caption(
    "⚠️ Always verify important information using "
    "trusted and reliable news sources."
)