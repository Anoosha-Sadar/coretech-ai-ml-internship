import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Client Feedback Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# NLTK DOWNLOADS
# ==========================================================

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except:
    nltk.download("omw-1.4")

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

.hero {
    text-align:center;
    padding:20px;
    border-radius:20px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color:white;
    margin-bottom:20px;
}

.hero h1{
    font-size:48px;
}

.hero p{
    font-size:18px;
}

.metric-box{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.result-box{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.15);
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
}

.stButton>button:hover{
    background:#1d4ed8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD FILES
# ==========================================================

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================================================
# TEXT CLEANING
# ==========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🤖 AI Client Feedback Sentiment Analyzer</h1>
<p>
Advanced NLP & Machine Learning System for
Customer Feedback Analysis
</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# PROJECT INFO
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Extra Trees"
    )

with col2:
    st.metric(
        "Accuracy",
        "89.71%"
    )

with col3:
    st.metric(
        "CV Score",
        "89.88%"
    )

st.markdown("---")

# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("📝 Enter Customer Feedback")

user_text = st.text_area(
    "",
    height=220,
    placeholder="Type customer feedback here..."
)

# ==========================================================
# ANALYZE BUTTON
# ==========================================================

if st.button("🚀 Analyze Sentiment"):

    if user_text.strip() == "":

        st.warning(
            "Please enter customer feedback."
        )

    else:

        cleaned_text = clean_text(
            user_text
        )

        transformed_text = tfidf.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            transformed_text
        )[0]

        probabilities = model.predict_proba(
            transformed_text
        )[0]

        sentiment = label_encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = np.max(
            probabilities
        ) * 100

        # ==================================================
        # SENTIMENT DISPLAY
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📊 Prediction Result"
        )

        if sentiment.lower() == "positive":

            st.success(
                f"😊 Sentiment: {sentiment}"
            )

            satisfaction = "High"
            risk = "Low"

            recommendation = """
Continue delivering high-quality service.
Maintain customer engagement and loyalty.
"""

        elif sentiment.lower() == "negative":

            st.error(
                f"😠 Sentiment: {sentiment}"
            )

            satisfaction = "Low"
            risk = "High"

            recommendation = """
Immediate attention required.
Investigate customer concerns and resolve issues quickly.
"""

        elif sentiment.lower() == "neutral":

            st.info(
                f"😐 Sentiment: {sentiment}"
            )

            satisfaction = "Moderate"
            risk = "Medium"

            recommendation = """
Monitor customer experience and encourage further engagement.
"""

        else:

            st.warning(
                f"⚠️ Sentiment: {sentiment}"
            )

            satisfaction = "Unknown"
            risk = "Low"

            recommendation = """
Feedback is not directly relevant for sentiment evaluation.
"""

        # ==================================================
        # METRICS
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                "Satisfaction",
                satisfaction
            )

        with col3:
            st.metric(
                "Risk Level",
                risk
            )

        # ==================================================
        # PROBABILITY CHART
        # ==================================================

        st.subheader(
            "📈 Sentiment Confidence Distribution"
        )

        prob_df = pd.DataFrame({

            "Sentiment":
            label_encoder.classes_,

            "Probability":
            probabilities * 100

        })

        st.bar_chart(
            prob_df.set_index(
                "Sentiment"
            )
        )

        # ==================================================
        # BUSINESS INSIGHTS
        # ==================================================

        st.subheader(
            "💡 Business Insights"
        )

        st.info(
            f"""
Predicted Sentiment: {sentiment}

Customer Satisfaction Level: {satisfaction}

Business Risk Level: {risk}

Confidence Score: {confidence:.2f}%

Recommendation:

{recommendation}
"""
        )

        # ==================================================
        # AI SUMMARY
        # ==================================================

        st.subheader(
            "🤖 AI Generated Summary"
        )

        st.write(
            f"""
The machine learning model analyzed the submitted customer feedback and classified it as **{sentiment}**.

The prediction confidence is **{confidence:.2f}%**, indicating a reliable sentiment assessment.

This feedback suggests a **{satisfaction.lower()} customer satisfaction level** with a **{risk.lower()} business risk profile**.
"""
        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
### 📌 Project Highlights

✅ Natural Language Processing (NLP)

✅ Text Cleaning & Lemmatization

✅ TF-IDF Feature Engineering

✅ Extra Trees Classifier

✅ 89.71% Test Accuracy

✅ 89.88% Cross Validation Score

✅ Real-Time Sentiment Prediction

✅ Business Recommendation System
""")

st.markdown("---")

st.markdown(
"""
<center>

Developed for CoreTech AI/ML Internship Project

AI Client Feedback Sentiment Analyzer

Intern: Anoosha Sadar

</center>
""",
unsafe_allow_html=True
)