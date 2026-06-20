import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
import plotly.graph_objects as go
import plotly.express as px
import base64

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="SentimentAI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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

#MainMenu {
visibility:hidden;
}

footer {
visibility:hidden;
}

header {
visibility:hidden;
}

.main {

background:
linear-gradient(
135deg,
#020617,
#0f172a,
#111827
);

color:white;
}

.block-container{
padding-top:1rem;
}

.metric-card{

background:
rgba(
255,
255,
255,
0.08
);

backdrop-filter:
blur(15px);

padding:20px;

border-radius:20px;

border:
1px solid rgba(
255,
255,
255,
0.15
);
}

.hero{

background:
linear-gradient(
135deg,
#06b6d4,
#3b82f6,
#8b5cf6
);

padding:50px;

border-radius:30px;

text-align:center;

color:white;

box-shadow:
0px 15px 40px rgba(
0,
0,
0,
0.35
);
}

.glass{

background:
rgba(
255,
255,
255,
0.08
);

backdrop-filter:
blur(15px);

padding:20px;

border-radius:20px;

border:
1px solid rgba(
255,
255,
255,
0.15
);
}

.stButton > button{

width:100%;

height:60px;

font-size:20px;

font-weight:bold;

border-radius:15px;

background:
linear-gradient(
135deg,
#06b6d4,
#3b82f6
);

color:white;

border:none;
}

.stButton > button:hover{

background:
linear-gradient(
135deg,
#0891b2,
#2563eb
);

color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL FILES
# ==========================================================

model = joblib.load(
    "sentiment_model.pkl"
)

tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)

# ==========================================================
# TEXT CLEANING
# ==========================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"www\S+",
        "",
        text
    )

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
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120
    )

    st.title("SentimentAI")

    st.markdown("""
    ### Neural Intelligence Core

    🟢 NLP Engine Online

    🟢 Prediction Engine Active

    🟢 Analytics Ready

    🟢 Business Insights Enabled
    """)

    st.markdown("---")

    st.markdown("""
    ### Project Information

    **Model:** Extra Trees Classifier

    **Accuracy:** 89.71%

    **Cross Validation:** 89.88%

    **Classes:** 4

    **Technology:** NLP + Machine Learning
    """)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
"""
<div style='
background:linear-gradient(135deg,#0f172a,#1e293b,#0f766e);
padding:60px;
border-radius:30px;
text-align:center;
margin-bottom:30px;
box-shadow:0px 15px 40px rgba(0,0,0,0.4);
'>

<h1 style='font-size:70px;color:white;margin-bottom:10px;'>
🧠 SentimentAI Pro
</h1>

<h3 style='color:#67e8f9;'>
NEXT GENERATION CUSTOMER INTELLIGENCE
</h3>

<p style='
font-size:20px;
color:#e2e8f0;
max-width:1000px;
margin:auto;
'>
Transform customer feedback into actionable business intelligence using Artificial Intelligence,
Natural Language Processing, Machine Learning and Sentiment Analytics.
</p>

<br>

<div style='display:flex;justify-content:center;gap:15px;flex-wrap:wrap;'>

<div style='background:rgba(255,255,255,0.1);padding:12px 20px;border-radius:20px;color:white;'>
🤖 AI Powered
</div>

<div style='background:rgba(255,255,255,0.1);padding:12px 20px;border-radius:20px;color:white;'>
📊 Real-Time Analytics
</div>

<div style='background:rgba(255,255,255,0.1);padding:12px 20px;border-radius:20px;color:white;'>
⚡ NLP Engine
</div>

<div style='background:rgba(255,255,255,0.1);padding:12px 20px;border-radius:20px;color:white;'>
🎯 89.88% Accuracy
</div>

</div>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# INPUT AREA
# ==========================================================

st.subheader("💬 Customer Feedback")

user_text = st.text_area(
    "",
    height=250,
    placeholder="Enter customer feedback here..."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Characters",
        len(user_text)
    )

with col2:
    st.metric(
        "Words",
        len(user_text.split())
    )

with col3:
    st.metric(
        "Status",
        "Ready"
    )

st.write("")

analyze = st.button(
    "🚀 Analyze Feedback"
)

# ==========================================================
# AI ANALYSIS ENGINE
# ==========================================================

if analyze:

    if user_text.strip() == "":

        st.warning(
            "Please enter customer feedback."
        )

    else:

        with st.spinner(
        "🧠 AI Neural Engine Processing..."
        ):
            import time
            time.sleep(1.5)
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

            confidence = (
                np.max(probabilities)
                * 100
            )

        # ==================================================
        # SENTIMENT CARD
        # ==================================================

        st.markdown("---")

        if sentiment == "Positive":

            st.success(
                f"😊 POSITIVE SENTIMENT DETECTED | Confidence: {confidence:.2f}%"
            )

            satisfaction = "Excellent"

            risk = "Low"

            recommendation = """
            Maintain customer engagement,
            reward loyalty,
            and encourage referrals.
            """

        elif sentiment == "Negative":

            st.error(
                f"😠 NEGATIVE SENTIMENT DETECTED | Confidence: {confidence:.2f}%"
            )

            satisfaction = "Poor"

            risk = "High"

            recommendation = """
            Immediate action required.
            Resolve customer concerns quickly.
            """

        elif sentiment == "Neutral":

            st.info(
                f"😐 NEUTRAL SENTIMENT DETECTED | Confidence: {confidence:.2f}%"
            )

            satisfaction = "Moderate"

            risk = "Medium"

            recommendation = """
            Improve engagement and
            gather additional feedback.
            """

        else:

            st.warning(
                f"⚠️ IRRELEVANT FEEDBACK DETECTED | Confidence: {confidence:.2f}%"
            )

            satisfaction = "Unknown"

            risk = "Low"

            recommendation = """
            Feedback does not clearly
            relate to customer satisfaction.
            """

        # ==================================================
        # EXECUTIVE METRICS
        # ==================================================

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            if confidence >= 90:
                st.success("🟢 Very High Confidence Prediction")

            elif confidence >= 75:
                st.info("🔵 High Confidence Prediction")

            elif confidence >= 50:
                st.warning("🟡 Moderate Confidence Prediction")

            else:
                st.error("🔴 Low Confidence Prediction")

        with m2:

            st.metric(
                "Satisfaction",
                satisfaction
            )

        with m3:

            st.metric(
                "Business Risk",
                risk
            )

        st.markdown("---")

        # ==================================================
        # PROBABILITY CHART
        # ==================================================

        st.subheader(
            "📊 Sentiment Probability Distribution"
        )

        prob_df = pd.DataFrame({

            "Sentiment":
            label_encoder.classes_,

            "Probability":
            probabilities * 100

        })

        fig = px.pie(
            prob_df,
            names="Sentiment",
            values="Probability",
            hole=0.55,
            title="Prediction Confidence"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================================
        # BUSINESS INSIGHTS
        # ==================================================

        st.subheader(
            "💡 AI Business Insights"
        )

        st.info(
            f"""
            Predicted Sentiment: {sentiment}

            Customer Satisfaction: {satisfaction}

            Business Risk Level: {risk}

            Model Confidence: {confidence:.2f}%

            Recommendation:

            {recommendation}
            """
        )


        sentiment_score = {
            "Positive": 100,
            "Neutral": 60,
            "Negative": 20,
            "Irrelevant": 40
        }[sentiment]

        st.subheader("🌡 Sentiment Meter")

        st.progress(sentiment_score / 100)

        st.write(
            f"Sentiment Strength: {sentiment_score}%"
        )






        # ==================================================
        # AI EXECUTIVE SUMMARY
        # ==================================================

        st.subheader(
            "🤖 Executive Summary"
        )

        st.write(
            f"""
The AI engine analyzed the submitted customer feedback
and classified it as **{sentiment}**.

The prediction confidence is
**{confidence:.2f}%**.

The feedback indicates a
**{satisfaction.lower()} customer experience level**
with a
**{risk.lower()} business risk profile**.

Management should consider the
recommended actions to improve
customer satisfaction and retention.
"""
        )


# ==========================================================
# AI CONFIDENCE GAUGE
# ==========================================================

        st.markdown("---")

        st.subheader(
    "🎯 AI Confidence Gauge"
        )

        gauge = go.Figure(
            go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={
            "text":"Prediction Confidence"
        },
        gauge={
            "axis":{
                "range":[0,100]
            },
            "bar":{
                "color":"cyan"
            },
                "steps":[

                {
                    "range":[0,50],
                    "color":"darkred"
                },

                {
                    "range":[50,75],
                    "color":"orange"
                },

                {
                    "range":[75,100],
                    "color":"green"
                }

                ]
            }
             )
        )

    st.plotly_chart(
    gauge,
    use_container_width=True
    )

# ==========================================================
# BUSINESS HEALTH SCORE
# ==========================================================

    st.markdown("---")

    st.subheader(
    "🏥 Customer Health Analysis"
    )

    if sentiment == "Positive":

        health_score = 92

    elif sentiment == "Neutral":

        health_score = 65

    elif sentiment == "Negative":

        health_score = 25

    else:

        health_score = 50

    st.progress(
        health_score / 100
    )

    st.metric(
    "Customer Health Score",
        f"{health_score}%"
    )

# ==========================================================
# RISK METER
# ==========================================================

    st.subheader(
    "⚠ Business Risk Meter"
    )

    if risk == "Low":

        risk_score = 15

    elif risk == "Medium":

        risk_score = 55

    else:

        risk_score = 90

    risk_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={
            "text":"Risk Level"
        },
        gauge={
            "axis":{
                "range":[0,100]
            },
            "bar":{
                "color":"red"
            }
        }
    )
    )

    st.plotly_chart(
    risk_fig,
    use_container_width=True
    )

# ==========================================================
# FEEDBACK ANALYTICS
# ==========================================================

    st.markdown("---")

    st.subheader(
    "📈 Feedback Analytics"
)

    analytics1, analytics2 = st.columns(2)

    with analytics1:

        st.metric(
        "Characters",
        len(user_text)
        )

        st.metric(
        "Words",
        len(user_text.split())
        )

    with analytics2:

        avg_word_length = round(
        np.mean(
            [
                len(word)
                for word in user_text.split()
            ]
        ),
        2
    ) if len(user_text.split()) > 0 else 0

        st.metric(
        "Average Word Length",
        avg_word_length
    )

        st.metric(
        "Prediction",
        sentiment
    )

# ==========================================================
# EXECUTIVE DECISION PANEL
# ==========================================================

    st.markdown("---")

    st.subheader(
    "🧠 Executive Decision Panel"
)

    if sentiment == "Positive":

            st.success("""
    Customer is satisfied.

    Recommended Action:

    • Offer loyalty rewards

    • Request public review

    • Upsell premium services

    • Maintain engagement
    """)

    elif sentiment == "Negative":

        st.error("""
    Customer dissatisfaction detected.

    Recommended Action:

    • Immediate follow-up

    • Escalate complaint

    • Assign support team

    • Recover customer trust
    """)

    elif sentiment == "Neutral":

        st.info("""
    Customer sentiment is neutral.

    Recommended Action:

    • Increase engagement

    • Gather more feedback

    • Offer personalized experience
    """)

    else:

        st.warning("""
    Feedback classified as irrelevant.

    Recommended Action:

    • No immediate business action required.
    """)
        


# ==========================================================
# KEYWORD ANALYSIS
# ==========================================================

    st.markdown("---")

    st.subheader("🔍 Keyword Intelligence")

    words = cleaned_text.split()

    if len(words) > 0:

        word_freq = pd.Series(words).value_counts().head(10)

        keyword_df = pd.DataFrame({
        "Keyword": word_freq.index,
        "Frequency": word_freq.values
    })

        keyword_fig = px.bar(
        keyword_df,
        x="Frequency",
        y="Keyword",
        orientation="h",
        title="Top Keywords"
    )

        st.plotly_chart(
        keyword_fig,
        use_container_width=True
    )

    st.subheader("☁ Feedback Keywords")

    if len(words) > 0:

            keywords = word_freq.index.tolist()

            keyword_html = ""

            for word in keywords:

                keyword_html += f"""
        <span style='
        background:#0ea5e9;
        padding:10px 18px;
        margin:6px;
        border-radius:20px;
        display:inline-block;
        color:white;
        '>
        {word}
        </span>
        """

    st.markdown(
        keyword_html,
        unsafe_allow_html=True
        )





# ==========================================================
# FEEDBACK QUALITY ANALYSIS
# ==========================================================

    st.markdown("---")

    st.subheader("📋 Feedback Quality Assessment")

    feedback_score = min(
    100,
    len(words) * 2
)

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.metric(
        "Feedback Quality Score",
        f"{feedback_score}%"
    )

    with quality_col2:

        if feedback_score >= 80:

            st.success(
            "High Quality Feedback"
        )

        elif feedback_score >= 50:

            st.info(
            "Moderate Quality Feedback"
        )

        else:

            st.warning(
            "Limited Feedback Content"
        )

# ==========================================================
# AI SENTIMENT PROFILE
# ==========================================================

    st.markdown("---")

    st.subheader("🧠 AI Sentiment Profile")

    profile_data = pd.DataFrame({

    "Metric": [
        "Customer Satisfaction",
        "Brand Trust",
        "Engagement",
        "Retention Potential",
        "Business Value"
    ],

    "Score": [

        health_score,

        min(
            100,
            health_score + 5
        ),

        max(
            20,
            health_score - 10
        ),

        max(
            25,
            health_score - 5
        ),

        min(
            100,
            health_score + 8
        )
    ]
})

    radar = go.Figure()

    radar.add_trace(
    go.Scatterpolar(
        r=profile_data["Score"],
        theta=profile_data["Metric"],
        fill="toself",
        name="Customer Profile"
    )
)

    radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,100]
        )
    ),
    showlegend=False
)

    st.plotly_chart(
    radar,
    use_container_width=True
)

# ==========================================================
# EXECUTIVE ACTION PLAN
# ==========================================================

    st.markdown("---")

    st.subheader("🚀 Executive Action Plan")

    if sentiment == "Positive":

        st.success("""
### Recommended Growth Strategy

✅ Request customer review

✅ Promote loyalty programs

✅ Upsell premium services

✅ Encourage referrals

✅ Maintain customer engagement
""")

    elif sentiment == "Negative":

        st.error("""
### Recommended Recovery Strategy

🚨 Contact customer immediately

🚨 Escalate issue to support team

🚨 Provide compensation if needed

🚨 Monitor future interactions

🚨 Rebuild customer trust
""")

    elif sentiment == "Neutral":

        st.info("""
### Recommended Engagement Strategy

📌 Gather additional feedback

📌 Personalize customer experience

📌 Increase interaction

📌 Offer targeted promotions
""")

    else:

        st.warning("""
### Recommended Monitoring Strategy

⚠ Feedback appears unrelated

⚠ No immediate action required

⚠ Continue collecting feedback
""")

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

    st.markdown("---")

    st.subheader("📄 Executive Report")

    report = f"""
SENTIMENT ANALYSIS REPORT

Predicted Sentiment: {sentiment}

Confidence Score: {confidence:.2f}%

Customer Satisfaction: {satisfaction}

Business Risk: {risk}

Health Score: {health_score}%

Feedback:

{user_text}

Recommendation:

{recommendation}
    """

    st.download_button(
    label="⬇ Download Executive Report",
    data=report,
    file_name="sentiment_report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

    st.markdown("---")

    st.markdown("""
### 🌐 Platform Capabilities

✅ NLP Text Processing

✅ Sentiment Classification

✅ Confidence Analysis

✅ Customer Health Scoring

✅ Business Risk Assessment

✅ Keyword Intelligence

✅ Executive Recommendations

✅ Analytics Dashboard

✅ Report Generation

Built using Artificial Intelligence, Natural Language Processing, and Machine Learning.
    """)


st.markdown("---")

c1, c2 = st.columns([1,3])

with c1:

    st.markdown("""
<div style="text-align:center;">

<img src="data:image/jpeg;base64,{}"
style="
width:220px;
height:220px;
border-radius:50%;
object-fit:cover;
border:5px solid #06b6d4;
box-shadow:0px 0px 25px rgba(6,182,212,0.7);
">

</div>
""".format(
    base64.b64encode(
        open("IMG_20260619_211804.jpg.jpeg", "rb").read()
    ).decode()
),
unsafe_allow_html=True)

with c2:

    st.markdown("""
# 💻 Meet The Developer

### Anoosha Sadar

Artificial Intelligence & ML Engineer

Developer of SentimentAI Pro — an enterprise-grade customer feedback intelligence platform designed to transform customer opinions into actionable business insights using Machine Learning and Natural Language Processing.

### Expertise

✅ Machine Learning

✅ Natural Language Processing

✅ Sentiment Analysis

✅ Data Analytics

✅ Business Intelligence
""")
    






st.markdown("---")

st.markdown("""



            



<div style="
text-align:center;
">

<h4 style="color:#38bdf8;">

Built with Python • Streamlit • NLP • Machine Learning

</h4>

<p style="color:#64748b;">

© 2026 SentimentAI Pro

Enterprise Customer Feedback Intelligence Platform

All Rights Reserved.

</p>

</div>

</div>
""",
unsafe_allow_html=True)