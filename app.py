import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import gdown

# --------------------------------------------------
# NLTK (download locally ONCE, NOT on Streamlit Cloud)
# --------------------------------------------------
# nltk.download('stopwords')  # ❌ keep commented for deployment

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --------------------------------------------------
# Google Drive model download (handles >25MB files)
# --------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1rLgD7JgqnrMhS3K8TlInqH2R8Czf42O0"
VECT_URL  = "https://drive.google.com/uc?id=1mE2KdwOt2uo4CFQ51g4Wok6sfSCRi0Qu"

def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"⬇️ Downloading {filename}..."):
            gdown.download(url, filename, quiet=False)

download_file(MODEL_URL, "fake_news_model.pkl")
download_file(VECT_URL, "tfidf_vectorizer.pkl")

# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------
def preprocess_text(content):
    if not isinstance(content, str):
        return ""
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# --------------------------------------------------
# Load model & vectorizer (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# --------------------------------------------------
# Sidebar – Project Info
# --------------------------------------------------
st.sidebar.title("📌 Project Information")

project_info = pd.DataFrame({
    "Field": ["Project", "Group", "University", "Supervisor", "Year"],
    "Details": [
        "Fake News Detection System",
        "Fifth Year Group III",
        "West Yangon Technological University",
        "Dr. Hsu Wai Hnin (Associate Professor)",
        "2025"
    ]
})

members_info = pd.DataFrame({
    "Name": [
        "Ma Thet Thet Tun",
        "Ma Thida Myat Noe Zaw",
        "Ma May Myo Pwint",
        "Ma Phyu Zin Win Htein"
    ],
    "Roll Number": [
        "VIT-4 (Leader)",
        "VIT-5",
        "VIT-Ext-11",
        "VIT-Ext-12"
    ]
})

st.sidebar.subheader("📄 General Info")
st.sidebar.dataframe(project_info, use_container_width=True, hide_index=True)

st.sidebar.subheader("👥 Members")
st.sidebar.dataframe(members_info, use_container_width=True, hide_index=True)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 3.5rem;
        font-weight: bold;
        color: #4a90e2;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .stButton>button {
        padding: 10px 40px;
        border-radius: 10px;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter a news statement below to check if it is <b>Real</b> or <b>Fake</b>.</p>',
    unsafe_allow_html=True
)

model, vectorizer = load_model()

user_input = st.text_area("🧾 Enter News Statement:", height=170)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid news statement.")
    else:
        processed = preprocess_text(user_input)
        vector_input = vectorizer.transform([processed])
        prediction = model.predict(vector_input)[0]

        probs = model.predict_proba(vector_input)[0]
        real_prob = probs[0]
        fake_prob = probs[1]

        if prediction == 1:
            st.error(f"🟥 FAKE NEWS  (Confidence: {fake_prob*100:.1f}%)")
            st.progress(int(fake_prob * 100))
        else:
            st.success(f"🟩 REAL NEWS  (Confidence: {real_prob*100:.1f}%)")
            st.progress(int(real_prob * 100))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d;'>Developed with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
