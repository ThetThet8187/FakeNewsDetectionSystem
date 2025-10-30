import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd

nltk.download('stopwords')  # Uncomment this once if needed

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(content):
    if not isinstance(content, str):
        return ""
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# Sidebar – Project & Team Info
st.sidebar.title("📌 Project Information")

# General Project Info
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

# Team Members with Name & Roll Number
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

# Show tables without index numbers
st.sidebar.subheader("📄 General Info")
st.sidebar.dataframe(project_info, use_container_width=True, hide_index=True)

st.sidebar.subheader("👥 Members")
st.sidebar.dataframe(members_info, use_container_width=True, hide_index=True)




# Custom CSS for main UI
st.markdown(
    """
    <style>
    .title {
        font-size: 3.5rem;
        font-weight: bold;
        color: #4a90e2;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #6c757d;
        margin-top: 0;
        margin-bottom: 30px;
    }
    # .stTextArea>div>div>textarea {
    #     font-size: 1.1rem;
    #     padding: 15px;
    #     border-radius: 8px;
    #     border: 1.5px solid #4a90e2;
    #     resize: vertical;
    # }
    .stButton>button {
        padding: 10px 40px;
        border-radius: 10px;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and instructions
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news statement below to check if it\'s <b>Real</b> or <b>Fake</b>.</p>', unsafe_allow_html=True)

# Load model/vectorizer
model, vectorizer = load_model()

# Input text
user_input = st.text_area("🧾 Enter News Statement:", height=170)

# Prediction logic
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid news statement.")
    else:
        processed = preprocess_text(user_input)
        vector_input = vectorizer.transform([processed])
        prediction = model.predict(vector_input)[0]

        probs = model.predict_proba(vector_input)[0]
        fake_prob = probs[1]
        real_prob = probs[0]

        if prediction == 1:
            st.error(f"🟥 FAKE NEWS  (Confidence: {fake_prob*100:.1f}%)")
            st.progress(int(fake_prob * 100))
        else:
            st.success(f"🟩 REAL NEWS  (Confidence: {real_prob*100:.1f}%)")
            st.progress(int(real_prob * 100))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:0.9rem;'>"
    "Developed with ❤️ using Streamlit"
    "</p>", unsafe_allow_html=True)
