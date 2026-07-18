import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection System")
st.markdown("### Check whether a news article is Real or Fake")

# Text input
news = st.text_area("Enter News Text:", height=200)

# Prediction button
if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Convert text to vector
        news_vec = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vec)[0]
        probability = model.predict_proba(news_vec)[0]
        confidence = max(probability) * 100

        # Show result
        if prediction == 1:
            st.success("✅ REAL NEWS")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("❌ FAKE NEWS")
            st.info(f"Confidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.markdown("Developed by Mohammad Faheem | MCA Project")