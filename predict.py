import joblib

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Example news
news = "WASHINGTON (Reuters) - U.S. President Donald Trump said on Friday that the administration would review trade policies with several countries and consider new measures to reduce the trade deficit and support domestic manufacturing industries."
# Transform text
news_vec = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vec)[0]

if prediction == 1:
    print("Real News")
else:
    print("Fake News")