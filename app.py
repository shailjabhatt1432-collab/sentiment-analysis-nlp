import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load saved model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis using NLP")
st.write("Classifies text into **Positive**, **Negative**, and **Neutral** sentiments.")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success("😊 Sentiment: Positive")
        elif prediction == "negative":
            st.error("😠 Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")
