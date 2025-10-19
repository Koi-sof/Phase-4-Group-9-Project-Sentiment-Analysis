import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# Loading the model

# Define the sentiment mapping based on the notebook's categories (0=Negative, 1=Neutral, 2=Positive)
sentiment_mapping = {
    0: "Negative ",
    1: "Neutral ",
    2: "Positive "
}

MODEL_FILE = 'multi_nlp_model.sav'

# Load the model (the trained Pipeline object)
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure your trained model is saved under this name in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Text preprocessing setup
# Download necessary NLTK data (Checking if data is already available to avoid unnecessary downloads)
try:
    WordNetLemmatizer()
    set(stopwords.words('english'))
except LookupError:
    # Only download if not found (a common issue in deployment environments)
    nltk.download('wordnet')
    nltk.download('stopwords')

wordLemm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(tweet):
    """
    Applies the same cleaning steps as used during model training in the notebook.
    """
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = str(tweet).lower()
    # Remove URLs and user mentions
    tweet = re.sub(urlPattern, ' ', tweet)
    tweet = re.sub(userPattern, ' ', tweet)
    # Remove non-alphabetic characters
    tweet = re.sub(alphaPattern, " ", tweet)
    # Replace repeated sequences of characters (e.g., 'coooool' -> 'cool')
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    words = []
    for word in tweet.split():
        # Remove stop words and single-character words
        if word not in stop_words and len(word) > 1:
            word = wordLemm.lemmatize(word)
            words.append(word)
    return " ".join(words)


# Streamlit App UI

st.set_page_config(page_title="Tweet Sentiment Classifier", layout="centered")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: bold;
        }
        .stTextArea, .stTextInput {
            border: 2px solid #333;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Sentiment Classification for Tech Tweets")
st.markdown("This app predicts the sentiment of a tweet (Apple/Google related) using the trained **SVM with TF-IDF** model.")

# Input box
user_input = st.text_area(
    "Enter your tweet or sentence below:",
    placeholder="e.g., The new iPhone update totally broke my battery life! 😠",
    height=150
)

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess the input
        cleaned_text = clean_tweet(user_input)

        # Make prediction
        try:
            # The model is expected to be a Pipeline (TFIDF + SVM)
            prediction = model.predict([cleaned_text])[0]
            probas = model.predict_proba([cleaned_text])[0]
        except Exception as e:
            st.error(f"Prediction Error: Could not run model prediction. {e}")
            st.stop()

        # Map the prediction to human-readable sentiment
        predicted_sentiment = sentiment_mapping.get(prediction, "Unknown (Error)")

        st.divider()

        st.markdown(f"### Predicted Sentiment: **{predicted_sentiment}**")
        st.write(f"The model's internal class ID is: **{prediction}**")

        st.write("#### Class Probabilities:")
        col1, col2, col3 = st.columns(3)
        class_cols = [col1, col2, col3]

        for i, prob in enumerate(probas):
            sentiment_label = sentiment_mapping.get(i, f"Class {i}")
            color = ""
            if i == 0: color = "red"
            elif i == 2: color = "green"
            else: color = "orange"

            with class_cols[i]:
                st.metric(
                    label=sentiment_label,
                    value=f"{prob*100:.2f}%"
                )
