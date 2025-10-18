import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# streamlit set-up
st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="centered"
)

st.title("Sentiment Analysis Comparison App")
st.write("Compare traditional **TF-IDF + SVM** vs **Transformer-based RoBERTa** sentiment predictions.")


# load model
@st.cache_resource(show_spinner=False)
def load_svm_components():

    """Load the SVM model and TF-IDF vectorizer from disk once."""

    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    svm_model = joblib.load("sentiment_svm_balanced_model.pkl")
    return vectorizer, svm_model



@st.cache_resource(show_spinner=True)
def load_roberta_pipeline():
    """Load RoBERTa model & tokenizer safely on CPU (no meta tensors)."""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map=None,            # prevent meta tensors
        torch_dtype=torch.float32,  # use float32 for CPU
    )
    model = model.to("cpu")

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1  # force CPU
    )
    return sentiment_analyzer


# helper functions
def predict_svm(text, vectorizer, model):
    processed = text.lower()
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    return prediction


def predict_roberta(text, analyzer):
    result = analyzer(text)[0]
    label = result["label"].replace("LABEL_", "")
    score = f"{result['score']:.2f}"
    return f"{label} ({score})"




# App
vectorizer, svm_model = load_svm_components()
sentiment_analyzer = load_roberta_pipeline()

user_text = st.text_area("Enter tweet to analyze:", "I love my new iPhone!")

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing sentiment... please wait"):
        svm_result = predict_svm(user_text, vectorizer, svm_model)
        roberta_result = predict_roberta(user_text, sentiment_analyzer)

        st.subheader("RESULTS")
        st.write(f"**SVM Prediction:** {svm_result}")
        st.write(f"**RoBERTa Prediction:** {roberta_result}")

        st.markdown("---")
        st.caption("✅ Model weights cached for faster reloads.")


#footer
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
    Built with Streamlit, scikit-learn, and Hugging Face Transformers.
    </div>
    """,
    unsafe_allow_html=True
)



