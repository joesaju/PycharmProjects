import streamlit as st
import pickle, os

# Cache the model and vectorizer so they load only once
@st.cache_resource
def load_model_and_vectorizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, r"D:\python\New folder\nlp_spam_detector\models\spam_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(current_dir, r"D:\python\New folder\nlp_spam_detector\models\vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.title("📧 NLP Spam Detector")

input_msg = st.text_input("Enter your message")

if st.button("Predict"):
    msg_vec = vectorizer.transform([input_msg])
    result = model.predict(msg_vec)[0]
    st.success("✅ Not Spam" if result == 0 else "🚨 Spam Message Detected")
