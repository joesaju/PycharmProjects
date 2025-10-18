import streamlit as st
import pickle, os

# Cache the model and vectorizer so they load only once
@st.cache_resource
def load_model_and_vectorizer():
    # Get the absolute directory of this app file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build correct absolute paths
    model_path = os.path.join(current_dir, "models", "spam_model.pkl")
    vectorizer_path = os.path.join(current_dir, "models", "vectorizer.pkl")


    # Check if files actually exist
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Vectorizer file not found at: {vectorizer_path}")
        st.stop()

    # Load them safely
    with open(model_path, "rb") as mf:
        model = pickle.load(mf)
    with open(vectorizer_path, "rb") as vf:
        vectorizer = pickle.load(vf)

    return model, vectorizer

# Load both files
model, vectorizer = load_model_and_vectorizer()

st.title("📧 NLP Spam Detector")

input_msg = st.text_input("Enter your message")

if st.button("Predict"):
    msg_vec = vectorizer.transform([input_msg])
    result = model.predict(msg_vec)[0]
    st.success("✅ Not Spam" if result == 0 else "🚨 Spam Message Detected")
