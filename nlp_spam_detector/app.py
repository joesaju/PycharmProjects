import streamlit as st
import pickle
import os

# Get the absolute path of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "spam_model.pkl")
vector_path = os.path.join(current_dir, "vectorizer.pkl")

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open(vector_path, "rb") as f:
    vectorizer = pickle.load(f)
# Load saved model and vectorizer
# model = pickle.load(open('models/spam_model.pkl', 'rb'))
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))


st.title("SMS Spam Detector")
input_msg = st.text_area("Enter a message:")
if st.button("Predict"):
    msg_vec = vector_path.transform([input_msg])
    prediction = model.predict(msg_vec)
    st.write("Result:", "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam")
