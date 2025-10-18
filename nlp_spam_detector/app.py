import streamlit as st
import pickle
# Load saved model and vectorizer
model = pickle.load(open(r'spam_model.pkl', 'rb'))
vectorizer = pickle.load(open(r'vectorizer.pkl', 'rb'))


st.title("SMS Spam Detector")
input_msg = st.text_area("Enter a message:")
if st.button("Predict"):
    msg_vec = vectorizer.transform([input_msg])
    prediction = model.predict(msg_vec)
    st.write("Result:", "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam")
