import streamlit as st
import pickle
import numpy as np

with open(r"D:\python\New folder\Streamlit\classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Revenue Predictor")
st.write("Enter the following details:")

Page_Views = st.slider("Views", min_value=50, max_value=500)
Clicks = st.slider("Clicks", min_value=10, max_value=100)
Signups = st.slider("Signups", min_value=1, max_value=200)

if st.button("Predict"):
    features = np.array([[Page_Views, Clicks, Signups]])
    prediction = model.predict(features)
    st.write(f"Predicted Revenue: {prediction[0]}")