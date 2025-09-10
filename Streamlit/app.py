import streamlit as st
import pickle
import numpy as np

with open(r"D:\python\New folder\Streamlit\classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Revenue Predictor")
st.write("Enter the following details:")

Page_Views = st.slider("Views", min_value=50.0, max_value=500.0)
Clicks = st.slider("Clicks", min_value=10.0, max_value=100.0)
Signups = st.slider("Signups", min_value=1.0, max_value=20.0)

if st.button("Predict"):
    features = np.array([[Page_Views, Clicks, Signups]])
    prediction = model.predict(features)
    st.write(f"Predicted Revenue: {prediction[0]}")