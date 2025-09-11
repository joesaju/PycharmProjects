import streamlit as st
import pickle
import numpy as np
import datetime

# Load model
with open(r"classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("💰 Revenue Predictor")
st.write("Enter the following details:")

# Inputs for numeric features
page_views = st.slider("Page Views", min_value=50, max_value=1000, step=10)
clicks = st.slider("Clicks", min_value=10, max_value=100, step=1)
signups = st.slider("Signups", min_value=1, max_value=50, step=1)

# Input for date (so we can generate Year, Month, Day, DayOfWeek)
date_input = st.date_input("Select Date", datetime.date.today())

# Convert date into features (same as training script)
year = date_input.year
month = date_input.month
day = date_input.day
day_of_week = date_input.weekday()

if st.button("Predict"):
    # Arrange features in the same order as training
    features = np.array([[page_views, clicks, signups, year, month, day, day_of_week]])
    
    prediction = model.predict(features)
    st.success(f"📊 Predicted Revenue: {prediction[0]:.2f}")


