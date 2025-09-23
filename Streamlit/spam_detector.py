import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Sample Training Data ---
emails = [
    "Win a free iPhone now!!! Click here to claim your prize",  # spam
    "Lowest price on meds, buy now and save big",              # spam
    "Congratulations, you have won $1000 gift card",           # spam
    "Meeting scheduled tomorrow at 10am",                      # ham
    "Can we have a call about the project updates?",            # ham
    "Your Amazon order has been shipped successfully",          # ham
]

labels = ["spam", "spam", "spam", "ham", "ham", "ham"]

# --- Train a simple model ---
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(emails, labels)

# --- Streamlit App ---
st.title("📧 Spam Detector")
st.write("Enter an email and check if it is **Spam** or **Ham**.")

# User input
user_email = st.text_area("Paste your email text here:")

if st.button("Check"):
    if user_email.strip():
        prediction = model.predict([user_email])[0]
        if prediction == "spam":
            st.error("🚨 This email looks like **SPAM**")
        else:
            st.success("✅ This email looks like **HAM (Not Spam)**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
