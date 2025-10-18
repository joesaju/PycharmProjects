import re
import pickle
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed first time)
nltk.download('stopwords', quiet=True)

# Load saved model and vectorizer
model = pickle.load(open(r'D:\python\New folder\nlp_spam_detector\spam_model.pkl', 'rb'))
vectorizer = pickle.load(open(r'D:\python\New folder\nlp_spam_detector\vectorizer.pkl', 'rb'))

# Text cleaning function (same as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Prediction function
def predict_message(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    if prediction == 1:
        return "🚫 Spam"
    else:
        return "✅ Not Spam"

# Test script (you can run directly)
if __name__ == "__main__":
    print("=== SMS Spam Detector ===")
    while True:
        msg = input("\nEnter a message (or type 'exit' to quit): ")
        if msg.lower() == 'exit':
            break
        result = predict_message(msg)
        print("Result:", result)
