# filename: text_analysis.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK resources (run once)
nltk.download('punkt_tab')
nltk.download('punkt')

# Sample text
text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence 
that focuses on the interaction between computers and humans through natural language.
"""

# Step 1: Tokenize the text into words
tokens = word_tokenize(text.lower())  # convert to lowercase

# Step 2: Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

# Step 3: Count word frequency
word_freq = Counter(filtered_tokens)

# Display results
print("Filtered Tokens:", filtered_tokens)
print("\nWord Frequencies:")
for word, freq in word_freq.items():
    print(f"{word}: {freq}")
