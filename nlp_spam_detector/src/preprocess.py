import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

df = pd.read_csv(r'D:\python\New folder\nlp_spam_detector\data\spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)
df['clean_message'] = df['message'].apply(clean_text)

print(df.head())

