from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(preprocess.df['clean_message']).toarray()
y = preprocess.df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:",confusion_matrix(y_test, y_pred))
print("report:",classification_report(y_test, y_pred))


pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
