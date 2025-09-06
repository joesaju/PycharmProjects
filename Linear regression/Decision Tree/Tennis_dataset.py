import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

td = pd.read_csv(r"D:\MYDOCS\AI\play_tennis_dataset.csv")
print(td)

X = pd.get_dummies(td.drop('PlayTennis', axis=1))
y = td['PlayTennis']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
print("Train Accuracy:", clf.score(X_train, y_train))
print("Test Accuracy:", clf.score(X_test, y_test))
# Visualize
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=y.unique().astype(str), filled=True)
plt.show()