import joblib
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

# --- Predict for a new sample ---
print("\n--- Predict PlayTennis for new input values ---")
# Example: Accept input for all features except 'PlayTennis'
input_data = {}
for col in td.drop('PlayTennis', axis=1).columns:
	val = input(f"Enter value for {col}: ")
	input_data[col] = val

# Convert to DataFrame and encode categorical features
new_df = pd.DataFrame([input_data])
new_df_encoded = pd.get_dummies(new_df)
# Align columns with training data
new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)
# Predict
pred = clf.predict(new_df_encoded)
print(f"Predicted PlayTennis: {pred[0]}")

# --- Dump model to pickle file ---
joblib.dump(clf, r"D:\python\New folder\Linear regression\Decision Tree\decision_tree_model.pkl")
print("Decision tree model has been saved to 'decision_tree_model.pkl'.")
print("\n--- Load model from pickle and predict ---")
loaded_clf = joblib.load(r"D:\python\New folder\Linear regression\Decision Tree\decision_tree_model.pkl")
loaded_pred = loaded_clf.predict(new_df_encoded)
print(f"Predicted PlayTennis (from loaded model): {loaded_pred[0]}")


# Visualize
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=y.unique().astype(str), filled=True)
plt.show()

