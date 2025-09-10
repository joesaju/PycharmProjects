import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

df = pd.read_csv(r"D:\MYDOCS\AI\streamlit_practice_dataset.csv")

df.drop('User_ID', axis=1, inplace=True)

X = df.drop(columns=['Page_Views', 'Signups', 'Date'])
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

with open(r"D:\python\New folder\Streamlit\classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

