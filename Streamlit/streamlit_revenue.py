import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle

# Load dataset
df = pd.read_csv(r"D:\MYDOCS\AI\streamlit_practice_dataset.csv")

# Drop irrelevant column
df.drop('User_ID', axis=1, inplace=True)

# ---- Feature Engineering ----
# Convert "Date" into useful features instead of dropping it
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df.drop('Date', axis=1, inplace=True)

# Define features & target
X = df.drop(columns=['Revenue'])   # keep ALL features except target
y = df['Revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Model Training with tuned parameters ----
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
with open(r"D:\python\New folder\Streamlit\classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
