import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load dataset
data=pd.read_csv(r"D:\MYDOCS\AI\adults.csv")
print(data.head(20))

#data cleaning
print(data.head())
print(data.tail())
print(data.columns)
print(data.shape)
print(data.info)
print(data.describe())
print(data.isnull().sum())
print(data.isnull())
print(data.duplicated())
print(data.drop_duplicates(inplace=True))
print(data)
print(data.head())

# Define column names
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

# Create DataFrame with proper column names
df = pd.DataFrame(data.values, columns=columns)
# Clean data - replace '?' with NaN and handle missing values
df = df.replace('?', np.nan)
df = df.dropna()
print(df)

# Explicitly convert potential object columns to string type and strip leading/trailing
object_cols = ['workclass', 'education', 'marital_status', 'occupation',
'relationship', 'race', 'sex', 'native_country', 'income']
for col in object_cols:
	df[col] = df[col].astype(str).str.strip()
	
# Convert numeric columns
numeric_cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Drop any remaining rows with NaN values (if any after numeric conversion)
df = df.dropna()
print(df)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
x = df.drop('income', axis=1)
y = df['income']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
# Select only numeric columns for scaling
x_train_scaled = scaler.fit_transform(x_train[numeric_cols])
x_test_scaled = scaler.transform(x_test[numeric_cols])

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200,random_state=42)
model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", matrix)

# --- Deploy: Predict for new input values ---
print("\n--- Predict income for new input values ---")
# Example: Accept input for all features except 'income'
input_data = {}
for col in x.columns:
    if col in numeric_cols:
        val = float(input(f"Enter value for {col}: "))
        input_data[col] = val
    elif col in categorical_cols:
        le = label_encoders[col]
        val = input(f"Enter value for {col} (choose from: {list(le.classes_)}): ")
        # If value not found, default to first class
        if val not in le.classes_:
            print(f"Value not found, using default: {le.classes_[0]}")
            val = le.classes_[0]
        input_data[col] = le.transform([val])[0]

# Convert to DataFrame
new_df = pd.DataFrame([input_data])
# Scale numeric columns
new_df_scaled = scaler.transform(new_df[numeric_cols])
# Predict
pred = model.predict(new_df_scaled)
# Decode prediction
income_pred = label_encoders['income'].inverse_transform(pred)
print(f"Predicted income class: {income_pred[0]}")

# --- Scatter plot (visualization) ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='age', y='hours_per_week', hue='income', data=df)
plt.title('Scatter Plot of Age vs. Hours per Week')
plt.show()

#heatmap correlation
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

