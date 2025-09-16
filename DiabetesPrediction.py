import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns

#load and explore data
dt = pd.read_csv(r"D:\MYDOCS\AI\unclean_diabetes_dataset.csv")
#showing first 5 rows
dt.head(5)

#check for missing values
dt.isnull().any()
dt.isna().any()
#data cleaning
df = dt.copy() # Create a copy to avoid modifying the original DataFrame directly
df = df.drop_duplicates()

# List of columns to process
cols_to_process = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Convert columns to numeric, coercing errors to NaN
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace NaN values (originally '?') with the mean of each column
for col in cols_to_process:
    df[col] = df[col].fillna(df[col].mean())


#remove zeros in the columns
df = df[(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] != 0).all(axis=1)]

X=df.drop(columns=['Outcome'],axis=1)
y=df['Outcome']

print(df)
#Display dataset statistics (mean, median, etc.)
