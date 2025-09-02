import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

new_data = pd.read_csv(r"D:\python\New folder\Linear regression\Youtube Data.csv")

#removing the column name 'unnamed':
new_data=new_data.drop(['Unnamed: 15'],axis=1)

new_data.drop(401,axis=0,inplace=True)
print(new_data)

dataframe = new_data
le = LabelEncoder()
dataframe['title'] = le.fit_transform(dataframe['title'])
dataframe['description'] = le.fit_transform(dataframe['description'])
dataframe['channel_title'] = le.fit_transform(dataframe['channel_title'])
dataframe['tags'] = le.fit_transform(dataframe['tags'])
print(dataframe)

#plotting the data in graph
sns.pairplot(dataframe[['comment_count','view_count','like_count']])
plt.show()
#predicting view counts
x=new_data[['comment_count','like_count']]
y=new_data['view_count']
model = LinearRegression()
model.fit(x, y)
joblib.dump(model, r"D:\python\New folder\Linear regression\view_count.pkl")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = X_train.fillna(X_train.mean())
X_test  = X_test.fillna(X_test.mean())
y_train = y_train.fillna(y_train.mean())
model = LinearRegression()
model.fit(X_train, y_train)


like_var = int(input("Enter like count: "))
comment_var = int(input("Enter comment count: "))
# Get the current working directory
current_dir = os.getcwd()

# Construct the full path to the model file
model_path = os.path.join(current_dir, r"D:\python\New folder\Linear regression\view_count.pkl")

def predicted_view(like, comment):
  # Load the model
  model = joblib.load(model_path)

  user_data = np.array([[like, comment]])
  predicted_view = model.predict(user_data)[0]
  return predicted_view

print(predicted_view(like_var, comment_var))


y_pred = model.predict(X_test)
#calculating rsquare
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

corr = new_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


