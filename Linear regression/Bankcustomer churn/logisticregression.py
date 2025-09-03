import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,auc

bd = pd.read_csv(r"D:\MYDOCS\AI\Bank Customer Churn Prediction.csv")
print(bd.head(20))
print(bd.info())
print(bd.describe())
print(bd.isnull().any())
print(bd.isnull().sum())

bd.drop_duplicates()
print(bd)

print(bd.keys())

print(bd.head())
print(bd.columns)

#classification of data
x = bd[['credit_score', 'age']]
y = bd['churn']
print(np.median(y))
y_binary=(y>np.median(y)).astype(int)
print(y_binary)

#split the data into testing and training sets
x_train,x_test,y_train,y_test=train_test_split(x,y_binary,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

#evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix)
print("Accuracy:{:.2f}%".format(accuracy*100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

#predict
y_pred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("AUC: {:.2f}".format(roc_auc))

#visualize the decision boundary with matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, palette={0:'blue',1:'red'}, marker='o')
plt.xlabel('credit_score')
plt.ylabel('age')
plt.title('Accuracy rate of customer churn:{:.2f}%'.format(accuracy*100))
plt.legend(title="Customer churn", loc="upper right")
plt.show()

#heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()