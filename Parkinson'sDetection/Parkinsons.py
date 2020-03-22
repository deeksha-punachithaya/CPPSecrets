import pandas as pd
import numpy as np
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('parkinsons.csv')
print(df.head())

print(df['class'].unique())
print(df.shape)
print(df.info())
print(df.describe())

X = df.drop(['class'],axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = svm.SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# Apply feature selection techniques :- Recursive Feature Elimination

from sklearn.feature_selection import RFE
rfe = RFE(model)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

print(len(rfe.support_))
arr = X.columns[rfe.support_]
print(arr)

X2 = X[arr].set_index('id')
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
