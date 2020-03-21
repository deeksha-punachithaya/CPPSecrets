#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()


# In[11]:


import numpy as np
df['class'].unique()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[22]:


X = df.drop(['class'],axis=1)
y = df['class']


# In[53]:


from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train)


# In[54]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[55]:


accuracy = accuracy_score(y_test, predictions)
accuracy


# In[23]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = XGBClassifier()
model.fit(X_train, y_train)


# In[24]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[25]:


accuracy = accuracy_score(y_test, predictions)
accuracy


# ### Apply feature selection techniques

# #### Recursive Feature Elimination

# In[30]:


from sklearn.feature_selection import RFE
rfe = RFE(model)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)


# In[31]:


len(rfe.support_)


# In[33]:


arr = X.columns[rfe.support_]


# In[34]:


arr


# In[36]:


X2 = X[arr].set_index('id')
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)

model.fit(X_train, y_train)


# In[37]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy

