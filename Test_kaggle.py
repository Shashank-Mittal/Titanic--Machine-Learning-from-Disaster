
# coding: utf-8

# In[194]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_test = pd.read_csv("test.csv")
df_test.info()


# In[195]:


X = df_test[["PassengerId",'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',]].copy()


# # outliers

# In[196]:


#X["Fare"] = np.log(X["Fare"]+1)


# In[197]:


X.boxplot(column='Fare')


# # Missing Values

# In[198]:


X["Age"].fillna(X["Age"].mean(),inplace=True)
X["Fare"].fillna(X["Fare"].mean(),inplace=True)
X["Embarked"].fillna("S",inplace=True)
X.info()


# ## Encoding

# In[199]:


X['Sex'].replace(to_replace='male',value=1,inplace=True)
X['Sex'].replace(to_replace='female',value=0,inplace=True)


# In[200]:


X['Embarked'].replace(to_replace='C',value=0,inplace=True)
X['Embarked'].replace(to_replace='Q',value=1,inplace=True)
X['Embarked'].replace(to_replace='S',value=2,inplace=True)


# In[201]:


X.info()


# ## One Hot Encoding

# In[202]:


from sklearn.preprocessing import OneHotEncoder
integer_encoded = X["Embarked"].values.reshape(len(X["Embarked"]), 1)

### One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

ab = pd.DataFrame({'0':onehot_encoded[:,0],'1':onehot_encoded[:,1],'2':onehot_encoded[:,2]})
X['0'] = ab[["0"]]
X['1'] = ab[["1"]]
X['2'] = ab[["2"]]
X


# # Feature Scalling

# In[203]:


for i in X:
    if( i!="PassengerId"):
        X[i] = (X[i] - min(X[i]))/(max(X[i]) - min(X[i]))


# In[204]:


X


# In[205]:


X.to_csv("AfterPreprocessing2.csv")

