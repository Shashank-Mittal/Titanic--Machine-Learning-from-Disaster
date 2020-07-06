
# coding: utf-8

# In[1103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train=pd.read_csv("train.csv")


# In[1104]:


df_train.info()


# In[1105]:


X = df_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']].copy()


# # Analysis

# # Outliers

# In[1106]:


X.boxplot(column='Age')


# In[1107]:


count=0
for i in X["Age"]:
    if(i>65):
        count+=1
count


# In[1108]:


X.boxplot(column='Pclass')


# In[1109]:


X.boxplot(column='SibSp')


# In[1110]:


X.boxplot(column='Parch')


# In[1111]:


#X["Fare"] = np.log(X["Fare"]+1)


# In[1112]:


#X.boxplot(column='Fare')


# ## Missing values

# In[1113]:


X["Age"].fillna(X["Age"].mean(),inplace=True)
X["Embarked"].fillna("S",inplace=True)
X.info()


# ## Encoding

# In[1114]:


X['Sex'].replace(to_replace='male',value=1,inplace=True)
X['Sex'].replace(to_replace='female',value=0,inplace=True)


# In[1115]:


X['Embarked'].replace(to_replace='C',value=0,inplace=True)
X['Embarked'].replace(to_replace='Q',value=1,inplace=True)
X['Embarked'].replace(to_replace='S',value=2,inplace=True)
X.info()


# # One Hot Encoding

# In[1116]:


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

# In[1117]:


for i in X:
    #if(i!="Fare"):
    X[i] = (X[i] - min(X[i]))/(max(X[i]) - min(X[i]))


# # Feature Selection

# In[1118]:


X.corr()


# In[1119]:


X.info()


# In[1120]:


Features = X[["Pclass","Sex","Fare","0","1","2"]].copy()


# # Train Test split

# In[1121]:


x = Features.iloc[:,:]
y = X.iloc[:,7:8]


# In[1122]:


split_value = round(len(x) - (len(x)*0.25))
x_train=np.array(x[:split_value])
x_test=np.array(x[split_value:])
y_train = np.array(y[:split_value])
y_test = np.array(y[split_value:])


# In[1123]:


y_test.shape


# # Model

# In[1136]:


class Logistic_Regression:
    
    def __init__(self,learning_rate=0.001, Iteration=1000):
        
        self.learning_rate = learning_rate
        self.Iteration = Iteration
        self.Intercept = 0
    
    def Hypothesis_func(self,x_train):
        
        Z = np.dot(x_train,self.slope) + self.Intercept  #slope: nx1    x_train: mxn  Z: mx1
        return Z
    
    def sigmoid(self,Z):
        
        h = 1/ (1+ np.exp((-Z))) # h: mx1
        return h
    
    def Cost_function(self,h,y_train):
        
        m=y_train.shape[0]
        J = (1/m)*(np.dot(((-1)*(y_train).transpose()),np.log(h)) - np.dot((1-y_train).transpose(),np.log(1-h))) # y__train : 1xm h
        return J
    
    def Gradient_descent(self,h,y_train,x_train):
        
        m=y_train.shape[0]
        dw = (1/m)*np.dot(x_train.transpose(),(h-y_train)) # dw : nx1    y_train: mx1   x__train.transpose: nxm
        db = (1/m)* np.sum(h-y_train)
        
        self.slope = self.slope - (self.learning_rate*dw)  # self.slope: nx1
        self.Intercept = self.Intercept - (self.learning_rate*db)
        
    
    def fit(self,x_train,y_train):
        self.slope=np.zeros((x_train.shape[1],1))  # nX1
        c=[]
        it=[]
        
        for i in range(self.Iteration):
            
            Z = self.Hypothesis_func(x_train) #mx1
            
            h = self.sigmoid(Z) # mx1
            
            J = self.Cost_function(h,y_train) # h: mx1  y_train: mx1
            
            a=self.Gradient_descent(h,y_train,x_train) # h: mx1 ; y_train: mx1 ; x_train: mxn
            
            if(i%10==0):
                c.append(J[0][0])
                it.append(i)

        plt.plot(it,c) 
        plt.show()
    
    def predict(self,x_test):
        y_pred = np.dot(x_test,self.slope) + self.Intercept # x_test: mxn self.slope:nx1
        y_pred= self.sigmoid(y_pred)
        #new=[]
        #y_pred = [new.append([1]) if i>0.5 else new.append([0]) for i in y_pred]
        return y_pred

Lg = Logistic_Regression(learning_rate=0.008,Iteration=2000)
Lg.fit(x_train,y_train)
y_p=Lg.predict(x_test)
new=[]
for i in y_p:
    if(i[0]<0.5):
        new.append([0])
    else:
        new.append([1])
from sklearn.metrics import confusion_matrix
results = confusion_matrix(y_test,new)
print(results)


# In[1025]:


#from sklearn.linear_model import LogisticRegression
#lg = LogisticRegression() 
#lg.fit(x_train,y_train)


# In[1026]:


#y_pred = lg.predict(x_test)


# In[1027]:


#from sklearn.metrics import confusion_matrix
#results = confusion_matrix(y_test,y_pred)
#print(results)


# In[1028]:


Test = pd.read_csv("AfterPreprocessing2.csv")
Features_test = Test[["Pclass","Sex","Fare","0","1","2"]].copy()
Feature_test1 = np.array(Features_test.iloc[:,:])
#lg_new = LogisticRegression() 
#lg_new.fit(x,y)
#prediction = lg_new.predict(Features_test)


# In[1029]:


Feature_test1.shape


# In[1030]:


prediction=Lg.predict(Feature_test1)
new=[]
for i in prediction:
    if(i[0]<0.5):
        new.append(0)
    else:
        new.append(1)
print(type(np.array(new)))
#from sklearn.metrics import confusion_matrix
#results = confusion_matrix(y_test,new)
#print(results)
#prediction = L.g


# In[1031]:



prediction2 = pd.DataFrame({'Survived':new[:]})


# In[1032]:


Submission = Test[["PassengerId"]]
Submission["Survived"] = prediction2.copy()


# In[1033]:


Submission


# In[1034]:


Submission.to_csv("Submission2.csv")

