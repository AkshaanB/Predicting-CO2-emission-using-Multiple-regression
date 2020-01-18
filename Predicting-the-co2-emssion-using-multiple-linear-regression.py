#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[2]:


url = "D:\My Work\Otherthan Syllabus\Data Science\Projects\FuelConsumption.csv"


# In[3]:


df = pd.read_csv(url)


# In[4]:


df


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[8]:


wanted_data = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]


# In[9]:


wanted_data


# In[12]:


plt.scatter(wanted_data.ENGINESIZE,wanted_data.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[14]:


msk = np.random.rand(len(df)) <0.8
train = wanted_data[msk]
test = wanted_data[~msk]


# In[18]:


plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()


# In[23]:


#Multiple linear regression is an extension of simple linear model
regression = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(x,y)
print("Coefficents: ",regression.coef_)
print("Intercept: ",regression.intercept_)


# In[24]:


y_hat = regression.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])


# In[27]:


print("Predicted co2 emission: ",y_hat)


# In[28]:


#finding residual
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares (MSE): %.2f"%np.mean((y_hat-y)**2))


# In[42]:


#finding variance
# 𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )=1−𝑉𝑎𝑟{𝑦−𝑦̂ }𝑉𝑎𝑟{𝑦}
explained_variance = regression.score(x,y)
print('Explained Variance score: %.2f' % explained_variance) #1 is perfect prediction


# In[44]:


print("So, the model is: %.2f" %(explained_variance*100),"% perfect")


# In[ ]:




