#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib as plt
from matplotlib.pyplot import boxplot
from sklearn.linear_model import LogisticRegression


# In[8]:


current_directory = os.getcwd()
print(current_directory)


# In[9]:


updated_dir = os.chdir(r"C:\Users\sanja")


# In[11]:


filepath = 'Week14Assignment.txt'
df = pd.read_csv(filepath)


# In[13]:


print(df.columns)


# In[15]:


filepath = 'Week14Assignment.txt'
df = pd.read_csv(filepath)


# In[16]:


print(df.columns)


# In[17]:


# calculating statistics
num_readmitted = np.sum(df[' Readmission'])
satisfaction_staff = np.mean(df[' StaffSatisfaction'])
satisfaction_cleanliness = np.mean(df[' CleanlinessSatisfaction'])
satisfaction_food = np.mean(df[' FoodSatisfaction'])
satisfaction_comfort = np.mean(df[' ComfortSatisfaction'])
satisfaction_communication = np.mean(df[' CommunicationSatisfaction'])


# In[18]:


#printing out descriptive statistics
print(f"Number of patients readmitted: {num_readmitted}.")
print(f"Average staff satisfaction: {satisfaction_staff}.")
print(f"Average cleanliness satisfaction: {satisfaction_cleanliness}.")
print(f"Average food satisfaction: {satisfaction_food}.")
print(f"Average comfort satisfaction: {satisfaction_comfort}.")
print(f"Average communication satisfaction: {satisfaction_communication}.")


# In[23]:


#calculated overall satisfaction
df['OverallSatisfaction'] = df[[' StaffSatisfaction', ' CleanlinessSatisfaction', 
                                ' FoodSatisfaction', ' ComfortSatisfaction',
                               ' CommunicationSatisfaction']].mean(axis=1)

boxplot(df['OverallSatisfaction'], showfliers=True)


# In[25]:


# logistic regression
X = df['OverallSatisfaction'].values.reshape(-1, 1)
Y = df[' Readmission']

log_reg = LogisticRegression().fit(X, Y)


# In[28]:


# correlation results
correlation_coefficient = log_reg.coef_[0][0]

if correlation_coefficient > 0:
    print("Logistic regression results indicated a: ")
    if correlation_coefficient > 0.5:
        print ("Moderate Correlation")
    elif correlation_coefficient > 0.7:
        print ("Strong Correlation")
    else:
        print ("Weak correlation")
else:
    print("Logistic regression results indicated: ")
    print("No correlation")
    
print(f"The correlation coefficient was {correlation_coefficient}.")


# In[29]:


# plot the data
plt.pyplot.scatter(X, Y)
plt.pyplot.xlabel('Overall Satisfaction Scores')
plt.pyplot.ylabel('Readmission')
plt.pyplot.title('Logistic Regression - Overall Satisfaction vs Readmission')
plt.pyplot.plot(X, log_reg.predict(X), color = 'blue')
plt.pyplot.xlim(2, 5)

