#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("accident_data_duplicates_dropped.csv")


df


# # Label Encoder

# In[2]:


from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    
    
df.head()


# # Chi Squared Statistics

# In[8]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X= df.drop(columns="Accident severity")
y=df["Accident severity"]

selector=SelectKBest(score_func=chi2, k=15)
selector.fit(X,y)

chi_scores=selector.scores_
p_values=selector.pvalues_

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))

plt.bar(X.columns, chi_scores)
plt.xlabel("Features")
plt.ylabel("Chi Squared Scores")
plt.title("Chi-squared vs Features")
plt.xticks(rotation=90)  # Rotate X-axis labels by 90 degrees


plt.show()





# In[6]:


df.columns


# In[8]:


df.drop(columns=["Movement","Divider", "Light", "Road geometry", "Surface type", "Road feature",
        "Location type", "Vehicle defects", "Seat belt"])


# In[9]:


df.to_csv("accident_data_Label_encoded_best_features_selected.csv", index=False)

