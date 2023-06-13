#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Accident Prediction\Github Repo Decision Tree\final_accident_data.csv")

df


# In[11]:


X= df.drop(columns="Accident severity")
y= df["Accident severity"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

encoder = OneHotEncoder(drop="first")

X_train_encoded = encoder.fit_transform(X_train)

X_test_encoded = encoder.transform(X_test)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=4,criterion="entropy",random_state=42)

clf.fit(X_train_encoded, y_train_encoded)

y_pred = clf.predict(X_test_encoded)

training_accuracy = clf.score(X_train_encoded, y_train_encoded)
testing_accuracy = clf.score(X_test_encoded, y_test_encoded)
diff = abs(training_accuracy - testing_accuracy)
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)
print("Difference: ", diff)


# In[14]:


# # Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test_encoded, y_pred)

print(f"Confusion Matrix: \n", confusion_matrix)



# In[21]:


# # Classification Report

from sklearn.metrics import classification_report

cr = classification_report(y_test_encoded, y_pred)

print(cr)

