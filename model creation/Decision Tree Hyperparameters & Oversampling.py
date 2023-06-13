#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np

df = pd.read_csv("balanced_oversampled_accident_data_1180_942.csv")

df


# In[26]:


from sklearn.model_selection import train_test_split

X= df.drop(columns="Accident severity")
y= df["Accident severity"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import OneHotEncoder

# Instantiate the OneHotEncoder
encoder = OneHotEncoder(drop="first")

# Fit and transform the categorical features in the training set
X_train_encoded = encoder.fit_transform(X_train)

# Transform the categorical features in the test set using the encoder learned from the training set
X_test_encoded = encoder.transform(X_test)

from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder
le = LabelEncoder()

# Fit and transform the categorical labels
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


# In[45]:


from sklearn.tree import DecisionTreeClassifier

# Instantiate the DecisionTreeClassifier


clf = DecisionTreeClassifier(max_depth=7, min_samples_split=2, min_samples_leaf=5,criterion="gini", random_state=50)

# clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=4,criterion="entropy")


# Fit the classifier to the training data
clf.fit(X_train_encoded, y_train_encoded)

# Predict on the test data
y_pred = clf.predict(X_test_encoded)

# Evaluate the accuracy
training_accuracy = clf.score(X_train_encoded, y_train_encoded)
testing_accuracy = clf.score(X_test_encoded, y_test_encoded)

# Print the accuracy
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)

diff = abs(training_accuracy - testing_accuracy)

print(f"Difference: ", diff)


# In[47]:


from sklearn.metrics import classification_report

cr = classification_report(y_test_encoded, y_pred)

print(cr)

