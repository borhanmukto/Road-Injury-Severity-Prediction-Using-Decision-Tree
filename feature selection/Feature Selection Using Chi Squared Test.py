#!/usr/bin/env python
# coding: utf-8

# # Feature Selection

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

accident_data = pd.read_csv(r"D:\Accident Prediction\Publication 2023\Preparing the raw dataset\final_accident_data.csv")

# Data Encoding
X = accident_data.drop(columns="Accident severity")  # Set the independent variables
y = accident_data["Accident severity"]  # Set the target variable

# Encoding categorical features using OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Split the datasets into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Perform chi2 feature selection and get the f-scores
f_scores, p_values = chi2(X_train, y_train)

# Create a DataFrame to store the feature names and corresponding f-scores
feature_scores = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores})

# Sort the DataFrame by f-scores in descending order
feature_scores = feature_scores.sort_values(by='F-Score', ascending=False)

# Plotting the feature scores in a bar chart
plt.figure(figsize=(10, 6))
plt.bar(feature_scores['Feature'], feature_scores['F-Score'])
plt.xlabel('Features')
plt.ylabel('F-Score')
plt.title('Feature Importance (Chi2)')
plt.xticks(rotation=90)
plt.show()

# Print the feature scores in descending order
print("Feature Importance (Chi2):")
print(feature_scores.head(15)) # printing important features


# In[4]:


feature_scores.iloc[15:, 0] # features that are not important


# In[ ]:




