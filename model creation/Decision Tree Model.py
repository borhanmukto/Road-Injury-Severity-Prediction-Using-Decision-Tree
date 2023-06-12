#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Accident Prediction\filled_accident_data_with_all_features.csv")

df = df.drop(columns=["Movement","Divider", "Light", "Road geometry", "Surface type", "Road feature",
        "Location type", "Vehicle defects", "Seat belt"])

df.shape


# In[2]:


df["Accident severity"].unique()


# In[3]:


df["Accident severity"] = df["Accident severity"].replace("Grievious", "Non Fetal Injury")
df["Accident severity"] = df["Accident severity"].replace("Simple Injury", "Non Fetal Injury")
df["Accident severity"] = df["Accident severity"].replace("Motor Collision", "Non Fetal Injury")
df["Junction type"] = df["Junction type"].replace("T Junction", "Junction")
df["Junction type"] = df["Junction type"].replace("Cross Road", "Junction")
df["Junction type"] = df["Junction type"].replace("Round About", "Junction")
df["Junction type"] = df["Junction type"].replace("Staggered Junction", "Junction")
df["Junction type"] = df["Junction type"].replace("Other", "Junction")
df["Traffic control"] = df["Traffic control"].replace("Police + Traffic Lights", "Other")
df["Traffic control"] = df["Traffic control"].replace("Centreline", "Other")
df["Traffic control"] = df["Traffic control"].replace("Pedestrian Crossing", "Other")
df["Traffic control"] = df["Traffic control"].replace("Traffic Lights", "Other")
df["Traffic control"] = df["Traffic control"].replace("Stop/Give Way Sign", "Other")

df


# In[4]:


df["Traffic control"].unique()


# In[5]:


from sklearn.model_selection import train_test_split

X= df.drop(columns="Accident severity")
y= df["Accident severity"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)



X_train.shape


# In[6]:


X_test.shape


# In[7]:


y_train.shape


# In[8]:


y_test.shape


# # One-Hot Encooding in X_train,X_test,y_train,y_test

# In[9]:


mode_value = df.iloc[:, 12].mode()[0]

mode_value

# df.iloc[:, 5].replace("Railway", mode_value, inplace=True)


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# In[11]:


X_train_encoded.shape


# In[12]:


X_test_encoded


# In[52]:


from sklearn.tree import DecisionTreeClassifier

# Instantiate the DecisionTreeClassifier

# class_weights = {0:1, 1: 2}

clf = DecisionTreeClassifier()

# clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=2,criterion="gini",class_weight=class_weights)


# clf = DecisionTreeClassifier(ccp_alpha= 0.0476, class_weight=class_weights)


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


# # Metrics

# In[49]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate precision
precision = precision_score(y_test_encoded, y_pred, average='micro')

# Calculate recall
recall = recall_score(y_test_encoded, y_pred, average='micro')

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Calculate F1 score
f1 = f1_score(y_test_encoded, y_pred, average='micro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# # Confusion Matrix

# In[50]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_encoded, y_pred)

print(f"Confusion Matrix: \n", cm)


# # Classification Report

# In[45]:


le.classes_


# In[51]:


from sklearn.metrics import classification_report

cr = classification_report(y_test_encoded, y_pred, zero_division=0)

print(cr)


# # New Sampling

# In[18]:


df['Accident severity'].value_counts()


# In[19]:


# df_sampled_1 = df[df['Accident severity'] == "Non Fetal Injury"].copy()


# In[20]:


# df_sampled_1.shape


# In[21]:


# np.random.seed(42)
# df_new_1 = df_sampled_1.sample(942, replace=True)

# df_new_1.shape


# In[22]:


# df_sampled_0 = df[df["Accident severity"]=="Fetal Injury"].copy()


# In[23]:


# np.random.seed(42)
# df_new_0 = df_sampled_0
# df_new_0.shape


# In[24]:


# df_sampled_0 = df_class_0.sample(314)
# df_sampled_0


# In[25]:


# df_combined = pd.concat([df_new_0, df_new_1])

# df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# df_combined.to_csv("balanced_oversampled_accident_data_1180_942.csv", index=False)

df_combined = pd.read_csv("balanced_oversampled_accident_data_1180_942.csv")

df_combined.shape


# In[26]:


mode_value = df_combined.iloc[:, 8].mode()[0]

mode_value


# In[27]:


df_combined.columns[8]


# In[28]:


from sklearn.model_selection import train_test_split

X_co= df_combined.drop(columns="Accident severity")
y_co= df_combined["Accident severity"]

X_train_co,X_test_co,y_train_co,y_test_co = train_test_split(X_co,y_co,test_size=0.2, random_state=42, stratify=y_co)


from sklearn.preprocessing import OneHotEncoder

# Instantiate the OneHotEncoder
encoder = OneHotEncoder(drop="first")

# Fit and transform the categorical features in the training set
X_train_encoded_co = encoder.fit_transform(X_train_co)

# Transform the categorical features in the test set using the encoder learned from the training set
X_test_encoded_co = encoder.transform(X_test_co)

from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder
le = LabelEncoder()

# Fit and transform the categorical labels
y_train_encoded_co = le.fit_transform(y_train_co)
y_test_encoded_co = le.transform(y_test_co)


# In[29]:


from sklearn.tree import DecisionTreeClassifier

# Instantiate the DecisionTreeClassifier

np.random.seed(50)

clf = DecisionTreeClassifier(max_depth=7, min_samples_split=2, min_samples_leaf=5,criterion="gini")

# clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=4,criterion="entropy")


# clf = DecisionTreeClassifier(ccp_alpha= 0.0058)


# Fit the classifier to the training data
clf.fit(X_train_encoded_co, y_train_encoded_co)

# Predict on the test data
y_pred_co = clf.predict(X_test_encoded_co)

# Evaluate the accuracy
training_accuracy = clf.score(X_train_encoded_co, y_train_encoded_co)
testing_accuracy = clf.score(X_test_encoded_co, y_test_encoded_co)

# Print the accuracy
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)

diff = abs(training_accuracy - testing_accuracy)

print(f"Difference: ", diff)


# # Oversampling class_0=1184, and class_1=942

# In[30]:


from sklearn.metrics import classification_report

cr = classification_report(y_test_encoded_co, y_pred_co, zero_division=0)

print(cr)


# In[31]:


y_test_encoded_co


# In[32]:


y_pred_co


# In[33]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_encoded_co, y_pred_co)

print("Confusion Matrix: ")
print(cm)


# In[34]:


from sklearn import metrics

# Convert the predicted labels back to the original class labels
y_pred_original_co = le.inverse_transform(y_pred_co)

# Convert the test labels back to the original class labels
y_test_original_co = le.inverse_transform(y_test_encoded_co)

# Calculate confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test_original_co, y_pred_original_co)

# Extract values from confusion matrix
true_negatives = confusion_matrix[0, 0]
false_positives = confusion_matrix[0, 1]
false_negatives = confusion_matrix[1, 0]
true_positives = confusion_matrix[1, 1]

# Calculate specificity
specificity = true_negatives / (true_negatives + false_positives)

# Calculate sensitivity
sensitivity = true_positives / (true_positives + false_negatives)

# Calculate AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test_encoded_co, clf.predict_proba(X_test_encoded_co)[:, 1])
auc = metrics.auc(fpr, tpr)

# Print the specificity, sensitivity, and AUC
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("AUC:", auc)


# # Learning Curve

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

# Instantiate the DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=2,criterion="gini")

# Define the sizes of the training set
train_sizes = np.linspace(0.1, 1.0, 10)

# Calculate the learning curves
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train_encoded_co, y_train_encoded_co, train_sizes=train_sizes, cv=5, scoring='accuracy', n_jobs=-1
)

# Calculate the mean and standard deviation of the training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', linestyle='-', color='b', label='Training Accuracy')
plt.plot(train_sizes, test_mean, marker='o', linestyle='-', color='r', label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='b')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='r')
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curves - Decision Tree")
plt.legend()
plt.grid(True)
plt.show()


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

np.random.seed(50)

# Define the range of hyperparameters
param_grid = {
    'max_depth': [5, 6, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [2, 3, 5],
    'criterion': ['gini', 'entropy']
}

# Instantiate the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_encoded_co, y_train_encoded_co)

# Get the best classifier
best_classifier = grid_search.best_estimator_

# Calculate the learning curves for the best classifier
train_sizes, train_scores, test_scores = learning_curve(
    best_classifier, X_train_encoded_co, y_train_encoded_co, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

# Calculate the mean and standard deviation of the training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Calculate the difference between training accuracy and validation accuracy
diff = np.abs(train_mean - test_mean)

# Find the index of the lowest difference
min_diff_index = np.argmin(diff)

# Plot the learning curve with the lowest difference
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', linestyle='-', color='b', label='Training Accuracy')
plt.plot(train_sizes, test_mean, marker='o', linestyle='-', color='r', label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='b')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='r')
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curves - Decision Tree (Best Hyperparameters)")
plt.legend()
plt.grid(True)
plt.show()

# Print the best hyperparameters and the lowest difference
print("Best Hyperparameters:", grid_search.best_params_)
print("Lowest Difference (Training Accuracy - Validation Accuracy):", diff[min_diff_index])


# In[ ]:




