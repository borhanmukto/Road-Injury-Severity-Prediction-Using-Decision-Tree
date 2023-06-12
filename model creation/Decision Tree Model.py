
import pandas as pd
import numpy as np

df = pd.read_csv("final_accident_data.csv")

X= df.drop(columns="Accident severity")
y= df["Accident severity"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder(drop="first")

X_train_encoded = encoder.fit_transform(X_train)

X_test_encoded = encoder.transform(X_test)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

# clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=2,criterion="gini",class_weight=class_weights)


# clf = DecisionTreeClassifier(ccp_alpha= 0.0476, class_weight=class_weights)

clf.fit(X_train_encoded, y_train_encoded)

y_pred = clf.predict(X_test_encoded)

training_accuracy = clf.score(X_train_encoded, y_train_encoded)
testing_accuracy = clf.score(X_test_encoded, y_test_encoded)

print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)

# # Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_encoded, y_pred)

print(f"Confusion Matrix: \n", cm)


# # Classification Report

from sklearn.metrics import classification_report

cr = classification_report(y_test_encoded, y_pred, zero_division=0)

print(cr)
