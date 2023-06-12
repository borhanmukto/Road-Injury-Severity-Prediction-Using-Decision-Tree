
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

clf = DecisionTreeClassifier(random_state=42)

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

# Seperate sensitivity and specificity for both classes

sensitivity = cm.diagonal() / cm.sum(axis=1)


specificity = []
num_classes = len(le.classes_)
for i in range(num_classes):
    tn = np.delete(cm, i, axis=0).sum()  
    fp = np.delete(cm[:, i], i, axis=0).sum()  
    specificity.append(tn / (tn + fp))


for i, class_name in enumerate(le.classes_):
    print(f"Class: {class_name}")
    print(f"Sensitivity: {sensitivity[i]}")
    print(f"Specificity: {specificity[i]}")
    print()
