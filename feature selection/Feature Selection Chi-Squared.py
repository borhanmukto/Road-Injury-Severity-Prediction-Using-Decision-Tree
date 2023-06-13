
import pandas as pd
import numpy as np

df = pd.read_csv("accident_data.csv")

from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    
    


# # Chi Squared Statistics




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

