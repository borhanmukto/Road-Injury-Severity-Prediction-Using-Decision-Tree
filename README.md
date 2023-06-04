
![Banner](https://drive.google.com/file/d/1iS3uh5wDqZ27Yihq8XM9-gJK1JsK2WpI/view?usp=sharing)



In this project I analyzed Accident Data from Accident Research Institute in Bangladesh from 2017 to 2020. Using this dataset I have prepared a model using Supervised Machine Learning Technics, Decision Tree.


# Workflow
#### 1. Data Collection from Accident Research Institute, BUET
#### 2. Data Preprocessing
    - Data Cleaning
    - Irrelevant Features Elimination (initial)
    - Preparation of final dataset with 26 Features from more than 60 features
    - Decoding the dataset and preparation of processed dataset
    - Exploratory Data Analysis (EDA)
    - Final Feature Selection using Chi-Squared Feature Selection
        - For Target Variable Label Encoding
        - For independent variables one-hot encoding with drop="first"

    - Preparation of final classification dataset for Machine Learning Model

#### 3. Encoding
    - For Target Variable Label Encoding
    - For independent variables one-hot encoding with drop="first"

#### 4. Model training and testing
    - 80% data for training and 20% data for testing
#### 5. Machine Learning Model Preparation using Decision tree
    - first classifier model is created for full depth of the tree
#### 6. Model Evaluation and Checking for Overfitting/Underfitting Conditions
    - Checking difference between training and testing accuracy
#### 7. Hyperparameter Tuning
    - Changing hyperparameters of decision tree like max_depth, max_sample_split, max_sample_leaf, criterion
#### 8. Performance Evaluation from Classification Report
    - Sensitivity, Specificity, AUC, Recall, Precision, f1 all are evaluated
#### 9. Random Oversampling for Target Variables Minority class to mitigate class imbalances
    - Oversampling ratio = 1 : 3X, Majority Class (1184 records) and Minority Class (314 records= X) initially
#### 10. Repeat the process from step- 4 untill good model performance

## Acknowledgement

1. M. F. Labib, A. S. Rifat, M. M. Hossain, A. K. Das and F.
Nawrine, “Road Accident Analysis and Prediction of Accident
Severity by Using Machine Learning in Bangladesh,” 2019 7th
International Conference on Smart Computing Communications (ICSCC), Sarawak, Malaysia, Malaysia, 2019, pp. 1-5,
doi: 10.1109/ICSCC.2019.8843640

2. M. M. L. Elahi, R. Yasir, M. A. Syrus, M. S. Q. Z. Nine, I. Hossain
and N. Ahmed, "Computer vision based road traffic accident and
anomaly detection in the context of Bangladesh," 2014 International
Conference on Informatics, Electronics & Vision (ICIEV), Dhaka,
2014, pp. 1-6.

3. M. S. Satu, S. Ahamed, F. Hossain, T. Akter and D. M. Farid, "Mining
traffic accident data of N5 national highway in Bangladesh employing
decision trees," 2017 IEEE Region 10 Humanitarian Technology
Conference (R10-HTC), Dhaka, 2017, pp. 722-725.

The authors read these papers and wanted to expand the literature for more recent datasets. These papers used datasets from 2001 to 2015, and the authors in this model used datasets from 2017-2020. Also, these papers give limited information about class Imbalances, which is very common in classifier problems like accident severity in Bangladesh. Also, in these papers, for decision trees, the authors did not provide all of the metrics for different classes of the target variable, which may lead to poor performance in minority classes even though the accuracy is high.


## Authors

- [Borhan Uddin Rabbai](https://www.github.com/borhanmukto)
- [B.M Tazbiul Anik](https://www.github.com/tazbiulanik)


## Datasets

- [Accident Research Institute, Bangladesh University of Engineering and Technology (BUET)](https://ari.buet.ac.bd/)
## Authors Contribution


Borhan Uddin Rabbani- Coding, Model Preparation, Literature Study

B.M Tazbiul Anik- ML Model Review
