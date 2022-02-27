# Credit Risk Problem

LendingClub, a peer-to-peer lending services company, has assigned the task of analyzing a credit card dataset to determine which models can be best employed to train for evaluating credit risk. Our job is to use the following libraries to build and evaluate models using resampling techniques:
- imbalanced-learn
- scikit-learn

## Overview

### Data Cleaning
First step in the process of analyzing the dataset was to load and clean the 'Loan Stats.csv' file provided:
1. Read Raw CSV file to Dataframe
2. Add Column Header Names
3. Drop Null columns and rows
4. Convert interest rate % to float
5. Convert the target column values to low_risk and high_risk

### Target of Supervised Learning
- Target: Loan Status (High or Low Risk)
- Features: Other Columns (Ex. Loan Amount, Interest Rate, Annual Income, Debt to Income Ratio, etc.)

## Use Resampling Models to Predict Credit Risk

Compare various ML algorithms to determine which results in the best performance. The following algorithms were used for the credit card data analysis:

| Library | Oversample | Undersample | Combination | 
|:---|:---|:---|:---|
| Imbalanced Learn | [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) <br> [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)| [Cluster Centroids](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html) | [SMOTEEN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)

Follow the same 5 steps for each algorithm (using random_state = 1):
1. View the count of the target classes using Counter from the collections library.
2. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.

### Algorithm Results
| Algorithm | Sampling Type | Balanced Accuracy Score | Confusion Matrix |
|:--|:--|:--|:--|
| Naive Random | Oversampling | 0.64472 | [[ 71, 30],[ 7073, 10031]] |
| SMOTE | Oversampling | 0.66231 | [[ 64, 37],[ 5286, 11818]] | 
| Cluster Centroids| Undersampling | 0.66231 | [[ 70, 31], [10341, 6763]] |
| SMOTEEN | Combination | 0.54424 | [[ 82, 19], [7593, 9511]] |

### Classification Report Results

<strong>

  1. Naive Random
![Naive Random](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/Naive%20Random%20Oversampling.png)
  2. SMOTE
![SMOTE](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/SMOTE%20Oversampling.png)
  3. Cluster Centroids
![Cluster Centroids](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/Cluster%20Centroids%20Undersampling.png)
  4. SMOTEEN
 ![SMOTEEN](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/SMOTEEN.png) 
</strong>

## Ensemble Classifiers to Predict Credit Risk
The next step is to compare different machine learning models to reduce bias:
| Library | ML Model | 
|:---|:---|
| Scikit-Learn | [BalancedRandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
| Imbalanced Learn | [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html)

For each ensemble learner algorithm, complete the following steps (using random_state = 1):
1. Train the model using the training data.
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Print the confusion matrix from sklearn.metrics.
4. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

| Algorithm | Balanced Accuracy Score | Confusion Matrix |
|:--|:--|:--|
| Random Forest | 0.67288 | [[ 63, 38],[4755, 12349]] |
| Easy Ensemble Classifier | 0.91409 | [[93, 8],[ 1584, 15520]] |

### Classification Report Results

<strong>

  1. Random Forest
![Random Forest](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/Random%20Forest.png)
  2. Easy Ensemble Classifier
![Easy Ensemble](https://github.com/srfassihi/Credit_Risk_Analysis/blob/98ed52ba38fdc9428296c1d2d8d9d267fef4d47b/images/Easy%20Ensemble%20Classifier.png)  
</strong>

### Top 10 Features Sorted by Importance (Random Forest)
1. Last Payment Amount (11.39)
2. Total Payment (7.82)
3. Total Received Principal (5.62)
4. Total Payment Inv (3.54)
5. Total Received Interest (2.97)
6. Total High Credit Limit (2.48)
7. Months since Recent (2.25)
8. Debt to Income (2.08)
9. Interest Rate (1.94)
10. BC Util (1.93)

## Interpreting the Results
By reviewing all the model results, the *Easy Ensemble Classifier* model gave the best results in terms of accuracy and precision for both High Risk and Low Risk applicants. The sensitivity rate was highest amongst the other models. Therefore we recommend the use of this model for determining which loan applicants fall under the high or low risk category, based on the Loan dataset provided. 
