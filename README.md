# Credit Risk Problem

LendingClub, a peer-to-peer lending services company, has assigned the task of analyzing a credit card dataset to determine which models can be best employed to train for evaluating credit risk. Our job is to use the following libraries to build and evaluate models using resampling techniques:
- imbalanced-learn
- scikit-learn

## Overview
The following algorithms were used for the credit card data analysis:

| Library | Oversample | Undersample | Combination | 
|:---|:---|:---|:---|
| Imbalanced Learn | [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) <br> [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)| [Cluster Centroids](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html) | [SMOTEEN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)

The next step is to compare different machine learning models to reduce bias:
| Library | ML Model | 
|:---|:---|
| Scikit-Learn | [BalancedRandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
| Imbalanced Learn | [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html)

The results from each of the models is as follows:

## Results

### Resampling
First step in the process of analyzing the dataset was to load and clean the 'Loan Stats.csv' file provided:
1. Read Raw CSV file to Dataframe
2. Add Column Header Names
3. Drop Null columns and rows
4. 
