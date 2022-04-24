# Credit_Risk_Analysis
Module 17 challenge

## Analysis Overview
In this project, we use Python to build and evaluate several machine learning models to predict credit risk.\
We adopted the following procedure:
- oversample the data using the **RandomOverSampler** and **SMOTE** algorithms.
- Undersample the data using the **ClusterCentroids** algorithm.
- Use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm.
- Compare two machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**.

We will evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Resources
- Data Source: LoanStats_2019Q1.csv(Not provided due to file being too Large to upload)
- Software: Python 3.7.9scikit-learn,[scikit-learn](https://scikit-learn.org/stable/install.html), Jupyter Notebook 6.0.3

## Results (Balanced Accuracy Scores, Confusion Matrixes and Imbalanced Classification Reports)
- ### OverSampled
  - Naive Random Oversampling results:
  - SMOTE Oversampling results:

- ### UnderSampled
  - UnderSampling via Cluster Centroids Algorithm:

- ### Combination (Over and Under) Sampling
  - SMOTEENN algorithm
