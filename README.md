# Credit_Risk_Analysis
Module 17 challenge

## Overview
&nbsp;&nbsp;&nbsp;&nbsp;In this project, we use Python to build and evaluate several machine learning models to predict credit risk.\
We adopted the following procedure:
- oversample the data using the **RandomOverSampler** and **SMOTE** algorithms.
- Undersample the data using the **ClusterCentroids** algorithm.
- Use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm.
- Compare two machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**.

We will evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Resources
- Data Source: LoanStats_2019Q1.csv(Not provided due to file being too Large to upload)
- Software: Python 3.7.9scikit-learn,[scikit-learn](https://scikit-learn.org/stable/install.html), Jupyter Notebook 6.0.3

## Results(RESAMPLING)
- ### OverSampled

  - Naive Random Oversampling results:We notice a `accuracy_score` roughly at 65%(65.16) however, our `classification_report_imbalanced` displays disparity when it comes to the precision and recall(Sensitivity) thus impacting our f1 score. Indicating that the model is more accurate then it claims to be. This is cause for concern as we are given a false sense of security that the model is appropriately prediciting the diference between High and Low Risk credit scores.
<p align="center">  
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Naive_Random_Oversampling.png" />
  <br>  </br>
</p>

  - SMOTE Oversampling results:SMOTE(synthetic minority oversampling technique) intends to generate similar datasets as `RandomOverSampler` however, it does so differently by implementing [K-Nearest Neighbor algorithm](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e). Essentially,<em>"a simple algorithm that stores all the available cases and classifies the new data or case based on a similarity measure."</em>  
&nbsp;&nbsp;&nbsp;&nbsp;This resampled data also produces a roughly 65%(64.62) however, due to SMOTE implementation it was able to produce a higher reacall score for "low_risk" than it did in Random OverSampling thus providing us a higher f1 score. Although both methods suffer from the flaws of OverSampling(relying on random generated data not part of original dataset), SMOTE was able to mitigate some of the bias when generating data from the minority class when addressing class imbalance in the dataset.
<p align="center">
  
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_SMOTE_OverSampling.png"/>
    <br>  </br>
</p>

- ### UnderSampled
  - UnderSampling via Cluster Centroids Algorithm: Here we will do the inverse of OverSampling, we will downscale the Large Class to match the minority class. In doing so we are working with 100% of real acquired data within the data set at the exchange of losing important information.  
&nbsp;&nbsp;&nbsp;&nbsp;Here we are seeing accuracy roughly at 65%(64.62) however, due to the nature of crunching down the classes we are seeing clear signs of bias. The recall percentages for High_Risk:69% and Low_Risk:40% inidicates that the minority class forced the dataset into contaiting mostly "High Risk Credit Scores," giving us a false sense of concern since we lost so much important info.
<p align="center">
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Cluster_Centroid_UnderSampling.png"/>
    <br>  </br>
</p>

- ### Combination (Over and Under) Sampling
  - SMOTEENN algorithm: SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms, a two step process:First, oversample the minority class with SMOTE then second, clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped. Essentially, we generate results but keep those with ONLY a single class identifying them thus ensuring we work with the benifts of SMOTE but without sacrificing important info from the original dataset.  
&nbsp;&nbsp;&nbsp;&nbsp; Here the accuracy score is roughly 54.5%(54.47), it creates the impression that a little over half of the dataset after being resampled is accurate. However, despite having a low recall value for low_risk, the f1 score(.73 or 73%) indicates the dataset to be more accurate than it apears to be. It appears this is providing the most desirable results but we must remember that SMOTEENN has its flaws that being, by oversampling the dataset with similarities in class measures but dropping datasets with more than one class identifier the dataset actaully becomes unbalanced again.
<p align="center">
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Combination_(Over%20and%20Under)_Sampling.png"/>
   <br>  </br>
</p>

## Results(ENSEMBLED)  

- ### Balanced Random Forest Classifier results: The balanced accuracy score is roughly 79%(78.77), upon further inspection we notice that the f1 score has drastically increased to `.95`. This method of grouping data will have similarities to "Random OverSampling" since we are overfitting the data to deal with class imbalance thus leaving us with a weak learning machine.
[Balanced Random Forest in python](https://www.linuxtut.com/en/0f6faf5629f6c563d36f/)
<p align="center">
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/1c91742c3fc45b934a0ac2f40a564985dedd895f/Images/ENSEMBLE_BalancedRandomForestClassifier.png"/>
   <br>  </br>
</p>

- ### Easy Ensemble AdaBoost Classifier:
<p align="center">
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/1c91742c3fc45b934a0ac2f40a564985dedd895f/Images/ENSEMBLE_EasyEnsembleClassifier_AdaBoosting.png"/>
   <br>  </br>
</p>

## Summary
&nbsp;&nbsp;&nbsp;&nbsp;In the first four models we undersampled, oversampled and did a combination of both to try and determine which model is best at predicting which loans are the highest risk. The next two models we resampled the data using ensemble classifiers to try and predict which which loans are high or low risk. In our first four models our accuracy score is not as high as the ensemble classifiers and the recall in the oversampling/undersampling/mixed models is low as well. Despte <b>SMOTE</b> and <b>SMOTEENN</b> showing that they can mitigate some of the imbalances of the class identifiers,thus leading to a more reliable F1 score, they still embody the flaws of both Over/Under-Sampling. Typically in your models you want a good balance of recall and precision,resulting in a higher F1 score to measure against accuracy, which is why I recommend the `EasyEnsembleClassifiers`(AdaBoosting) over any other models presented. It appears that the Easy Ensemble had the best balance of all the models because of it's high F1 score and good balance of precision and recall scores generated from the repeated process of removing similar errors through each iteration until they reach Zero(Ideal goal but not possible) or nearly negligible.
