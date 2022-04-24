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

## Results 
- ### OverSampled

  - Naive Random Oversampling results:We notice a `accuracy_score` roughly at 65% however, our `classification_report_imbalanced` displays disparity when it comes to the precision and recall(Sensitivity) thus impacting our f1 score. Indicating that the model is more accurate then it claims to be. This is cause for concern as we are given a false sense of security that the model is appropriately prediciting the diference between High and Low Risk credit scores.
<p align="center">  
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Naive_Random_Oversampling.png" />
  <br>  </br>
</p>

  - SMOTE Oversampling results:SMOTE(synthetic minority oversampling technique) intends to generate similar datasets as `Random OverSampling` however, it does so differently by implementing [K-Nearest Neighbor algorithm](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e). Essentially,<em>"a simple algorithm that stores all the available cases and classifies the new data or case based on a similarity measure."</em> 
<p align="center">
  
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_SMOTE_OverSampling.png"/>
    <br>  </br>
</p>

- ### UnderSampled
  - UnderSampling via Cluster Centroids Algorithm:
<p align="center">
  
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Cluster_Centroid_UnderSampling.png"/>
    <br>  </br>
</p>

- ### Combination (Over and Under) Sampling
  - SMOTEENN algorithm:
<p align="center">
   
<img src="https://github.com/KdotGhai/Credit_Risk_Analysis/blob/608df1ecfc22e18b050f9b106636858274850b45/Images/RESAMPLING_Combination_(Over%20and%20Under)_Sampling.png"/>
   <br>  </br>
</p>

## Summary
