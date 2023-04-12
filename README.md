# Credit_Risk_Analysis

## Overview

The purpose of this analysis is to evaluate credit card risk using supervised machine learning.  Using Python's imbalanced-learn and scikit-learn libraries, I trained and evaluated various models on data with unbalanced classes using resampling. First, I oversampled the data using <b>RandomeOverSampler</b> and <b>SMOTE</b> algorithms. Then I undersampled using <b>ClusterCentroids</b> followed by a combination approach using <b>SMOTEEN</b>. Finally, I compared two new machine learning models used to reduce bias: <b>BalancedRandomForestClassifier</b> and <b>EasyEnsembleClassifier</b>.

## Results

In the world of credit card risk, the data is very imbalanced since there are more instances of low-risk versus high-risk. In this data set, our low-risk count is <b>68,470</b> versus a high-risk count of <b>347</b>. In order to account for this difference, we can use various machine learning models to help predict more accurately.  However, accuracy does not necessarily provide the best fit for our needs. Using each method, I calculated the balanced accuracy scores, precision scores, and recall (or sensitivity) scores for each model so we can compare them all and choose the best model to accuractely predict high-risk credit situations.

- #### Naive Random Oversampling using ```RandomOverSampler```:
    - Accuracy Score:  64.64%
    - Precision Score: 99.00% 
    - Recall Score:    58.00%
    
  ![ros_scores]()

- #### Oversampling using ```SMOTE```:
    - Accuracy Score: 65.86%
    - Precision Score: 99.00%
    - Recall Score: 68.00%
    
  ![smote_scores]()

- #### Undersampling using ```ClusterCentroids```:
    - Accuracy Score: 54.47%
    - Precision Score: 99.00%
    - Recall Score: 40.00%
    
  ![cc_scores]()
  
- #### Combination (over & under) Sampling using ```SMOTEEN```:
    - Accuracy Score: 66.65%
    - Precision Score: 99.00%
    - Recall Score: 60.00%
    
  ![smoteen_scores]()

- #### Ensemble Learning using ```BalancedRandomForestClassifier```:
    - Accuracy Score: 78.86%
    - Precision Score: 99.00%
    - Recall Score: 87.00%
  
  ![brfc_scores]()
  
- F1 (balance): 

- #### Ensemble Learnign using ```EasyEnsembleClassifier```:
    - Accuracy Score: 93.17%
    - Precision Score: 99.00%
    - Recall Score: 94.00%
  
  ![eec_scores]()
  
- F1 (balance):

## Summary

summarize results of models, include recommendation on model to use, if any. if not any...justify reasoning

This means, a high accuracy score does not necessarily help us since predicting more low-risk instances does not improve the instances of finding high-risk, which if we rely on accuracy, we  accuracy or precision is most likely a more important 
