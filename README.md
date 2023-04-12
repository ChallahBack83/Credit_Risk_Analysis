# Credit_Risk_Analysis

## Overview

The purpose of this analysis is to evaluate credit card risk using supervised machine learning.  Using Python's imbalanced-learn and scikit-learn libraries, I trained and evaluated various models on data with unbalanced classes using resampling. First, I oversampled the data using <b>RandomeOverSampler</b> and <b>SMOTE</b> algorithms. Then I undersampled using <b>ClusterCentroids</b> followed by a combination approach using <b>SMOTEEN</b>. Finally, I compared two new machine learning models used to reduce bias: <b>BalancedRandomForestClassifier</b> and <b>EasyEnsembleClassifier</b>.

## Results

In the world of credit card risk, the data is very imbalanced since there are more instances of low-risk versus high-risk. In this data set, our low-risk count is <b>68,470</b> versus a high-risk count of <b>347</b>. In order to account for this difference, we can use various machine learning models to help predict more accurately.  However, accuracy does not necessarily provide the best fit for our needs. Using each method, I calculated the balanced accuracy scores, precision scores, and recall (or sensitivity) scores for each model so we can compare them all and choose the best model to accuractely predict high-risk credit situations.

- #### Naive Random Oversampling using ```RandomOverSampler```:
    - Accuracy Score:  64.64%
    - Precision Score: 99.00% 
    - Recall Score:    58.00%
    - Summary: This algorithm creates equal classes before training by increasing the minority class. Though precision is high, both accuracy and recall are low.  This may not be the best model to use.
    
  ![ros_scores]()

- #### Oversampling using ```SMOTE```:
    - Accuracy Score: 65.86%
    - Precision Score: 99.00%
    - Recall Score: 68.00%
    - Summary: Once again, this algorithm creates equal classes by increasing the minority class, and once again, precision is high.  However accuracy and recall, though slightly improved, are still low.
    
  ![smote_scores]()

- #### Undersampling using ```ClusterCentroids```:
    - Accuracy Score: 54.47%
    - Precision Score: 99.00%
    - Recall Score: 40.00%
    - Summary: This algorithm creates equal classes by decreasing the majority class to create equal classes.  Here we see a drop in both accuracy and recall from the previous models.
    
  ![cc_scores]()
  
- #### Combination (over & under) Sampling using ```SMOTEEN```:
    - Accuracy Score: 66.65%
    - Precision Score: 99.00%
    - Recall Score: 60.00%
    - Summary: By combining over and under sampling, the classes are not equal but they are closer in count.  Precision remains the same, but accuracy and recall are both improved.  However, the SMOTE method results are still higher.
    
  ![smoteen_scores]()

- #### Ensemble Learning using ```BalancedRandomForestClassifier```:
    - Accuracy Score: 78.86%
    - Precision Score: 99.00%
    - Recall Score: 87.00%
    - Summary: Ensemble Learning aggregates several simpler algorithms and takes the consensus. This BalancedRandomeForestClassifier results in a MUCH higher accuracy and recall than any of our previous methods. 
  
  ![brfc_scores]()
  
- F1 (balance): 93%
- high-risk recall: 70%

- #### Ensemble Learning using ```EasyEnsembleClassifier```:
    - Accuracy Score: 93.17%
    - Precision Score: 99.00%
    - Recall Score: 94.00%
    - Summary: EasyEnsembleClassifier is another ensemble learner. Though our last one was vastly different than the previous models, this one's accuracy and recall are the highest of all six models.
  
  ![eec_scores]()
  
- F1 (balance): 97%
- high_risk recall: 92%

## Summary

Comparing all six models, it is easy to see that the first four can be ruled out since their accuracy and recall scores all are well under 70%, meaning they are moderately okay at best. The ensemble learners both provide much more accurate results. However, a high accuracy score, just like precision, does not necessarily mean it is the best model since we want to find as many high-risk instances as possible without mislabeling customers risks and impacting their credit scores. So we must take sensitivity (recall) into account alongside accuracy.  Out of our ensemble learners, the ```EasyEnsembleClassifier``` proves the best model since accuracy, precision, and recall are over 90%.  Looking at the imbalanced classification report for this method also shows that the recall of the individual classes are well balanced with our minority class (high-risk) having a recall of 92% and low-risk having a recall of 94%.  These are the best recall scores out of all six of our models.
