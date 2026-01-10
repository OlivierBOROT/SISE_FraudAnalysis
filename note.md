
# Methodology

## Data cleaning
- Brief analysis of the data; checking for anomalies, NaN Values
- Transformation into correct data types
- Addition of the 'weekday' column from the date

## Data checking
- Brief analysis of the distribution:
    - Verifiance CP2 may be an important variable to where we observe a different distribution for frauds
    - The "montant" column show a higher mean for Frauds compare to normal

## Models selection

### Problematics
- Less than 1% of Frauds observed in the data
- Large dataset (> 4 Millions obesrvations for around 20 variables)

We do need fast computing models, able to learn complex patterns to reach the < 1% of the label to detect. The low bias model we will develop may have a high risk of overfitting.
To avoid this problematic, we thought of the following strategy:

- Trying multiple complex models and test the F1 score on the positive label.
    - Use a weighing strategy to punish more bad classifications on our positive class.
    
- Trying to improve the individual predictions by stacking the models
    - Using simple linear model to predict over the stacked predictions    

### Model pre-selection

- Boosting models to push the learning process to the misclassified labels, taking advantage of the unbalanced weighing over the errors.
    - XGBoost (HistBoostClassifier) 
    - AdaBoost

- RandomForest: To try to learn complex models into detecting the positive labels, while trying to reduce the variance

- Deplearning MLP algortihm: able to learn complex patterns; if the variance may be to high, we hope that the stacking process will compensate.

- SVM with an approximation of the Gaussian Kernel to learn complex pattern using landmarks to speed up the learning process -> Still to slow on the compelte dataframe.

- Mix with under sampling methods (Tomek links or sampling method)

### Combine the models:

- Each model make a prediction on the test set.
- Find a simpler model to predict on all the other predictions. (like a linear regression)