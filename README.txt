# Customer Purchase Prediction using Decision Tree
#Demo of a Supervised Learning System Application

This project demonstrates a supervised learning system that predicts whether a customer will purchase a product based on customer information and review data.  

---
## Overview
The dataset contains customer information such as Age, Gender, Review, Education, and Purchased status (Yes/No).  
A Decision Tree Classifier from the scikit-learn library was used to build and evaluate the supervised model.

* AI Task: Binary Classification  
* Goal: Predict if a customer will purchase a product.  
* Algorithm Used: Decision Tree Classifier (Supervised Learning)
---
## Files
* Customer_Review.csv:  Input dataset from Kaggle
* purchase_prediction_decision_tree.py: Python source code for data loading, training, testing, and prediction
* README.md: Project documentation
---
## Requirements
* Install the required Python libraries:
    pip install pandas scikit-learn seaborn matplotlib
---
## How to Run

* Place both Customer_Review.csv and purchase_prediction_decision_tree.py in the same directory.
* Run the Python script:
    python purchase_prediction_decision_tree.py
---
## Output Details
* Print the feature columns and target variable.
* Show training/testing dataset sizes.
* Train a Decision Tree Classifier.
* Display model metrics:
    * Accuracy
    * Precision
    * Recall
    * F1-score
* Plot a Confusion Matrix using Seaborn.
* Predict outcomes for two new customer samples.

* Example output:
Feature Columns: ['Age', 'Gender', 'Review', 'Education']
Target Column: Purchased
Training set size: (80, 4)
Testing set size: (20, 4)
Model trained with hyperparameter tuning.
Feature importances: [0.83278281 0.         0.11258525 0.05463194]
Model Accuracy: 55.00%
Classification Report:
              precision    recall  f1-score   support

           0       0.60      0.55      0.57        11
           1       0.50      0.56      0.53         9

    accuracy                           0.55        20
   macro avg       0.55      0.55      0.55        20
weighted avg       0.55      0.55      0.55        20


Future Predictions:
   Age  Gender  Review  Education Predicted Purchased
0   45       0       1          0                  No
1   25       1       2          2                 Yes


---
Interpretation
* Accuracy: Around 55%, due to small dataset size and limited features.
* Feature Importance: Age and Review influence most purchase decisions.
* Confusion Matrix: Visualizes how many “Yes” and “No” purchases were correctly predicted.
* Predictions: The script predicts purchase likelihood for new customer cases.

Example Predictions
 Age  Gender  Review  Education Predicted Purchased
0   45       0       1          0                  No
1   25       1       2          2                 Yes