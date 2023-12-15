Please view the code through my [Kaggle Notebook](https://www.kaggle.com/code/osbornepereira/ehc-classification) to look at the plotly graphs used during the analyses. 

## Objective:
The task embedded to the dataset is classification prediction.

## Dataset:
This a Kaggle dataset which contains Electronic Health Records collected from a private hospital in Indonesia.
It contains the patients laboratory test results used to determine whether the next patient would require in care or out care treatment.

## Methods:
We first test 4 classification models:
1. Logistic Regression
2. Support Vector Machine
3. K-Nearest Neighbor
4. Random Forest

* Builiding on the model predictions, we then optimize using GridSearchCV to find the best suited hyperparameters.

* We use the accuracy, precision, recall, F2 scores, and confusion matrix to assess the classification output of the best performing models.

## Disclaimer:
This data has been obtained from the 'Patient Treatment Classification' dataset avialsble freely on Kaggle.
