# Loan-Payback-DL-Project

## The Data

We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

## Our Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we will build a model that can predict whether or nor a borrower will pay back their loan. This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.

The "loan_status" column contains our label.

## About the Project

The project first runs exploratory data analysis on the dataset. It analyses different relationships between the columns, and feature engineers a set of data actually relevant to the loan-repayment status.

Then the data is pre-processed, i.e missing data points are accomodated either by regression or by removal, redundant columns are removed, and the dataframe is scaled and normalized using ```MinMaxScaler``` and ```One-Hot Encoding```.

Then we train the neural-network model. We choose a ```Sequential``` model and add 4 ```Dense``` layers to it, 2 input and output layers and 2 hidden layers. We used ```relu``` activation function. ReLU stands for Rectified Linear Unit. The ReLU activation function is defined as f(x)=max(0,x), meaning it returns zero for negative input values and passes positive input values directly. ReLU is known for its simplicity and efficiency in training deep neural networks.

The output layer consisted of 1 neuron only, with ```sigmoid``` activation function, also known as the logistic function, as this was a binary classification problem.

The model was compiled using ```adam``` optimizer and ```binary_crossentropy``` loss function.

Early stopping callback and Dropout layers were used to prevent overfitting to the training data.

## Model Evaluation

The model returned excellent results. It returned the following classification report:

```shell

               precision    recall  f1-score   support

           0       0.99      0.43      0.60     15556
           1       0.88      1.00      0.93     63460

    accuracy                           0.89     79016
   macro avg       0.93      0.72      0.77     79016
weighted avg       0.90      0.89      0.87     79016

                

```

## End!
