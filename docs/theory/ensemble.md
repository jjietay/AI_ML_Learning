# Ensemble of Models

- combining multiple models to make a prediction

## 2 Methods
1) Majority
- choose most common prediction made

2) Summation
- sum the probabilties of option A and B seperately
- the higher sum of either option A or B will be chosen to be the most probably result
- performing average will still divide everything by the same amount, so we avoid doing an extra unecessary steps

# Bootstrap AGGregatING (BAGGING)

## Introduction

- to reduce overfitting (chances of model memorising dataset)
- used when we wanna improve stability or performance of ML algorithms

## How it works

- bagging generates a new dataset for each model you train
- it randomly samples (with replacement) from your original data
- for every model you train, bagging will take a subset of data and create a new dataset
- selection happens randomly with replacement, meaning new dataset could contain duplicates

# Boosting

## Introduction

- increase performance on harder or less common examples
- but slower to train than BAGGING

## How it works

- iteratively creating new models by training the models to better learn the data that the previous model misclassified
- boosting starts by training a model on the original dataset normally
- for every example that was misclassified, they increased the weights
- for those correctly classified, they decreased the weights
- weights are ways of mathematically expressing importance of a training sample to out model
- higher weight --> the more the model tries to learn to predict that example correctly
- $prediction = w_1out_1 + w_2out_2 + w_3out_3$
- weights here is the accuracy or error that the model for that particular dataset its trained on