# Introduction
Our project objective is to create an effective laptop pricing model based on the computer specifications and current demand and supply of the market, in order for laptop manufacturers to have a competitive pricing for their product.
We are using a mixed structured dataset of numeric and categorical datatype from Kaggle, and We are going to peform Machine Learning Algorithms on the dataset in order to predict the laptop price using multiple variables of categorial and numeric datatypes.
  Refer: Raw data_description

# Data preprocessing (Cleaning and Preparation of dataset) 
1) Conversion of Datatypes (from categorial to numeric)
2) Splicing
3) Removing outliers
4) Dropping variables not useful for project objective

# Exploratory Data Analysis
1) Uni-variate analysis
  a) Histogram
  b) Histogram with kernel density estimate (KDE)
  b) Boxplot
  c) Violin plot
  
2) Multi-variate analysis
  a) Correlation coefficient between numeric variables
  b) Boxplot with catergorical variables
  
From the EDA, we have found out 

# Data processing for ML Analysis
1) creating dummy variables from existing variables
2) Removing variables that are not useful for ML analysis

# ***Machine Learning Tools utilised:
Regression
We have utilised Regression as we are predicting the laptop price which is continuous (non-discrete) numeric variable.
-The Machine Learning algorithm here is provided with a small training dataset to work with, which is a smaller part of the bigger dataset.
-It serves to give the algorithm an idea of the problem, solution, and various data points to be dealt with.
-The training dataset here is also very similar to the final dataset in its characteristics and offers the algorithm with the labeled parameters required for the problem.
-The Machine Learning algorithm then finds relationships between the given parameters, establishing a cause and effect relationship between the variables in the dataset.

We have tried 3 different regression methods in order to find out which method is the most effective in creating the best regression model.

1) Decision Tree


2) Random Forest


3) Random Forest with CV

(2) Random Forest and (3) Random Forest with CV were used to further improve the accuracy of (1) Decision Tree

In order to determine the most effective model, we use 2 performance metrics, Mean Square Error (MSE) and Accuracy of the model.


# Conclusion
We have found out that Random Forest with Randomized Search CV produces the best laptop price prediction model.
The model has the best Regressor accuracy score of 0.7085 and the least Mean Squared Error (MSE) of 381.32

#
## Team Members
#### Edmund
#### Joshua
#### Siyu


# 
============================================================================================================================================================  
Checkpoint 1. Introduction Detailed introduction to your project objective, e.g. what problem you are going to solve based on what dataset.

Briefly review the significance of your topic, e.g. any potential applications of your project.

Summarize the organization of your report, e.g. in Section xx we do xx.

Checkpoint 2. Data Preprocessing Detailed description on your dataset via statistics and visualization.

Detailed statements on why and how you perform data preprocessing, e.g. data cleaning, normalization, transformation, data augmentation.

Checkpoint 3. Methodology Explain the reason for choosing your machine learning model.

Detailed & formal introduction to your model. You must provide the formulation or diagram of the model you use thoroughly. Clarify the input and output of your model.

Clarify how you train and inference based on the model you choose.

Clarify the choice of hyperparameters of your model.

Checkpoint 4. Experiments Detailed introduction to the performance metrics you use for experiments.

Briefly introduce which baselines you are comparing with, e.g. you compare your model against a random guessing, a decision tree, a linear model, etc. This part is compulsory.

Detailed model selection and comparison: Is your model fitting well compared to your baselines? Discuss about underfitting/overfitting if any. Important

Which configuration (hyperparameter choices) performs the best? What numerical results lead to these conclusions? Your conclusion is held in what sense? The analysis of this part (not the performance alone) is the most important.

Checkpoint 5. Conclusion Briefly summarize your findings in Experiments.

The limitation of your current model. How you can improve your model.
