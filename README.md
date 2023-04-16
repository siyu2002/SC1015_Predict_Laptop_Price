# Introduction
Our project objective is to create an effective laptop pricing model based on the computer specifications and current demand and supply of the market, in order for laptop manufacturers to have a competitive pricing for their product. <br />
<br />
We are using a mixed structured dataset of numeric and categorical datatype from Kaggle, and We are going to peform Machine Learning Algorithms on the dataset in order to predict the laptop price using multiple variables of categorial and numeric datatypes. <br />
<br />    Refer: [Raw data_description](https://github.com/siyu2002/SC1015_Predict_Laptop_Price/blob/main/Data%20Description/raw%20data_description) <br />

# Data Pre-processing (Cleaning and Preparation of Dataset) 
1) Conversion of Datatypes (from categorial to numeric)
2) Splicing
3) Removing outliers
4) Dropping variables not useful for project objective

<br /> Refer: [Clean data_description](https://github.com/siyu2002/SC1015_Predict_Laptop_Price/blob/main/Data%20Description/Clean%20data_description) <br /> 

# Exploratory Data Analysis
**Visualisation**
1) Uni-variate analysis
<br /> a. Histogram
<br /> b. Histogram with kernel density estimate (KDE)
<br /> c. Boxplot
<br /> d. Violin plot
  
2) Multi-variate analysis
<br /> a. Correlation coefficient between numeric variables
<br /> b. Boxplot with catergorical variables
  
<br />From the EDA, we have found out numeric variables **SSD and ScreenResolution** has the highest positive correlation with Laptop price.
<br />Also, from the Boxplot, Catergorial variables **Company, RAM, CPU_company and CPU_model** are the best predictors for Laptop price.

# Data Processing for ML Analysis
1. Creating dummy variables from existing variables
2. Removing variables that are not useful for ML analysis

Input for the ML model: SSD, ScreenResolution, Company, RAM, CPU_company and CPU_model
<br />Output: Laptop Price in Euros

# Machine Learning Tools utilised:
We have utilised a form of supervised learning: 
<br />Regression
<br />
<br />We have utilised Regression as we are predicting the laptop price which is continuous (non-discrete) numeric variable.
<br />-The Machine Learning algorithm here is provided with a small training dataset to work with, which is a smaller part of the bigger dataset.
<br />-It serves to give the algorithm an idea of the problem, solution, and various data points to be dealt with.
<br />-The training dataset here is also very similar to the final dataset in its characteristics and offers the algorithm with the labeled parameters required for the problem.
<br />-The Machine Learning algorithm then finds relationships between the given parameters, establishing a cause and effect relationship between the variables in the dataset.
<br />
<br />
**First, the dataset is split into Test and Train data sets, with a split of 0.3 (30% test and 70% train)
<br />Then, Different Regression Models are fitted into the train dataset.
<br />The test dataset is used to determine the accuracy of the prediction.**

<br />We have tried 3 different regression methods in order to find out which method is the most effective in creating the best regression model.

<br />1) Decision Tree Regression
>Decision Tree was decided as we are using a combination of numeric and categorical variables to predict a continuous numeric datatype Laptop Price.
><br />Decision Tree finds the best split at a certain feature, like SDD, at a certain value.
><br />Each Decision Node is recursively split into leaf nodes using the features(variables) and values, creating different classes.
><br />By traversing the tree, we are then able to predict the Laptop price using the test data, and the predicted value is compared to the actual value to determine the accuracy of the model.
><br />However, Decision Tree often leads to data fragmentation, and overfitting of the model.
><br />This occurs when the training data is recursively split until each leaf node is pure and each leaf node have a 100% accuracy.

Hence, (2) Random Forest Regression and (3) Random Forest with Randomized Search CV Regression were used to further improve the accuracy of (1) Decision Tree Regression

<br />2) Random Forest Regression

>Random Forest is fast and robust, and fixes the overfitting problem by Decision Tree.
><br />It works by generating multiple decision trees and merges the output of multiple Decision Trees to generate the final output.
><br />

<br />3) Random Forest with Randomized Search CV Regression

> In order to further optimise Random Forest Model, hyperparameters were tuned before the train dataset is used for training of the model.
> <br />In the case of Random Forest, The hyperparameters are number of features used for spliting considered for each tree and number of decision trees generated.
> <br />Then, Cross Validation is used to account for the overfitting of model on the train dataset, and determine the best hyperparameters for Random Forest tree.
> <br />The train set is split into K number of subsets(folds)
> <br />K-1 folds are trained and fitted, and evaluated on the remaining fold.
> <br />The training dataset is iteratively fitted K times, evaluating each fold with respective of the remaining trained fold.
> <br />The average performance is obtained from all the evaluations and the final validation metrics for the model were determined.
> <br />For each hyperparameter combination, the whole CV process is repeated to find the best hyperparameters for the tree.

<br />Based on Randomized Search Cross Validation, 
The best hyperparameters determined are as follows:
 >| hyperparameters | values |
 >|-----------------|--------|
 >| 'bootstrap' | True  |
 >| 'max_depth'  | 110 |
 >| 'max_features'  | 'sqrt'  |
 >| 'min_samples_leaf' |  1  |
 >| 'min_samples_split' | 2 |
 >| 'n_estimators' |  450  |





# Conclusion
In order to determine the most effective model, we use 2 performance metrics, Mean Square Error (MSE) and Accuracy of the model.
<br />We have found out that Random Forest with Randomized Search CV produces the best laptop price prediction model.
<br />The model has the best Regressor accuracy score of 0.7085 and the least Mean Squared Error (MSE) of 381.32


### Team Members
Edmund
<br /> Joshua
<br /> Siyu
<br />

============================================================================================================================================================  
Checkpoint 1. Introduction Detailed introduction to your project objective, e.g. what problem you are going to solve based on what dataset.

Briefly review the significance of your topic, e.g. any potential applications of your project.

Summarize the organization of your report, e.g. in Section xx we do xx.

Checkpoint 2. Data Preprocessing Detailed description on your dataset via statistics and visualization.

Detailed statements on why and how you perform data preprocessing, e.g. data cleaning, normalization, transformation, data augmentation.

**Checkpoint 3. Methodology Explain the reason for choosing your machine learning model.

**Detailed & formal introduction to your model. You must provide the formulation or diagram of the model you use thoroughly. Clarify the input and output of your model.

**Clarify how you train and inference based on the model you choose.

**Clarify the choice of hyperparameters of your model.

**Checkpoint 4. Experiments Detailed introduction to the performance metrics you use for experiments.

**Briefly introduce which baselines you are comparing with, e.g. you compare your model against a random guessing, a decision tree, a linear model, etc. This part is compulsory.

**Detailed model selection and comparison: Is your model fitting well compared to your baselines? Discuss about underfitting/overfitting if any. Important

**Which configuration (hyperparameter choices) performs the best? What numerical results lead to these conclusions? Your conclusion is held in what sense? The analysis of this part (not the performance alone) is the most important.

Checkpoint 5. Conclusion Briefly summarize your findings in Experiments.

The limitation of your current model. How you can improve your model.
