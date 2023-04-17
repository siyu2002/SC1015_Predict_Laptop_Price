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

2) Bi-variate analysis
<br /> a. Joint plot
  
3) Multi-variate analysis
<br /> a. Correlation coefficient between numeric variables
<br /> b. Boxplot with catergorical variables
  
<br />From the EDA, we have found out numeric variables **SSD and ScreenResolution** has the highest positive correlation with Laptop price.
<br />Also, from the Boxplot, Catergorial variables **Company, RAM, CPU_company and CPU_model** are the best predictors for Laptop price.

# Data Processing for ML Analysis
1. Creating dummy variables from existing variables
2. Removing variables that are not useful for ML analysis

Input for the ML model: SSD, ScreenResolution, Company, RAM, CPU_company and CPU_model
<br />Output: Laptop Price in Euros

# Machine Learning Tools
We have utilised a form of supervised learning: 
<br />Regression
<br />
<br />We have utilised Regression as we are predicting the laptop price which is continuous (non-discrete) numeric variable.

<br />
**First, the dataset is split into Test and Train data sets, with a split of 0.3 (30% test and 70% train)
<br />Then, Different Regression Models are fitted into the train dataset.
<br />The test dataset is used to determine the accuracy of the prediction.**

<br />We have tried 3 different regression methods in order to find out which method is the most effective in creating the best regression model.

<br />1) Decision Tree Regression

Hence, (2) Random Forest Regression and (3) Random Forest with Randomized Search CV Regression were used to further improve the accuracy of (1) Decision Tree Regression

<br />2) Random Forest Regression


<br />3) Random Forest with Randomized Search CV Regression

<br />Based on Randomized Search Cross Validation, 
The best hyperparameters determined are as follows:
 | hyperparameters | values |
 |-----------------|--------|
 | 'bootstrap' | True  |
 | 'max_depth'  | 110 |
 | 'max_features'  | 'sqrt'  |
 | 'min_samples_leaf' |  1  |
 | 'min_samples_split' | 2 |
 | 'n_estimators' |  450  |







# Conclusion
In order to determine the most effective model, we use 2 performance metrics, Mean Square Error (MSE) and Accuracy of the model.
 | |  Decision Tree Regressor | Random Forest Regressor | Random Forest Regressor with Randomized Search CV |
 |-----------------|--------|------------------------|---------------------------|
 | MSE | 405.54 |  389.48  | 381.32  |
 | Score  |  0.6703  | 0.6959  | 0.7085  |



<br />We have found out that Random Forest with Randomized Search CV produces the best laptop price prediction model.
<br />The model has the best Regressor accuracy score of 0.7085 and the least Mean Squared Error (MSE) of 381.32
<br />

**Rooms for Improvement for model:**
<br />Random Forests can be computationally expensive, and hence this model will not work as well if the dataset is large or has many features. The randomized CV search can further increase the computational complexity of the algorithm.
<br />Bias towards categorical variables: Random Forests can be biased towards categorical variables with many categories. This is because the algorithm tends to split such variables into smaller groups, which can lead to overfitting.

### Team Members
Edmund
<br /> Joshua
<br /> Siyu
<br />
