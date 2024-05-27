# **Predicting Future Customers of a Bank**
**Author:** Anne Marie Bogar<br/>
**Active Project Dates:** May 6-20, 2024<br/>

## Summary
A Portuguese banking company conducted a marketing campaign (through phone calls) to gain customers. 
Given the data collected for each client (including demographic information and information on the campaign methods), 
three different machine learning models were implemented to predict whether or not a potential client would be likely to become a customer at the bank. 
The models implemented were: Logistic Regression, Decision Tree and Random Forest.

## Data
The features in the [dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) are as follows:
1. **age:** _numerical_
2. **job:** _categorical by job type_ - admin., unknown, unemployed, management, housemaid, entrepreneur, student, blue-collar, self-employed, retired, technician, services
3. **marital:** _categorical_ - married, single, divorced
4. **education:** _categorical_ - primary, secondary, tertiary, unknown
5. **default:** _categorical_ - whether or not they have credit in default
6. **balance:** _numerical_ - average yearly bank balance in euros
7. **housing:** _categorical_ - whether or not they have a housing loan
8. **loan:** _categorical_ - whether or not they have a personal loan
9. **contact:** _categorical_ by how they were contacted - telephone, cellular, unknown
10. **day:** _numerical_ - the day of the month they were last contacted
11. **month:** _categorical_ - the month they were last contacted
12. **duration:** _numerical_ - last contact duration, in seconds
13. **campaign:** _numerical_ - number of contacts performed during this campaign
14. **pdays:** _numerical_ - number of days since contact was last contacted from a previous campaign
15. **previous:** _numerical_ - number of contacts performed before this campaign
16. **poutcome:** _categorical by outcome of previous marketing campaign_ - failure, success, other, unknown

**Note:** _Duration_ should be excluded because the prediction is needed before contacting the potential client

## Methods
The three models used were Logistic Regression, Decision Tree and Random Forest. The baseline accuracy for the models was 87.78% for Logistic Regression, 82.28% for Decision Tree 
and 89.18% for Random Forest. The baseline accuracy was based off simply ingesting the dataset and categorical encoding the categorical features. 

Once a baseline accuracy was established, EDA was conducted to determine next steps in preprocessing the data. Looking at the percentage of missing values (denoted as **unknown** 
in the categorical features) led to the discovery that _job_ and _education_ had a small enough percentage that KNN Imputation would be beneficial. KNN Imputation was implemented 
by manually encoding the _job_ and _education_ columns, replacing **unknown** with **math.nan** and then using KNNImputation to fill in the null values.

The histograms revealed that many of the numerical features were strongly right-skewed, and performing a log or square root transformation normalized the distribution. A problem 
arose because 'balance' has both 0 and negative values, so the 0 values were left as is, and the negative values were transformed in their absolute value and then changed back to 
negative.

The boxplots, along with the statistics from df.describe(), revealed that both _balance_ and _pdays_ had very large standard deviations, so the outliers were binned before the 
features were transformed.

The numerical features were then binned so that they would become categorical features, and then for each feature, the percentage of **yes** outcome was calculated for each value. 
This reveal any variation between the values in terms of a **yes** outcome and whether they would be helpful to include in the model. This was also used to determine at what value 
outliers in _balance_ and _pdays_ should be binned.

Min/Max Scaling was applied to every feature so that the models would not favor any feature above another.

The correlation matrix and PCA were used for feature selection and dimensionality reduction. The correlation matrix revealed that _pdays_ and _previous_ were highly correlated, 
and therefore one of them could be dropped. PCA however did not increase the accuracy of the models because the dataset was too small to benefit from it. A for-each loop, iteratively 
removing one feature from the dataset and running the model, showed which feature omissions resulted in a higher accuracy score. Removing all the features suggested resulted in the 
highest gain.

Adaptive Synthetic Sampling (ADASYN) was applied to the dataset before test/train/validation splitting to combat the imbalanced dataset. Of the roughly 45,000 entries, about 40,000 
resulted in a **no** outcome while only about 5,000 resulted in **yes**. While the Decision Tree and Random Forest models strongly benefitted from this, the accuracy of the Logistic 
Regression model plummeted to 60%, and therefore ADASYN was not applied to the Logistic Regression dataset.

Finally, a Grid Search and Halved Grid Search was used to tune the hyperparameters. Unfortunately, the suggested hyperparameters resulted in overfitting, leading to less accurate 
scores, so tuning was performed manually.

## Results
The final accuracy scores were 89.02% for Logistic Regression with a gain of 1.24, 90.82% for Decision Tree with a gain of 8.53, and 94.21% for Random Forest with a gain of 5.02. 
The confusion matrix revealed that the Logistic Regression model's lack of substantial progress was due to the model rarely predicting **yes**. ADASYN helped combat this problem 
in the other two models.

## Future Improvements
With more time, I would like to regularize the data in hopes to improve the Grid Search tool for better hyperparameter tuning optimization. Grid Search did not result in higher performance due to overfitting, and regularization would have combatted this problem and reduce the need for manually tuning hyperparameters.

I would also like to explore Feature Creation but combining multiple features to reduce dimensionality. Both the Decision Tree and Logistic Regression models performed better when multiple features were omitted – I would like to see if combining some of those features would have would have been more beneficial than leaving them out entirely. For example, creating _economic status_ by combining _education_, _balance_ and _age_ could have provided some insight from features which were largely left out of the models.
