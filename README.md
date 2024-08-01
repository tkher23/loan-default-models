# loan-default-models

## Summary
This is a repository for a project conducted to provide small businesses alleviation with accepting or denying certain loan offers. Models were trained off of 800,000+ examples of previous loans from public data from the US Small Business Administration found here: https://data.sba.gov/en/dataset/7-a-504-foia

Using Neural Networks and Decision Tree frameworks, this project was able to develop a model with **96% accuracy** to predict a loan's status (Charged-off vs Paid-in-Full) based on the following variables: The Borrower's City, State and Zip, the Bank giving out the loan, The Gross Amount approved, Initial Interest Rate, Fixed/Variable, Term Length, Type of Business (aggregated in model using Naics Codes), Business Structure (Corporation vs Individual)

## Methodology:
1. Extracting relevant metrics from SBA data (expressed above)
2. Using EDA to combine data files and fill in missing values (Pandas)
3. Standardize numerical variables (Scikit-learn)
4. Use target encoding for high cardinality categorical variables and one-hot encoding for low cardinality categorical variables
5. Splitting up data into training, cross-validation, and test sets (Scikit-learn)
6. Automating the training of neural network models with different architectures and comparing accuracies (Tensorflow - Dense layers)
7. Correcting potential overfitting by comparing test vs training sets
8. Automating decision tree frameworks with Random Forest model (Scikit-learn: Ensemble)
9. Finding most accurate decision tree parameters using GridSearchCV and checking for overfitting
10. Comparing and choosing most effective and generalizable model

## Algorithmic Efficiency Comparison Results:
Random Forest algorithm was able to identify a model that predicted the test set (over 50,000 examples) with 96% accuracy. Complexity of neural network architecture did not improve accuracy (cross-validation set error): 7 models predicted loan status accuracy at 92%.
