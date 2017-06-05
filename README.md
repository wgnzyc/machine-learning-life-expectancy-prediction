# machine-learning-life-expectancy-prediction

# Project Description
## Main Purpose
The purpose is to predict life expectancy for the “average” person born in a certain year in one or more 
countries through machine learning algorithm.

## Data Description
Two files are used for the prediction.
1. life expectancy by country and year.csv 
2. GDP by country and year.csv

## Methodology
1. Use pandas to load datasets
2. Melt the datasets from wide to long format
3. Impute the missing values with the previous values first and then impute the remaining missing values with the next values
4. Merge the two long datasets. The datasets contains predictors 'country name','year' and 'GDP' and response variable 'life expectancy'.
5. Through cross validation, 'year' and 'country name' are selected to be the predictors and ExtraTreesRegressor is the algorithm.

## Example Result
If the input is 'China,2013,7.68380997', then predicted life expectancy for China in 2013 is around 75.
