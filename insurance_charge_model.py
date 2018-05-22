
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## =================================================

# import lib.. pandas, numpy, matplotlib, rbParams, seaborn, sklearn matrics

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

from pylab import rcParams
import seaborn as sb
from sklearn.metrics import r2_score

# set fig size
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
# "python.jupyter.startupCode": [
#        "%matplotlib inline"]

## =================================================

# load dataset /// chnage file location ///
dataset = pd.read_csv('/Users/samng/OneDrive/DBA/vscode/case_001 insurance_charge/insurance dataset.csv')

# name features
dataset.columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

## =================================================

# review head()
print(dataset.head(15))

sb.pairplot(dataset)
plt.show()

print(dataset.corr())
print(dataset.describe())

## =================================================

# set variables x, y
X = dataset[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = dataset['charges']  

# import sklearn lib, which is a popular ML lib
# split dataset into 80% train data and 20% test data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import sklearn linear regression model
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  

# get coefficient of independent variables
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print('Coeff_df:', coeff_df)  

## =================================================

# I/O of test data
y_pred = regressor.predict(X_test)  
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print('df:', df)

## =================================================

# import sklearn metrics for model evaluation
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R^2 score:', r2_score(y_test, y_pred))

## =================================================

