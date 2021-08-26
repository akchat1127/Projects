#Does explanatory variable(oil) do a good job of predicting the dependent variable (1 share of Exxon stock)

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

#Load the data
price_data = pd.read_excel('oil_exxon.xlsx')

#Use date column as index and check for errors with date
price_data.index = pd.to_datetime(price_data['date'])
price_data.head()

#Get rid of old date column
price_data = price_data.drop(['date'], axis = 1)
price_data.head()

#Check the data types
price_data.dtypes
#Looks like data types are ok and don't need to be changed

#Correct misspelled column name
new_column_names = {'exon_price': 'exxon_price'}
#Rename column
price_data = price_data.rename(columns = new_column_names)
price_data.head()

#Check for missing values
price_data.isna().any()

#Drop missing values
price_data = price_data.dropna()
price_data.isna().any()
#No missing values in data frame

#Explore the data
#Define x and y data
x = price_data['exxon_price']
y = price_data['oil_price']

#Create scatter plot
plt.plot(x,y,'o', color = 'cadetblue', label = 'Daily Price')

#Scatter plot formatting/labeling
plt.title('Exxon Stock Price vs. Oil Price')
plt.xlabel('Exxon Stock Price')
plt.ylabel('Oil Price')
plt.legend()
plt.show()

#Measure correlation
price_data.corr()
#Approx 60% correlation between the two. Possibly significant?

#Create statistical summary
price_data.describe()
#No concerning outliers at first glance

#Double check for outliers and look for skewness
price_data.hist(grid = False, color = 'cadetblue')
#Looks slightly skewed but want to quantify

#Calculate kurtosis. Fisher means kurtosis should be closer to 0 as opposed to 3
exxon_kurtosis = kurtosis(price_data['exxon_price'], fisher = True)
oil_kurtosis = kurtosis(price_data['oil_price'], fisher = True)

"Exxon Kurtosis: {:.2}".format(exxon_kurtosis)
"Oil Kurtosis: {:.2}".format(oil_kurtosis)

#Kurtosis for both variables is close to 0 and outliers are not a concern



#Calculate skewness
exxon_skew = skew(price_data['exxon_price'])
oil_skew = skew(price_data['oil_price'])

"Exxon Skew: {:.2}".format(exxon_skew)
"Oil Skew: {:.2}".format(oil_skew)

#Exxon is moderately skewed and oil is on the cusp of being significantly skewed. Can opt for log transformation depending on how conservative we want to be

#Perform kurtosis test

stats.kurtosistest(price_data['exxon_price'])

stats.kurtosistest(price_data['oil_price'])

#Perform skew test

stats.skewtest(price_data['exxon_price'])
stats.skewtest(price_data['oil_price'])


#Split the data
Y = price_data.drop('oil_price', axis = 1)
X = price_data[['oil_price']]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state= 1)

#Create linear regression model object
regression_model = LinearRegression()

#pass through X_train and y_Train model
regression_model.fit(X_train, y_train)

#Explore output
intercept = regression_model.intercept_[0]
coefficient = regression_model.coef_[0][0]

print("The intercept for our model is {:.4}".format(intercept))
print("The coefficient for our model is {:.2}".format(coefficient))

#Test a single prediction
prediction = regression_model.predict([[67.33]])
predicted_value = prediction[0][0]
print("The predicted value is {:.4}".format(predicted_value))

#To interpret output, it would say that for a given barrel of oil at price 67.33 would predict an Exxon mobile stock price of 86.0

#Test multiple predictions
y_predict = regression_model.predict(X_test)

#Show first 5 predictions
y_predict[:5]

#Evaluate the model

#Define the input
X2 = sm.add_constant(X)

#Create an OLS model
model = sm.OLS(Y, X2)

#Fit the model
est = model.fit()

#Condifence Intervals
est.conf_int()

#Interpreting the results, there is 95% confidence the oil price coefficient exists between 0.214 and 0.248

#Estimate the p-values
#Null Hypothesis: There is no relationship between the price of oil and the price of Exxon
#Alternative Hypothesis: There is a relationship between the price of oil and the price of Exxon and the coefficient is not 0

#If the null is rejected: There is a  relationship between oil price and the price of Exxon and the cofficient is not 0
#If the fail to reject the null: There is no relationship between oil price and the price of Exxon and the cofficient is 0

#Estimate the p-values
est.pvalues

#p-value < 0.05 therefore the null hypothesis is rejected and so there is relationship between the oil price and price of Exxon stock

#Model Fit

#Calculate the MSE (mean squared error). MSE punishes larger error terms and is therefore more popular than MAE
model_mse = mean_squared_error(y_test, y_predict)

#Calculate the MAE (mean absolute error). MAE provides the mean of the absolute value of errors but doesn't provide direction (too high or low)
model_mae = mean_absolute_error(y_test, y_predict)

#Calculate RMSE (root mean squared error). RMSE is the square root of the mean of the squared error. RMSE is even more favored because it allows for interpretation of output in y-units
model_rmse = math.sqrt(model_mse)

#Print output
print("MSE {:.3}".format(model_mse))
print("MSE {:.3}".format(model_mae))
print("MSE {:.3}".format(model_rmse))

#Test goodness of fit with R-squared metric. Usually, higher R-squared metric means better goodnes of fit
#However, more features inflates the R-squared metric, therefore sometimes the adjusted R-squared is preferable which penalizes more complex models

model_r2 = r2_score(y_test, y_predict)
print("R2 {:.2}".format (model_r2))

#R-squared is 0.36, therefore the data explains 36% of the variance. Is this good/bad? This is ok for a stock model..
#Adding variables will make the R-squared error however this may not mean that the model is necessarily better

#Print summary
print(est.summary())

#Here, adjusted r-squared is very similar because the model is not complicated

#Plot residuals
(y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plt.show

#Normally distributed for the most part

#Plot output
plt.scatter(X_test, y_test, color = 'gainsboro', label = 'Price')
plt.plot(X_test, y_predict, color = 'royalblue', label = 'Regression Line')

plt.title("Linear Regression Model Exxon vs. Oil Price")
plt.xlabel("Oil")
plt.ylabel("Exxon Mobile")
plt.legend()
plt.show()

#Save model for future use
import pickle
with open('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regression_model,f)

#Load back in
with open('my_linear_regression.sav', 'rb') as f:
    regression_model_2 = pickle.load(f)

#Make new prediction
regression_model_2.predict([[67.33]])
