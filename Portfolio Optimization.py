#Import Python Libraries

from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get Stock Symbols
#FAANG
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

#Assign weights to portfolio
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

#Get the stock/portfolio starting date
stockStartDate = '2013-01-01'

#Get the stock/portfolio ending date
today = datetime.today().strftime('%Y-%m-%d')
today

#Create a dataframe to store adjusted close price of the stocks
df = pd.DataFrame()

#Store the adjusted close price of the stock into the df
for stock in assets:
    df[stock] = web.DataReader(stock,data_source = 'yahoo', start = stockStartDate, end = today)['Adj Close']

#Show the df
df

#Visually show the stock/portfolio
title = 'Portfolio Adj. Close Price History'

#Get the stocks
my_stocks = df

#Create and plot the graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label = c)

plt.title(title)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj. Price in USD($)', fontsize = 18)
plt.legend(my_stocks.columns.values, loc = 'upper left')
plt.show()

#Show the daily simple return
returns = df. pct_change()
returns

#Create and show the annualized covariance matrix
#252 is number of trading days
cov_matrix_annual = returns.cov() * 252
cov_matrix_annual

#Calculate the portfolio variance
#Weights transposed x covariance matrix x weights
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance

#Calculate portfolio volatility (standard dev)
port_volatility = np.sqrt(port_variance)
port_volatility

#Calculate annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
portfolioSimpleAnnualReturn

#Show the expected annual return, volatility, and variance
percent_var = str(round(port_variance,2) * 100) + '%'
percent_vols = str(round(port_volatility,2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print('Expected Annual Return: ' + percent_ret)
print('Annual Volatility: ' + percent_vols)
print('Annual Variance: ' + percent_var)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#Portfolio Optimization
#Calculate expected returns and the annualized sample covariance matrix of asset returns

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for max sharpe ratio
ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

#Check Sum
0.14912 + 0.24703 + 0.26232 + 0.3043 + 0.03723

#Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights_ = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value= 15000)
allocation, leftover = da.lp_portfolio()
print('Discrete Allocation:', allocation)
print('Funds Remaining: ${:.2f}'.format(leftover))


