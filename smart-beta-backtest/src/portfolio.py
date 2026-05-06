import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import cvxpy as cp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def equal_weights(selected_tickers):
    n = len(selected_tickers)
    weight = 1/n
    return pd.Series(weight, index=selected_tickers)

def markowitz_weights(returns):
    lw = LedoitWolf().fit(returns)
    cov_matrix = lw.covariance_
    mu = returns.mean().values
    n = len(mu)
    
    y = cp.Variable(n)
    k = cp.Variable()
    
    risk = cp.quad_form(y, cov_matrix)
    objective = cp.Minimize(risk)
    
    constraints = [
        y @ mu == 1,
        cp.sum(y) == k,
        y >= 0,
        k >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    optimal_weights = y.value / k.value
    return pd.Series(optimal_weights, index=returns.columns)

def risk_parity_weights(returns):
    lw = LedoitWolf().fit(returns)
    cov_matrix = lw.covariance_
    volatilities = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    return pd.Series(weights, index=returns.columns)

if __name__ == "__main__":
    from data_loader import load_prices
    selected_tickers = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM']
    prices = load_prices()
    prices = prices[selected_tickers].resample('ME').last()
    returns = prices.pct_change().dropna()
    
    print(equal_weights(selected_tickers))
    print(markowitz_weights(returns))
    print(risk_parity_weights(returns))