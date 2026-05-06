import pandas as pd
import os
from math import sqrt


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
metrics_path = os.path.join(data_dir, 'metrics.csv')

def annualized_return(returns):
    annual_returns = (1 + returns).prod() ** (12 / len(returns)) - 1
    return annual_returns

def annualized_volatility(returns):
    annual_volatility =returns.std() * sqrt(12)
    return annual_volatility

def sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    return annualized_return(excess_returns) / annualized_volatility(excess_returns)

def max_drawdown(returns):
    equity_curve = (1 + returns).cumprod()
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calmar_ratio(returns):
    calmar = annualized_return(returns) / abs(max_drawdown(returns))
    return calmar

def compute_all(returns, risk_free_rate):
    metrics ={
        'Annualized Return': annualized_return(returns),
        'Annualized Volatility': annualized_volatility(returns),
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate),
        'Max Drawdown': max_drawdown(returns),
        'Calmar Ratio': calmar_ratio(returns)
    }
    return metrics

if __name__ == "__main__":
    returns = pd.read_csv(os.path.join(data_dir, 'results.csv'), index_col=0, parse_dates=True)['returns']
    risk_free = pd.read_csv(os.path.join(data_dir, 'risk_free.csv'), index_col=0, parse_dates=True)
    rf_mean = risk_free.mean().values[0]
    metrics = compute_all(returns, rf_mean)
    print(metrics)