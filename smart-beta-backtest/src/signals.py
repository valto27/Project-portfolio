import pandas as pd
from data_loader import load_prices
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
momentum_path = os.path.join(data_dir, 'momentum.csv')
volatility_path = os.path.join(data_dir, 'low_volatility.csv')

def compute_momentum(monthly_prices):
    momentum = monthly_prices.shift(1) / monthly_prices.shift(13) - 1
    momentum.to_csv(momentum_path)
    return momentum

def compute_low_volatility(monthly_returns):
    low_volatility = monthly_returns.rolling(6).std()
    low_volatility.to_csv(volatility_path)
    return low_volatility

if __name__ == "__main__":
    from data_loader import load_prices
    prices = load_prices()
    prices = prices.resample('ME').last()
    yields = prices.pct_change()
    momentum = compute_momentum(prices)
    print("Momentum shape:", momentum.shape)
    print(momentum.tail())
    low_volatility = compute_low_volatility(yields)
    print("Low-vol shape:", low_volatility.shape)
    print(low_volatility.tail())