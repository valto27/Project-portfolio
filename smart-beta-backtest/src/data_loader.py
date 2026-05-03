import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance
import pandas as pd
from config import TICKERS, START_DATE, END_DATE

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prices_dir = os.path.join(base_dir, 'data', 'prices')
file_path = os.path.join(prices_dir, 'prices.csv')

def load_prices():
    if not os.path.exists(prices_dir):
        os.makedirs(prices_dir)

    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    prices = yfinance.download(TICKERS, start=START_DATE, end=END_DATE)['Close']
    
    tickers_before = set(prices.columns)

    mean_NaN = prices.isna().mean()
    print(mean_NaN.sort_values(ascending=False).head(20)) 

    prices = prices.loc[:, mean_NaN < 0.2]
    prices = prices.ffill()

    tickers_after = set(prices.columns)
    removed_tickers = tickers_before - tickers_after    
    print(f"Removed tickers due to NaN values: {removed_tickers}")

    prices.to_csv(file_path)
    return prices

if __name__ == "__main__":
    prices = load_prices()
    print(prices.shape)
    print(prices.head())