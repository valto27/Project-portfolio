import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance
import pandas as pd
import requests
import zipfile
import io
from config import TICKERS, START_DATE, END_DATE

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
prices_path = os.path.join(data_dir, 'prices.csv')
ff_path = os.path.join(data_dir, 'fama_french.csv')
rf_path = os.path.join(data_dir, 'risk_free.csv')

os.makedirs(data_dir, exist_ok=True)

def load_prices():
    if os.path.exists(prices_path):
        return pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    prices = yfinance.download(TICKERS, start=START_DATE, end=END_DATE)['Close']
    
    tickers_before = set(prices.columns)

    mean_NaN = prices.isna().mean()
    print(mean_NaN.sort_values(ascending=False).head(20)) 

    prices = prices.loc[:, mean_NaN < 0.2]
    prices = prices.ffill()

    tickers_after = set(prices.columns)
    removed_tickers = tickers_before - tickers_after    
    print(f"Removed tickers due to NaN values: {removed_tickers}")

    prices.to_csv(prices_path)
    return prices

def load_fama_french():
    if os.path.exists(ff_path):
        return pd.read_csv(ff_path, index_col=0, parse_dates=True)

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            FF = pd.read_csv(f, skiprows=3, index_col=0)

    FF = FF[FF.index.astype(str).str.strip().str.len() == 6]
    FF.index = pd.to_datetime(FF.index.astype(str).str.strip(), format='%Y%m')
    FF = FF.apply(pd.to_numeric, errors='coerce')
    FF = FF / 100

    FF.to_csv(ff_path)
    return FF



def load_risk_free():
    if os.path.exists(rf_path):
        return pd.read_csv(rf_path, index_col=0, parse_dates=True)
    
    RF = yfinance.download('^IRX', start=START_DATE, end=END_DATE)['Close']
    RF = RF.resample('ME').last()
    RF = RF / 100 / 12

    RF.to_csv(rf_path)
    return RF

if __name__ == "__main__":
    prices = load_prices()
    print(prices.shape)
    print(prices.head())
    ff = load_fama_french()
    print(ff.shape)
    print(ff.head())
    rf = load_risk_free()
    print(rf.shape)
    print(rf.head())