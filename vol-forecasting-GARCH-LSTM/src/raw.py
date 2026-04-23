import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'data', 'raw')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_date = "1990-01-01"
end_date = datetime.date.today()

def load_data(start_date, end_date):
    data={}

    df=yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)
    if not df.empty:
        df.to_csv(os.path.join(output_dir, "SP500_raw.csv"), index=True)
        data["^GSPC"]=df['Close'].squeeze()

    return data

load_data(start_date, end_date)