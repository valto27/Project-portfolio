import numpy as np
import pandas as pd
import datetime
import yfinance as yf

start_date = "1990-01-01"
end_date = datetime.date.today()

def load_data(start_date, end_date):
    data={}

    df=yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)
    if not df.empty:
        data["^GSPC"]=df['Close'].squeeze()

    return data