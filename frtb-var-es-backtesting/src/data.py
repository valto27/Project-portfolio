import numpy as np
import pandas as pd
from src.config import TICKER_MAP

def load_data(assets, start_date, end_date):
    data={}
    import yfinance as yf
    for asset in assets:
        ticker=TICKER_MAP.get(asset)

        if ticker:
            df=yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                data[asset]=df['Close'].squeeze()

        else:
            print(f"No ticker mapping found for '{asset}'. Skipping.")
    return data

def read_sheet_series(path: str, sheet_name: str, date_col: str, price_col: str, start_row: int) -> pd.Series:
    skip = max(0, start_row - 1)
    usecols = f"{date_col}:{price_col}"
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols, skiprows=skip, header=None, names=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if not np.issubdtype(df["Price"].dtype, np.number):
        df["Price"] = pd.to_numeric(
            df["Price"].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d.-]", "", regex=True),
            errors="coerce"
        )
    df = df.dropna().drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    series = df.set_index("Date")["Price"].astype(float)
    if series.empty: raise ValueError(f"Sheet '{sheet_name}' produced an empty series.")
    return series

def load_prices(cfg):
    if cfg["use_yfinance"]:
        return load_data(cfg["target_assets"], cfg["yf_start"], cfg["yf_end"])
    else:
        return {asset: read_sheet_series(cfg["excel_path"], asset, cfg["date_col_letter"], cfg["price_col_letter"], cfg["start_row"]) for asset in cfg["target_assets"]}

def to_returns(price: pd.Series, log: bool = True) -> pd.Series:
    r = np.log(price / price.shift(1)) if log else price.pct_change()
    return r.rename("ret")