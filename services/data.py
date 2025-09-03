\
from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_ohlc(ticker: str, horizon: str = "ST") -> pd.DataFrame:
    ticker = (ticker or "AAPL").upper().strip()
    horizon = (horizon or "ST").upper()
    if horizon == "ST":
        period, interval = "6mo", "1d"
    elif horizon == "MID":
        period, interval = "3y", "1d"
    else:
        period, interval = "10y", "1wk"
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    data = data.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close"})
    data = data[["Open","High","Low","Close"]].dropna()
    return data
