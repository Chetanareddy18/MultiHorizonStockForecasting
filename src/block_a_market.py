import yfinance as yf
import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

def collect_market_data():
    ticker = "^NSEI"
    start = "2015-01-01"
    end = "2024-12-31"

    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']]

    # Returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # High-Low %
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']

    # Volume change
    df['Vol_Change'] = df['Volume'].pct_change()

    df.dropna(inplace=True)

    df.to_csv("data/raw/market_block.csv")
    print("Market block saved.")

if __name__ == "__main__":
    collect_market_data()