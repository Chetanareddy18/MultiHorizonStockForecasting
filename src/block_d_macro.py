import yfinance as yf
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

def collect_macro():
    start = "2015-01-01"
    end = "2024-12-31"

    vix = yf.download("^VIX", start=start, end=end)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix = vix[['Close']]
    vix.rename(columns={'Close':'VIX'}, inplace=True)

    vix.to_csv("data/processed/macro_block.csv")
    print("Macro block saved.")

if __name__ == "__main__":
    collect_macro()