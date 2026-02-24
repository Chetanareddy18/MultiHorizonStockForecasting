import pandas as pd
import numpy as np


def detect_regime(data_path="data/final/master_dataset.csv"):

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    df["Returns"] = df["Close"].pct_change()

    # Rolling volatility
    df["Volatility"] = df["Returns"].rolling(20).std()

    # Moving averages
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    latest = df.iloc[-1]

    regime = "normal"

    # High volatility check
    if latest["Volatility"] > df["Volatility"].mean():
        regime = "high_volatility"

    # Trend check
    if latest["MA50"] < latest["MA200"]:
        regime = "bear"

    if latest["MA50"] > latest["MA200"]:
        regime = "bull"

    return regime


if __name__ == "__main__":
    print("Detected Regime:", detect_regime())