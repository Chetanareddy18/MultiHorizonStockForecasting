import pandas as pd
import numpy as np


def detect_regime(data_path="data/final/master_dataset.csv"):
    """
    Detects combined market regime:
    - bull_low_volatility
    - bull_high_volatility
    - bear_low_volatility
    - bear_high_volatility
    """

    df = pd.read_csv(data_path)

    if "Close" not in df.columns:
        raise ValueError("Column 'Close' not found in dataset")

    df["Returns"] = df["Close"].pct_change()

    # Rolling volatility
    df["Volatility"] = df["Returns"].rolling(20).std()

    # Moving averages
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    df = df.dropna()

    if len(df) == 0:
        raise ValueError("Not enough data to compute regime")

    latest = df.iloc[-1]

    # Volatility classification
    avg_vol = df["Volatility"].mean()

    if latest["Volatility"] > avg_vol:
        vol_regime = "high_volatility"
    else:
        vol_regime = "low_volatility"

    # Trend classification
    if latest["MA50"] > latest["MA200"]:
        trend_regime = "bull"
    else:
        trend_regime = "bear"

    final_regime = f"{trend_regime}_{vol_regime}"

    return final_regime


if __name__ == "__main__":
    regime = detect_regime()
    print("Detected Regime:", regime)