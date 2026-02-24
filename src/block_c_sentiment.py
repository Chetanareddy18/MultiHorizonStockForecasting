import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

os.makedirs("data/processed", exist_ok=True)

def generate_sentiment():
    print("Generating engineered sentiment...")

    # -----------------------------
    # Load Market + Macro Blocks
    # -----------------------------
    market = pd.read_csv(
        "data/raw/market_block.csv",
        index_col=0,
        parse_dates=True
    )

    macro = pd.read_csv(
        "data/processed/macro_block.csv",
        index_col=0,
        parse_dates=True
    )

    # -----------------------------
    # Merge Market + VIX
    # -----------------------------
    df = market.join(macro, how="inner")

    # -----------------------------
    # Raw Sentiment Logic
    # -----------------------------
    df["Raw_Sentiment"] = (
        0.7 * df["Return"] -
        0.3 * (df["VIX"] / df["VIX"].max())
    )

    # Smooth it
    df["Raw_Sentiment"] = df["Raw_Sentiment"].rolling(
        window=5,
        min_periods=1
    ).mean()

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df["Sentiment"] = scaler.fit_transform(df[["Raw_Sentiment"]])

    # -----------------------------
    # 🔥 ADVANCED FEATURES
    # -----------------------------
    df["Sentiment_Lag1"] = df["Sentiment"].shift(1)
    df["Sentiment_Lag3"] = df["Sentiment"].shift(3)
    df["Sentiment_Roll5"] = df["Sentiment"].rolling(5).mean()

    # Drop initial NaNs caused by lag/rolling
    sentiment_df = df[
        ["Sentiment", "Sentiment_Lag1", "Sentiment_Lag3", "Sentiment_Roll5"]
    ].dropna()

    # Save
    sentiment_df.to_csv("data/processed/sentiment_block.csv")

    print("Sentiment Generated Successfully.")
    print("Shape:", sentiment_df.shape)
    print("\nSentiment Stats:")
    print(sentiment_df.describe())


if __name__ == "__main__":
    generate_sentiment()
