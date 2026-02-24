import pandas as pd
import os

os.makedirs("data/final", exist_ok=True)

def consolidate():

    market = pd.read_csv("data/raw/market_block.csv", index_col=0, parse_dates=True)
    tech = pd.read_csv("data/processed/technical_block.csv", index_col=0, parse_dates=True)
    sentiment = pd.read_csv("data/processed/sentiment_block.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv("data/processed/macro_block.csv", index_col=0, parse_dates=True)

    df = market.join(tech, how='inner')
    df = df.join(sentiment, how='inner')
    df = df.join(macro, how='inner')

    # Multi-horizon targets
    df['Target_1D'] = df['Close'].shift(-1)
    df['Target_7D'] = df['Close'].shift(-7)
    df['Target_30D'] = df['Close'].shift(-30)

    df.dropna(inplace=True)

    df.to_csv("data/final/master_dataset.csv")

    print("MASTER DATASET READY.")
    print("Shape:", df.shape)

if __name__ == "__main__":
    consolidate()