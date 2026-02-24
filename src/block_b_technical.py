import pandas as pd
import ta
import os

os.makedirs("data/processed", exist_ok=True)

def generate_technical():

    df = pd.read_csv("data/raw/market_block.csv", index_col=0, parse_dates=True)

    tech = pd.DataFrame(index=df.index)

    tech['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    macd = ta.trend.MACD(df['Close'])
    tech['MACD'] = macd.macd()
    tech['MACD_Signal'] = macd.macd_signal()

    tech['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    tech['SMA50'] = df['Close'].rolling(50).mean()

    tech['Rolling_Vol_14'] = df['Return'].rolling(14).std()
    tech['Rolling_Vol_30'] = df['Return'].rolling(30).std()

    tech.dropna(inplace=True)

    tech.to_csv("data/processed/technical_block.csv")

    print("Technical block saved cleanly.")

if __name__ == "__main__":
    generate_technical()