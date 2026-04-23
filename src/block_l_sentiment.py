import pandas as pd
import numpy as np
import os


def compute_synthetic_sentiment(
    data_path="data/final/master_dataset_multi_horizon.csv",
    horizon=1
):
    """
    Horizon-Aware Behavioral Sentiment Proxy

    Sentiment Components:
    - Short-term momentum
    - Horizon-specific volatility deviation

    Output:
        Sentiment score in range [-1, 1]
    """

    try:
        df = pd.read_csv(data_path)

        if "Close" not in df.columns:
            print("Close column missing. Returning neutral sentiment.")
            return 0.0

        # Horizon-specific volatility lookback
        if horizon <= 3:
            lookback_vol = 5
        elif horizon <= 14:
            lookback_vol = 15
        else:
            lookback_vol = 30

        # Compute returns
        df["Returns"] = df["Close"].pct_change()

        # Rolling volatility
        df["Volatility"] = df["Returns"].rolling(lookback_vol).std()

        df = df.dropna()

        if len(df) == 0:
            return 0.0

        latest = df.iloc[-1]

        latest_return = latest["Returns"]
        latest_volatility = latest["Volatility"]

        # Momentum component
        return_score = np.tanh(latest_return * 15)

        # Fear component
        vol_mean = df["Volatility"].mean()
        vol_score = -np.tanh((latest_volatility - vol_mean) * 20)

        # Blend depends on horizon
        if horizon <= 3:
            sentiment = 0.7 * return_score + 0.3 * vol_score
        elif horizon <= 14:
            sentiment = 0.6 * return_score + 0.4 * vol_score
        else:
            sentiment = 0.5 * return_score + 0.5 * vol_score

        sentiment = np.clip(sentiment, -1, 1)

        return float(sentiment)

    except Exception as e:
        print(f"Error computing sentiment: {e}")
        return 0.0


# -----------------------------------------
# Save Sentiment for Specific Horizon
# -----------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    else:
        horizon = 1

    sentiment_value = compute_synthetic_sentiment(horizon=horizon)

    print(f"\nHorizon {horizon}D Sentiment:")
    print(sentiment_value)

    os.makedirs("outputs", exist_ok=True)

    output_path = f"outputs/sentiment_score_{horizon}D.csv"

    df_out = pd.DataFrame({
        "Date": [pd.Timestamp.today()],
        "Sentiment": [sentiment_value]
    })

    df_out.to_csv(output_path, index=False)

    print(f"\nSentiment saved to {output_path}")