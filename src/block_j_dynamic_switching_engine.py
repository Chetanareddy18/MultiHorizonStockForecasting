import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_absolute_error


# ----------------------------------------
# Base Weights by Horizon
# ----------------------------------------
def get_base_weights(horizon):

    if horizon <= 3:
        return {"lstm": 0.5, "tft": 0.3, "prophet": 0.2}

    elif horizon <= 14:
        return {"lstm": 0.3, "tft": 0.5, "prophet": 0.2}

    else:
        return {"lstm": 0.2, "tft": 0.3, "prophet": 0.5}


# ----------------------------------------
# Regime Adjustment
# ----------------------------------------
def adjust_for_regime(weights, regime):

    if regime == "high_volatility":
        weights["tft"] += 0.1
        weights["prophet"] -= 0.1

    if regime == "bear":
        weights["lstm"] += 0.1

    return weights


# ----------------------------------------
# Sentiment Adjustment
# ----------------------------------------
def adjust_for_sentiment(weights, sentiment):

    if sentiment < -0.5:
        weights["lstm"] += 0.1

    if sentiment > 0.5:
        weights["prophet"] += 0.1

    return weights


# ----------------------------------------
# Normalize Weights
# ----------------------------------------
def normalize(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# ----------------------------------------
# Main
# ----------------------------------------
def main():

    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    else:
        horizon = 1

    print(f"\nRunning Dynamic Switching for {horizon}D horizon")

    # ----------------------------------------
    # Load prediction files
    # ----------------------------------------
    lstm = pd.read_csv(f"outputs/lstm_predictions_{horizon}D.csv")
    tft = pd.read_csv(f"outputs/tft_predictions_{horizon}D.csv")
    prophet = pd.read_csv(f"outputs/prophet_predictions_{horizon}D.csv")

    # ----------------------------------------
    # Align lengths
    # ----------------------------------------
    min_len = min(len(lstm), len(tft), len(prophet))

    lstm = lstm.tail(min_len)
    tft = tft.tail(min_len)
    prophet = prophet.tail(min_len)

    # IMPORTANT: All files have columns:
    # Date | Actual | Prediction

    y_true = lstm["Actual"].values
    lstm_pred = lstm["Prediction"].values
    tft_pred = tft["Prediction"].values
    prophet_pred = prophet["Prediction"].values

    # ----------------------------------------
    # Example regime & sentiment
    # (Later you can connect real modules)
    # ----------------------------------------
    regime = "high_volatility"
    sentiment_score = -0.6

    # ----------------------------------------
    # Get dynamic weights
    # ----------------------------------------
    weights = get_base_weights(horizon)
    weights = adjust_for_regime(weights, regime)
    weights = adjust_for_sentiment(weights, sentiment_score)
    weights = normalize(weights)

    print("Final Adaptive Weights:", weights)

    # ----------------------------------------
    # Final dynamic ensemble prediction
    # ----------------------------------------
    final_pred = (
        weights["lstm"] * lstm_pred +
        weights["tft"] * tft_pred +
        weights["prophet"] * prophet_pred
    )

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------
    mae = mean_absolute_error(y_true, final_pred)
    print(f"Dynamic MAE ({horizon}D): {mae:.4f}")

    # ----------------------------------------
    # Save output
    # ----------------------------------------
    os.makedirs("outputs", exist_ok=True)

    pd.DataFrame({
        "Date": lstm["Date"],
        "Actual": y_true,
        "Dynamic_Prediction": final_pred
    }).to_csv(f"outputs/dynamic_predictions_{horizon}D.csv", index=False)

    print("Dynamic predictions saved.")


if __name__ == "__main__":
    main()