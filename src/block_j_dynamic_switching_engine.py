# ============================================================
# BLOCK J: DYNAMIC SWITCHING ENGINE (HORIZON + SENTIMENT + REGIME + UNCERTAINTY)
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample


# ============================================================
# BASE WEIGHTS BY HORIZON
# ============================================================
def get_base_weights(horizon):

    if horizon <= 3:
        return {"lstm": 0.5, "tft": 0.3, "prophet": 0.2}

    elif horizon <= 14:
        return {"lstm": 0.3, "tft": 0.5, "prophet": 0.2}

    else:
        return {"lstm": 0.2, "tft": 0.3, "prophet": 0.5}


# ============================================================
# REGIME DETECTION
# ============================================================
def detect_regime(y_true):

    returns = np.diff(y_true) / y_true[:-1]
    volatility = np.std(returns)
    mean_return = np.mean(returns)

    if volatility > 0.03:
        return "high_volatility"
    elif mean_return < 0:
        return "bear"
    else:
        return "normal"


# ============================================================
# LOAD HORIZON-SPECIFIC SENTIMENT
# ============================================================
def load_sentiment(horizon):

    path = f"outputs/sentiment_score_{horizon}D.csv"

    if not os.path.exists(path):
        print("Sentiment file not found. Using neutral sentiment.")
        return 0.0

    df = pd.read_csv(path)

    if "Sentiment" not in df.columns:
        return 0.0

    return float(df["Sentiment"].iloc[-1])


# ============================================================
# REGIME ADJUSTMENT
# ============================================================
def adjust_for_regime(weights, regime):

    weights = weights.copy()

    if regime == "high_volatility":
        weights["tft"] += 0.1
        weights["prophet"] += 0.05
        weights["lstm"] -= 0.15

    elif regime == "bear":
        weights["prophet"] += 0.1
        weights["lstm"] -= 0.05
        weights["tft"] -= 0.05

    return weights


# ============================================================
# SENTIMENT ADJUSTMENT (INTENSITY-BASED)
# ============================================================
def adjust_for_sentiment(weights, sentiment):

    weights = weights.copy()

    # Strong Bearish Sentiment
    if sentiment < -0.4:
        weights["prophet"] += 0.1
        weights["lstm"] -= 0.05
        weights["tft"] -= 0.05

    # Mild Bearish
    elif sentiment < -0.1:
        weights["prophet"] += 0.05
        weights["lstm"] -= 0.03
        weights["tft"] -= 0.02

    # Strong Bullish
    elif sentiment > 0.4:
        weights["lstm"] += 0.1
        weights["prophet"] -= 0.05
        weights["tft"] -= 0.05

    # Mild Bullish
    elif sentiment > 0.1:
        weights["lstm"] += 0.05
        weights["prophet"] -= 0.03
        weights["tft"] -= 0.02

    return weights


# ============================================================
# NORMALIZE WEIGHTS
# ============================================================
def normalize(weights):

    total = sum(weights.values())

    # Safety check
    if total == 0:
        return {"lstm": 0.33, "tft": 0.33, "prophet": 0.34}

    return {k: max(v, 0) / total for k, v in weights.items()}


# ============================================================
# BOOTSTRAP UNCERTAINTY
# ============================================================
def bootstrap_uncertainty(y_true, final_pred, n_bootstrap=200):

    residuals = y_true - final_pred
    bootstrap_preds = []

    for _ in range(n_bootstrap):
        sampled_residuals = resample(residuals)
        new_pred = final_pred + sampled_residuals
        bootstrap_preds.append(new_pred)

    bootstrap_preds = np.array(bootstrap_preds)

    lower = np.percentile(bootstrap_preds, 5, axis=0)
    upper = np.percentile(bootstrap_preds, 95, axis=0)
    std = np.std(bootstrap_preds, axis=0)

    return lower, upper, std


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():

    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    else:
        horizon = 1

    print(f"\nRunning Dynamic Switching for {horizon}D horizon")

    # --------------------------------------------------------
    # Load Model Predictions
    # --------------------------------------------------------
    lstm = pd.read_csv(f"outputs/lstm_predictions_{horizon}D.csv")
    tft = pd.read_csv(f"outputs/tft_predictions_{horizon}D.csv")
    prophet = pd.read_csv(f"outputs/prophet_predictions_{horizon}D.csv")

    min_len = min(len(lstm), len(tft), len(prophet))
    lstm = lstm.tail(min_len)
    tft = tft.tail(min_len)
    prophet = prophet.tail(min_len)

    y_true = lstm["Actual"].values
    lstm_pred = lstm["Prediction"].values
    tft_pred = tft["Prediction"].values
    prophet_pred = prophet["Prediction"].values

    # --------------------------------------------------------
    # Regime + Sentiment
    # --------------------------------------------------------
    regime = detect_regime(y_true)
    sentiment_score = load_sentiment(horizon)

    print("Detected Regime:", regime)
    print("Latest Sentiment Score:", sentiment_score)

    # --------------------------------------------------------
    # Adaptive Weight Computation
    # --------------------------------------------------------
    weights = get_base_weights(horizon)
    weights = adjust_for_regime(weights, regime)
    weights = adjust_for_sentiment(weights, sentiment_score)
    weights = normalize(weights)

    print("Final Adaptive Weights:", weights)

    # --------------------------------------------------------
    # Final Ensemble Prediction
    # --------------------------------------------------------
    final_pred = (
        weights["lstm"] * lstm_pred +
        weights["tft"] * tft_pred +
        weights["prophet"] * prophet_pred
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    mae = mean_absolute_error(y_true, final_pred)
    print(f"Dynamic MAE ({horizon}D): {mae:.4f}")

    # --------------------------------------------------------
    # Uncertainty Estimation
    # --------------------------------------------------------
    lower, upper, std = bootstrap_uncertainty(y_true, final_pred)
    print("Uncertainty Estimation Added (90% CI)")

    # --------------------------------------------------------
    # Save Results
    # --------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    pd.DataFrame({
        "Date": lstm["Date"],
        "Actual": y_true,
        "Dynamic_Prediction": final_pred,
        "Lower_90": lower,
        "Upper_90": upper,
        "Std_Uncertainty": std
    }).to_csv(f"outputs/dynamic_predictions_{horizon}D.csv", index=False)

    print("Dynamic probabilistic predictions saved successfully.")


if __name__ == "__main__":
    main()