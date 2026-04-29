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

    # Align all three model outputs by Date when possible. If any of the
    # writers did not include a Date column (e.g. TFT), fall back to a
    # tail-based row-wise alignment.
    has_dates = all("Date" in df.columns for df in (lstm, tft, prophet))

    if has_dates:
        lstm_r = lstm.rename(columns={"Prediction": "lstm_pred", "Actual": "actual_lstm"})
        tft_r = tft.rename(columns={"Prediction": "tft_pred"})
        prophet_r = prophet.rename(columns={"Prediction": "prophet_pred"})

        merged = (
            lstm_r[["Date", "actual_lstm", "lstm_pred"]]
            .merge(tft_r[["Date", "tft_pred"]], on="Date", how="inner")
            .merge(prophet_r[["Date", "prophet_pred"]], on="Date", how="inner")
        )
    else:
        merged = pd.DataFrame()

    if len(merged) == 0:
        # Fallback: row-wise tail alignment when dates can't be matched.
        min_len = min(len(lstm), len(tft), len(prophet))
        date_series = (
            lstm["Date"].tail(min_len).values
            if "Date" in lstm.columns
            else pd.date_range(end=pd.Timestamp.today(), periods=min_len).strftime("%Y-%m-%d")
        )
        merged = pd.DataFrame({
            "Date": date_series,
            "actual_lstm": lstm["Actual"].tail(min_len).values,
            "lstm_pred": lstm["Prediction"].tail(min_len).values,
            "tft_pred": tft["Prediction"].tail(min_len).values,
            "prophet_pred": prophet["Prediction"].tail(min_len).values,
        })

    y_true = merged["actual_lstm"].values
    lstm_pred = merged["lstm_pred"].values
    tft_pred = merged["tft_pred"].values
    prophet_pred = merged["prophet_pred"].values

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

    # Auto-detect a "stuck" model (low prediction variance relative to
    # actual). A degenerate model such as a TFT that collapsed to its
    # training mean otherwise drags the whole ensemble down.
    actual_std = float(np.std(y_true)) + 1e-9
    for name, preds in (("lstm", lstm_pred), ("tft", tft_pred), ("prophet", prophet_pred)):
        rel_std = float(np.std(preds)) / actual_std
        if rel_std < 0.1:
            print(f"⚠ {name.upper()} predictions look stuck (rel_std={rel_std:.3f}); down-weighting.")
            weights[name] = weights[name] * 0.05

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
    # Partial rolling bias correction. We only remove a fraction of
    # the lagged trailing residual so the ensemble keeps its natural
    # ups/downs rather than collapsing onto the actual line. Uses
    # only past (shifted) residuals — no look-ahead.
    # --------------------------------------------------------
    bias_alpha = 0.5  # apply 50% of trailing bias only
    bias_window = 30
    residuals = y_true - final_pred
    rolling_bias = (
        pd.Series(residuals).shift(1).rolling(bias_window, min_periods=5).mean()
    )
    rolling_bias = rolling_bias.fillna(0.0).values
    final_pred = final_pred + bias_alpha * rolling_bias
    print(f"Applied partial bias correction (alpha={bias_alpha}, window={bias_window}).")

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
        "Date": merged["Date"].values,
        "Actual": y_true,
        "Dynamic_Prediction": final_pred,
        "Lower_90": lower,
        "Upper_90": upper,
        "Std_Uncertainty": std
    }).to_csv(f"outputs/dynamic_predictions_{horizon}D.csv", index=False)

    print("Dynamic probabilistic predictions saved successfully.")


if __name__ == "__main__":
    main()