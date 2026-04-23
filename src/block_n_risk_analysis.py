# ============================================================
# BLOCK N: PROFESSIONAL RISK ANALYSIS LAYER
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
from math import sqrt


# ============================================================
# VALUE AT RISK (VaR)
# ============================================================
def calculate_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)


# ============================================================
# CONDITIONAL VAR (Expected Shortfall)
# ============================================================
def calculate_cvar(returns, confidence=0.95):
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


# ============================================================
# RISK ANALYSIS
# ============================================================
def perform_risk_analysis(horizon):

    path = f"outputs/dynamic_predictions_{horizon}D.csv"

    if not os.path.exists(path):
        print("Dynamic prediction file not found.")
        return

    df = pd.read_csv(path)

    if "Actual" not in df.columns:
        print("Actual column not found in prediction file.")
        return

    # --------------------------------------------------------
    # Use ACTUAL RETURNS (Financially Correct)
    # --------------------------------------------------------
    df["Actual_Return"] = df["Actual"].pct_change()
    df = df.dropna()

    returns = df["Actual_Return"].values

    # --------------------------------------------------------
    # Core Risk Metrics
    # --------------------------------------------------------
    volatility = np.std(returns)
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    downside_prob = np.mean(returns < 0)
    sharpe_like = np.mean(returns) / (volatility + 1e-8)
    max_drawdown = calculate_max_drawdown(pd.Series(returns))

    # --------------------------------------------------------
    # Annualization (Horizon Adjusted)
    # --------------------------------------------------------
    if horizon == 1:
        annual_vol = volatility * sqrt(252)
    else:
        annual_vol = volatility * sqrt(252 / horizon)

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------
    print("\n========== PROFESSIONAL RISK ANALYSIS ==========")
    print(f"Horizon: {horizon}D")
    print(f"Daily Volatility: {volatility:.6f}")
    print(f"Annualized Volatility: {annual_vol:.6f}")
    print(f"VaR (95%): {var_95:.6f}")
    print(f"CVaR (95%): {cvar_95:.6f}")
    print(f"Downside Probability: {downside_prob:.4f}")
    print(f"Risk-Adjusted Score: {sharpe_like:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.6f}")

    # --------------------------------------------------------
    # Save Risk Report
    # --------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    risk_df = pd.DataFrame({
        "Horizon": [horizon],
        "Daily_Volatility": [volatility],
        "Annualized_Volatility": [annual_vol],
        "VaR_95": [var_95],
        "CVaR_95": [cvar_95],
        "Downside_Probability": [downside_prob],
        "Risk_Adjusted_Score": [sharpe_like],
        "Maximum_Drawdown": [max_drawdown]
    })

    risk_df.to_csv(f"outputs/risk_report_{horizon}D.csv", index=False)

    print("Professional risk report saved successfully.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    else:
        horizon = 1

    perform_risk_analysis(horizon)