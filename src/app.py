import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Smart Market Forecast Dashboard")
st.markdown("### Simple AI-Based Market Forecasting with Risk Analysis")

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("⚙ Dashboard Settings")
horizon = st.sidebar.selectbox("Forecast for next (Days)", [1, 7, 30])
show_raw = st.sidebar.checkbox("Show Raw Data")
show_ai = st.sidebar.checkbox("Show How AI Works")
nifty_scrub = st.sidebar.checkbox(
    "Scrub NIFTY history (any date)",
    help=(
        "Re-runs a fast lag-feature model on NIFTY truncated to any "
        "historical date. Useful to see how the forecast direction "
        "changes across bull / bear periods."
    ),
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
# Resolve outputs relative to the project root so the dashboard works
# regardless of the current working directory streamlit was launched from.
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
if not os.path.isdir(OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
BASE_DIR = PROJECT_DIR

pred_path = os.path.join(OUTPUT_DIR, f"dynamic_predictions_{horizon}D.csv")
risk_path = os.path.join(OUTPUT_DIR, f"risk_report_{horizon}D.csv")

# -------------------------------------------------
# Check Prediction File
# -------------------------------------------------
if not os.path.exists(pred_path):
    st.error("⚠ Forecast data not found. Please run forecasting first.")
    st.stop()

df = pd.read_csv(pred_path)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
latest_actual = df["Actual"].iloc[-1]
latest_pred = df["Dynamic_Prediction"].iloc[-1]
change_pct = ((latest_pred - latest_actual) / latest_actual) * 100

# -------------------------------------------------
# Headline forecast: replace the cached ensemble's stale tail with a
# fresh lag-feature prediction on NIFTY's latest data so the summary
# reflects current market state instead of being permanently anchored
# to the last training-window date.
# -------------------------------------------------
NIFTY_PATH = os.path.join(BASE_DIR, "data", "final", "master_dataset.csv")
headline_source = "ensemble"
if os.path.exists(NIFTY_PATH):
    try:
        import importlib, sys as _sys
        _sys.path.insert(0, SCRIPT_DIR)
        _pnd = importlib.import_module("predict_new_data")
        _ndf = pd.read_csv(NIFTY_PATH, parse_dates=["Date"]).sort_values("Date")
        _tmp = os.path.join(OUTPUT_DIR, "_nifty_tmp.csv")
        _ndf[["Date", "Close"]].to_csv(_tmp, index=False)
        _r = _pnd.predict_ticker(_tmp, horizon=int(horizon))
        latest_actual = _r["last_actual"]
        latest_pred = _r["last_pred"]
        change_pct = _r["change_pct"]
        headline_source = "lag-feature (live)"
    except Exception:
        pass

# -------------------------------------------------
# 🟢 BIG SUMMARY SECTION
# -------------------------------------------------
st.markdown("## 🧾 Overall Market Summary")

col1, col2, col3 = st.columns(3)

# Direction
if change_pct > 1:
    col1.success("📈 Market Likely to Go UP")
elif change_pct < -1:
    col1.error("📉 Market Likely to Go DOWN")
else:
    col1.info("➡ Market Likely to Stay Stable")

# Risk
risk_value = None
risk_df = None
if os.path.exists(risk_path):

    risk_df = pd.read_csv(risk_path)

    # Build a normalised 0-100 risk score from the available risk metrics.
    # Higher annualised volatility, higher downside probability and deeper
    # max drawdown all push the score up.
    try:
        ann_vol = float(risk_df.get("Annualized_Volatility", pd.Series([0.0])).iloc[0])
        downside = float(risk_df.get("Downside_Probability", pd.Series([0.0])).iloc[0])
        max_dd = float(risk_df.get("Maximum_Drawdown", pd.Series([0.0])).iloc[0])
    except Exception:
        ann_vol, downside, max_dd = 0.0, 0.0, 0.0

    # Each component is mapped to roughly 0..1 then averaged and scaled to 0..100.
    vol_component = min(ann_vol / 0.40, 1.0)          # 40% annual vol == max
    downside_component = min(downside / 0.60, 1.0)    # 60% down days == max
    drawdown_component = min(abs(max_dd) / 0.40, 1.0) # 40% drawdown == max

    risk_value = float(
        np.clip((vol_component + downside_component + drawdown_component) / 3 * 100, 0, 100)
    )

    if risk_value < 30:
        col2.success(f"🟢 Low Risk ({risk_value:.0f}/100)")
    elif risk_value < 70:
        col2.warning(f"🟡 Medium Risk ({risk_value:.0f}/100)")
    else:
        col2.error(f"🔴 High Risk ({risk_value:.0f}/100)")

else:
    col2.info("Risk Data Not Available")

# Confidence
confidence = max(0, 100 - abs(change_pct))

if confidence > 80:
    col3.success("High Confidence")
elif confidence > 50:
    col3.warning("Moderate Confidence")
else:
    col3.error("Low Confidence")

# -------------------------------------------------
# Explanation Section
# -------------------------------------------------
st.markdown("## 🧠 What Does This Mean?")

if change_pct > 1:
    st.write(f"For the next {horizon} days, the market may rise by about {change_pct:.2f}%.")
elif change_pct < -1:
    st.write(f"For the next {horizon} days, the market may fall by about {abs(change_pct):.2f}%.")
else:
    st.write(f"For the next {horizon} days, the market is expected to remain stable.")

# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
st.markdown("## 📌 Key Numbers")

k1, k2, k3 = st.columns(3)

k1.metric("Latest Actual Price", f"{latest_actual:.2f}")
k2.metric("Forecasted Price", f"{latest_pred:.2f}", f"{change_pct:.2f}%")
k3.metric("Forecast Horizon", f"{horizon} Days")

# -------------------------------------------------
# Price Trend Chart
# -------------------------------------------------
st.markdown("## 📈 Price Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["Actual"], label="Actual Price")
ax.plot(df["Dynamic_Prediction"], label="AI Forecast", linestyle="--")

if "Lower_90" in df.columns and "Upper_90" in df.columns:

    ax.fill_between(
        df.index,
        df["Lower_90"],
        df["Upper_90"],
        alpha=0.2,
        label="90% Confidence Band"
    )

ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Risk Details
# -------------------------------------------------
if risk_value is not None:

    st.markdown("## ⚠ Risk Details")

    st.dataframe(risk_df)

    if risk_value < 30:
        st.success("Market risk is currently low.")
    elif risk_value < 70:
        st.warning("Market risk is moderate.")
    else:
        st.error("Market risk is high. Be cautious.")

# -------------------------------------------------
# Market Volatility (computed from actual returns)
# -------------------------------------------------
if "Actual" in df.columns and len(df) > 21:

    st.markdown("## 🌪 Market Volatility")

    returns = df["Actual"].pct_change()
    # Annualised rolling volatility expressed as a percentage so the chart
    # is readable (raw daily std is ~0.008 which looks like zero).
    rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(rolling_vol.values, color="#d62728")
    ax2.set_title("20-Day Rolling Annualised Volatility (%)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Annualised Volatility (%)")
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

    last_vol = float(rolling_vol.iloc[-1]) if not pd.isna(rolling_vol.iloc[-1]) else 0.0
    mean_vol = float(rolling_vol.mean()) if not pd.isna(rolling_vol.mean()) else 0.0

    st.caption(f"Latest: {last_vol:.2f}%   |   Period average: {mean_vol:.2f}%")

    if last_vol > mean_vol:
        st.warning("Market is more volatile than its recent average.")
    else:
        st.success("Market is relatively stable.")

# -------------------------------------------------
# Sentiment Section
# -------------------------------------------------
if os.path.exists(OUTPUT_DIR):

    # Only pick horizon-specific sentiment files (e.g. sentiment_score_7D.csv).
    sentiment_files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("sentiment_score_") and f.endswith("D.csv")
    ]

else:
    sentiment_files = []

if sentiment_files:

    st.markdown("## 📰 Market Sentiment")

    sentiments = []
    horizons_list = []

    for file in sentiment_files:

        s_df = pd.read_csv(os.path.join(OUTPUT_DIR, file))

        sentiments.append(s_df["Sentiment"].values[0])
        horizons_list.append(file.split("_")[-1].replace("D.csv", ""))

    fig3, ax3 = plt.subplots()

    ax3.bar(horizons_list, sentiments)

    ax3.set_xlabel("Forecast Horizon (Days)")
    ax3.set_ylabel("Sentiment Score")

    st.pyplot(fig3)

    avg_sentiment = np.mean(sentiments)

    if avg_sentiment > 0:
        st.success("Overall news sentiment is positive.")
    else:
        st.error("Overall news sentiment is negative.")

# -------------------------------------------------
# Download Section
# -------------------------------------------------
st.markdown("## 📥 Download Forecast Data")

st.download_button(
    label="Download Forecast CSV",
    data=df.to_csv(index=False),
    file_name=f"forecast_{horizon}D.csv",
    mime="text/csv"
)

# -------------------------------------------------
# Raw Data
# -------------------------------------------------
if show_raw:

    st.markdown("## 📂 Raw Data")

    st.dataframe(df)

# -------------------------------------------------
# AI Explanation
# -------------------------------------------------
if show_ai:

    st.markdown("## 🤖 How The AI Works (Simple Explanation)")

    st.write("""
This system combines multiple prediction methods:

• Deep learning models  
• Statistical forecasting  
• Sentiment analysis  
• Volatility tracking  

The system dynamically adjusts predictions based on:

• Market trend  
• Market risk  
• News sentiment  
• Market stability  

The goal is to produce balanced and adaptive forecasts.
""")

# -------------------------------------------------
# NIFTY: Scrub through history (proves the model can predict UP and DOWN)
# -------------------------------------------------
if nifty_scrub:
    st.markdown("---")
    st.markdown("## 🕰 NIFTY Historical Scrubber")
    st.caption(
        "The main forecast above is anchored to the last test date in the "
        "stored prediction file. Use this scrubber to re-run a fast "
        "lag-feature model on NIFTY truncated to any date — bull periods "
        "produce UP forecasts, bear periods produce DOWN forecasts."
    )

    NIFTY_PATH = os.path.join(BASE_DIR, "data", "final", "master_dataset.csv")

    if os.path.exists(NIFTY_PATH):
        try:
            n_df = pd.read_csv(NIFTY_PATH, parse_dates=["Date"]).sort_values("Date")
            n_min = n_df["Date"].min().date()
            n_max = n_df["Date"].max().date()

            cs1, cs2 = st.columns([3, 1])
            default_n = (
                pd.Timestamp("2017-06-30").date()
                if n_min <= pd.Timestamp("2017-06-30").date() <= n_max
                else n_max
            )
            n_as_of = cs1.slider(
                f"As-of date  ({n_min} → {n_max})",
                min_value=n_min, max_value=n_max, value=default_n,
                key="nifty_scrub_date",
            )
            n_horizon = cs2.selectbox(
                "Horizon (Days)", [1, 7, 30], key="nifty_scrub_horizon"
            )
            n_run = st.button("Run NIFTY Scrub", use_container_width=True)

            if n_run:
                import importlib, sys as _sys
                _sys.path.insert(0, SCRIPT_DIR)
                pnd = importlib.import_module("predict_new_data")
                tmp = n_df[["Date", "Close"]].copy()
                tmp_path = os.path.join(OUTPUT_DIR, "_nifty_tmp.csv")
                tmp.to_csv(tmp_path, index=False)

                with st.spinner(f"Predicting NIFTY @ {n_as_of} ..."):
                    nr = pnd.predict_ticker(
                        tmp_path, horizon=int(n_horizon), as_of_date=n_as_of
                    )

                k1, k2, k3 = st.columns(3)
                k1.metric("Last NIFTY", f"{nr['last_actual']:.2f}")
                k2.metric(
                    "Forecast", f"{nr['last_pred']:.2f}",
                    f"{nr['change_pct']:+.2f}%",
                )
                k3.metric("Dir. Acc.", f"{nr['directional_accuracy']:.1f}%")

                if nr["change_pct"] > 0.5:
                    st.success(
                        f"📈 NIFTY expected UP by ~{nr['change_pct']:+.2f}% "
                        f"({n_horizon}D from {n_as_of})."
                    )
                elif nr["change_pct"] < -0.5:
                    st.error(
                        f"📉 NIFTY expected DOWN by ~{abs(nr['change_pct']):.2f}% "
                        f"({n_horizon}D from {n_as_of})."
                    )
                else:
                    st.info(f"➡ NIFTY expected ~flat ({n_horizon}D from {n_as_of}).")

                fr = nr["frame"].tail(150)
                fig_h, ax_h = plt.subplots(figsize=(11, 4))
                ax_h.plot(fr["Date"], fr["Actual_Future"], label="Actual",
                          color="#1f77b4")
                ax_h.plot(fr["Date"], fr["Prediction"], label="Predicted",
                          color="#ff7f0e", linestyle="--")
                ax_h.set_title(
                    f"NIFTY @ {n_as_of}  "
                    f"(MAE={nr['mae']:.2f}, MAPE={nr['mape']:.2f}%)"
                )
                ax_h.legend()
                ax_h.grid(alpha=0.3)
                st.pyplot(fig_h)
        except Exception as exc:
            st.error(f"NIFTY scrubber failed: {exc}")
    else:
        st.warning("data/final/master_dataset.csv not found.")

# -------------------------------------------------
# Final Advice
# -------------------------------------------------
st.markdown("## 📌 Simple Final Interpretation")

if change_pct > 1 and (risk_value is not None and risk_value < 50):
    st.success("Outlook looks positive with manageable risk.")
elif change_pct < -1 and (risk_value is not None and risk_value > 60):
    st.error("Downside risk appears significant.")
else:
    st.info("Market outlook is mixed. Monitor carefully.")

# -------------------------------------------------
# Test on New Data (any ETF/ticker from new_data/etfs/)
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Test Model on Other Markets")
st.caption(
    "The main forecast above is trained only on NIFTY 50. Use this section to "
    "verify the modelling pipeline on any other ticker — different instruments "
    "show different up/down forecasts depending on their own recent trend."
)

NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data", "etfs")

if os.path.isdir(NEW_DATA_DIR):
    available = sorted(
        f for f in os.listdir(NEW_DATA_DIR) if f.endswith(".csv")
    )

    if available:
        col_a, col_b, col_c = st.columns([3, 1, 1])
        ticker_file = col_a.selectbox(
            "Pick a ticker CSV",
            available,
            index=available.index("AAXJ.csv") if "AAXJ.csv" in available else 0,
        )
        new_horizon = col_b.selectbox(
            "Horizon (Days)", [1, 7, 30], key="new_data_horizon"
        )
        run_btn = col_c.button("Run Prediction", use_container_width=True)

        # Optional historical "as-of" cutoff so users can simulate
        # predictions from any earlier point (the bundled ETF CSVs all
        # end on 2020-04-01 during the COVID crash, which biases every
        # forecast downward — picking an earlier date such as 2018-06
        # demonstrates UP forecasts in bullish periods).
        try:
            sample_df = pd.read_csv(
                os.path.join(NEW_DATA_DIR, ticker_file), parse_dates=["Date"]
            )
            min_d = sample_df["Date"].min().date()
            max_d = sample_df["Date"].max().date()
            default_d = max_d
            as_of = st.slider(
                f"Run prediction as of date  (data range {min_d} → {max_d})",
                min_value=min_d, max_value=max_d, value=default_d,
            )
        except Exception:
            as_of = None

        if run_btn:
            try:
                # Lazy import so failure here doesn't break the main page.
                import importlib, sys as _sys
                _sys.path.insert(0, SCRIPT_DIR)
                pnd = importlib.import_module("predict_new_data")
                with st.spinner(f"Training on {ticker_file} ..."):
                    res = pnd.predict_ticker(
                        os.path.join(NEW_DATA_DIR, ticker_file),
                        horizon=int(new_horizon),
                        as_of_date=as_of,
                    )

                m1, m2, m3 = st.columns(3)
                m1.metric("Last Actual", f"{res['last_actual']:.2f}")
                m2.metric(
                    "Forecast", f"{res['last_pred']:.2f}", f"{res['change_pct']:+.2f}%"
                )
                m3.metric("Directional Acc.", f"{res['directional_accuracy']:.1f}%")

                if res["change_pct"] > 0.5:
                    st.success(
                        f"📈 Model expects **{res['ticker']}** to go UP "
                        f"by ~{res['change_pct']:+.2f}% over the next {new_horizon} day(s)."
                    )
                elif res["change_pct"] < -0.5:
                    st.error(
                        f"📉 Model expects **{res['ticker']}** to go DOWN "
                        f"by ~{abs(res['change_pct']):.2f}% over the next {new_horizon} day(s)."
                    )
                else:
                    st.info(
                        f"➡ Model expects **{res['ticker']}** to stay roughly flat."
                    )

                fr = res["frame"].tail(120)
                fig_n, ax_n = plt.subplots(figsize=(11, 4))
                ax_n.plot(fr["Date"], fr["Actual_Future"], label="Actual", color="#1f77b4")
                ax_n.plot(
                    fr["Date"], fr["Prediction"], label="Predicted",
                    color="#ff7f0e", linestyle="--",
                )
                ax_n.set_title(
                    f"{res['ticker']} — last 120 test points "
                    f"(MAE={res['mae']:.2f}, MAPE={res['mape']:.2f}%)"
                )
                ax_n.legend()
                ax_n.grid(alpha=0.3)
                st.pyplot(fig_n)

                with st.expander("Show prediction table"):
                    st.dataframe(fr.reset_index(drop=True))

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    else:
        st.warning("No CSV files found in new_data/etfs/.")
else:
    st.warning("new_data/etfs/ directory not found.")

st.caption("This dashboard provides AI-based forecasts for educational purposes only.")