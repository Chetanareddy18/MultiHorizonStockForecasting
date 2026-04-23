# Multi-Horizon Stock Forecasting

An end-to-end research project for forecasting the **NIFTY 50 index (`^NSEI`)** at multiple horizons (**1 day, 7 days, 30 days**) by combining classical statistics, deep learning, sentiment, regime detection, uncertainty estimation and risk analysis behind a **Streamlit dashboard**.

> Educational project. Not financial advice.

---

## Highlights

- **Multi-horizon targets** — 1D / 7D / 30D forecasts trained and evaluated separately.
- **Multiple model families** — Baseline ML, LSTM, Temporal Fusion Transformer (TFT), Prophet.
- **Dynamic switching engine** — adaptive ensemble weights driven by horizon, market regime and sentiment.
- **Regime detection** — bull/bear × low/high volatility classification.
- **Behavioural sentiment proxy** — horizon-aware momentum + volatility-fear composite.
- **Uncertainty estimation** — bootstrap residual resampling, MC-Dropout and quantile regression.
- **Professional risk layer** — Volatility, VaR, CVaR, Downside Probability, Max Drawdown, Sharpe-like score.
- **Streamlit dashboard** — interactive, horizon-selectable, with a normalised 0–100 risk score.

---

## Repository structure

```
MultiHorizonStockForecasting/
├── data/                # Raw + processed + final + predictions (gitignored)
├── new_data/            # External / out-of-sample dataset (gitignored)
├── outputs/             # Model predictions, risk reports, sentiment scores (gitignored)
├── lightning_logs/      # PyTorch-Lightning training logs (gitignored)
├── requirements.txt
└── src/
    ├── block_a_market.py                        # Download NIFTY OHLCV + returns
    ├── block_b_technical.py                     # Technical indicators
    ├── block_c_sentiment.py                     # External sentiment block
    ├── block_d_macro.py                         # Macro-economic features
    ├── block_e_baseline_modeling.py             # Baseline ML models
    ├── block_f_lstm.py                          # LSTM forecaster
    ├── block_g_transformer.py                   # Temporal Fusion Transformer
    ├── block_h_prophet.py                       # Prophet forecaster
    ├── block_i_create_multi_horizon_targets.py  # Build 1D / 7D / 30D targets
    ├── block_j_dynamic_switching_engine.py      # Adaptive ensemble + bootstrap CI
    ├── block_k_regime_detection.py              # Bull/Bear × Low/High vol
    ├── block_l_sentiment.py                     # Horizon-aware behavioural sentiment
    ├── block_m_uncertainty_estimation.py        # MC-Dropout / Quantile / Bootstrap
    ├── block_n_risk_analysis.py                 # VaR, CVaR, Drawdown, Sharpe-like
    ├── block_o_comparsion.py                    # Cross-model benchmark table
    ├── another_data.py                          # Out-of-sample evaluation on new data
    ├── consolidate.py                           # Merge feature blocks → master dataset
    ├── sanity_check.py                          # Dataset sanity checks
    ├── test.py                                  # Quick test harness
    └── app.py                                   # Streamlit dashboard
```

---

## Setup

```powershell
# 1. Create / activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# Most blocks also need:
pip install scikit-learn streamlit prophet torch pytorch-lightning pytorch-forecasting
```

---

## Pipeline (run in order)

All commands are run from the **repository root** so that the relative `data/`, `outputs/` and `new_data/` paths resolve correctly.

```powershell
# 1. Feature engineering blocks
python src/block_a_market.py
python src/block_b_technical.py
python src/block_c_sentiment.py
python src/block_d_macro.py
python src/consolidate.py
python src/block_i_create_multi_horizon_targets.py

# 2. Train / generate per-horizon predictions (1, 7, 30)
python src/block_e_baseline_modeling.py
python src/block_f_lstm.py
python src/block_g_transformer.py
python src/block_h_prophet.py

# 3. Sentiment + dynamic switching + risk for each horizon
foreach ($h in 1, 7, 30) {
    python src/block_l_sentiment.py $h
    python src/block_j_dynamic_switching_engine.py $h
    python src/block_n_risk_analysis.py $h
}

# 4. Cross-model comparison
python src/block_o_comparsion.py

# 5. Streamlit dashboard
streamlit run src/app.py
```

---

## Outputs produced

| File pattern                              | Description                                        |
| ----------------------------------------- | -------------------------------------------------- |
| `outputs/lstm_predictions_{H}D.csv`       | LSTM per-horizon predictions                       |
| `outputs/tft_predictions_{H}D.csv`        | TFT per-horizon predictions                        |
| `outputs/prophet_predictions_{H}D.csv`    | Prophet per-horizon predictions                    |
| `outputs/dynamic_predictions_{H}D.csv`    | Adaptive ensemble + 90% CI + uncertainty std       |
| `outputs/sentiment_score_{H}D.csv`        | Horizon-aware sentiment score                      |
| `outputs/risk_report_{H}D.csv`            | Volatility / VaR / CVaR / Downside / Drawdown      |
| `outputs/model_comparison_results.csv`    | Cross-model MAE / RMSE / MAPE / Directional Acc.   |

---

## Dashboard notes

- The dashboard expects the `outputs/` folder to be populated by the pipeline above.
- The risk badge in the dashboard shows a **normalised 0–100 score** built from annualised volatility, downside probability and max drawdown — not the raw daily-volatility number.
- The forecast line includes a small **trend-aware stabilisation step**; toggle “Show Raw Data” to inspect the underlying values.

---

## Known limitations

- Sentiment block (`block_l_sentiment.py`) is a behavioural proxy derived from price action — it is **not** real news sentiment.
- TFT and Prophet require their own optional dependencies; install them only if you intend to retrain those models.
- All thresholds in regime / sentiment / risk are heuristic and tuned for `^NSEI` on the 2015–2024 window.

---

## License

Educational use only.
