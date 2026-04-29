# =============================================================
# predict_new_data.py
# Standalone predictor for ANY ticker CSV in new_data/etfs/.
# Used by the dashboard "Test on New Data" section so users can
# verify the project predicts both UP and DOWN moves on different
# instruments (the main NIFTY pipeline only sees one historical
# window and thus shows one direction at the latest test point).
# =============================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def _build_lag_features(series: pd.Series, lags=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ret_1"] = df["y"].pct_change(1)
    df["ret_5"] = df["y"].pct_change(5)
    df["roll_mean_10"] = df["y"].rolling(10).mean()
    df["roll_std_10"] = df["y"].rolling(10).std()
    df["roll_mean_30"] = df["y"].rolling(30).mean()
    return df.dropna()


def predict_ticker(csv_path: str, horizon: int = 1, as_of_date=None) -> dict:
    """Train a Ridge regression on lag features of the Close price and
    walk-forward predict the held-out tail. Returns metrics + a frame
    with Date / Actual / Prediction.

    If `as_of_date` is provided (string or datetime), the dataset is
    truncated to that date so users can simulate predictions from any
    historical point. The held-out tail (last 20% of the truncated
    history) is what the dashboard displays.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        df = df[df["Date"] <= cutoff].reset_index(drop=True)
        if len(df) < 150:
            raise ValueError(
                f"Only {len(df)} rows available up to {cutoff.date()}; need >=150."
            )

    close_col = "Close" if "Close" in df.columns else df.select_dtypes("number").columns[0]
    close = df[close_col].astype(float)
    dates = df["Date"]

    # Predict price `horizon` days ahead
    target = close.shift(-horizon)

    feat = _build_lag_features(close)
    feat["target"] = target.loc[feat.index]
    feat = feat.dropna()

    if len(feat) < 100:
        raise ValueError(f"Not enough rows after feature build: {len(feat)}")

    X = feat.drop(columns=["target", "y"]).values
    y = feat["target"].values
    feat_dates = dates.loc[feat.index].values

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    d_te = feat_dates[split:]
    actual_today = feat["y"].values[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_pred_adj = model.predict(X_te)

    mae = float(mean_absolute_error(y_te, y_pred_adj))
    mape = float(mean_absolute_percentage_error(y_te, y_pred_adj) * 100)

    # Direction accuracy: did we predict the sign of the move correctly?
    actual_dir = np.sign(y_te - actual_today)
    pred_dir = np.sign(y_pred_adj - actual_today)
    dir_acc = float((actual_dir == pred_dir).mean() * 100)

    last_actual = float(actual_today[-1])
    last_pred = float(y_pred_adj[-1])
    change_pct = (last_pred - last_actual) / last_actual * 100

    out = pd.DataFrame({
        "Date": pd.to_datetime(d_te),
        "Actual_Today": actual_today,
        "Actual_Future": y_te,
        "Prediction": y_pred_adj,
    })

    return {
        "ticker": os.path.splitext(os.path.basename(csv_path))[0],
        "horizon": horizon,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "last_actual": last_actual,
        "last_pred": last_pred,
        "change_pct": float(change_pct),
        "frame": out,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "new_data/etfs/AAXJ.csv"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    r = predict_ticker(path, h)
    print(f"{r['ticker']} | horizon={h}D")
    print(f"  Last actual: {r['last_actual']:.2f}")
    print(f"  Forecast   : {r['last_pred']:.2f}  ({r['change_pct']:+.2f}%)")
    print(f"  MAE  : {r['mae']:.2f}")
    print(f"  MAPE : {r['mape']:.2f}%")
    print(f"  Dir Acc: {r['directional_accuracy']:.2f}%")
