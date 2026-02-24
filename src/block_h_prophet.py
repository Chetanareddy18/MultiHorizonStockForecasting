# block_h_prophet.py

import pandas as pd
import numpy as np
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/final/master_dataset.csv"
HORIZONS = [1, 7, 30]   # Add 14, 40 if available


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs("outputs", exist_ok=True)

    results = []

    for horizon in HORIZONS:

        print("\n==============================")
        print(f"Training Prophet for {horizon}-Day Horizon")
        print("==============================")

        target_col = f"Target_{horizon}D"

        if target_col not in df.columns:
            print(f"{target_col} not found. Skipping...")
            continue

        prophet_df = pd.DataFrame({
            "ds": df["Date"],
            "y": df[target_col]
        })

        # 80/20 split
        split_index = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_index]
        test_df = prophet_df.iloc[split_index:]

        print(f"Train size: {len(train_df)}")
        print(f"Test size : {len(test_df)}")

        # Model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )

        model.fit(train_df)

        # Forecast only test period
        future = model.make_future_dataframe(
            periods=len(test_df),
            freq="B"
        )

        forecast = model.predict(future)

        forecast_test = forecast.iloc[split_index:].copy()

        y_true = test_df["y"].values
        y_pred = forecast_test["yhat"].values

        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        mae, rmse, mape = evaluate(y_true, y_pred)

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")

        # Save predictions per horizon
        pd.DataFrame({
            "Date": forecast_test["ds"].values[:min_len],
            "Actual": y_true,
            "Prediction": y_pred
        }).to_csv(f"outputs/prophet_predictions_{horizon}D.csv", index=False)

        results.append({
            "Horizon": f"{horizon}D",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/prophet_overall_results.csv", index=False)

    print("\nMulti-Horizon Prophet Completed.")
    print(results_df)


if __name__ == "__main__":
    main()