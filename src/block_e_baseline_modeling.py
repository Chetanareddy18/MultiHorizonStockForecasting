# block_e_baseline_modeling.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/final/master_dataset.csv"
HORIZONS = [1, 7, 30]   # Short, Medium, Long


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    os.makedirs("outputs", exist_ok=True)

    results = []

    for horizon in HORIZONS:

        print(f"\n==============================")
        print(f"Training for {horizon}-Day Horizon")
        print(f"==============================")

        TARGET = f"Target_{horizon}D"

        if TARGET not in df.columns:
            print(f"{TARGET} not found in dataset. Skipping...")
            continue

        # Remove all target columns from features
        target_cols = [col for col in df.columns if "Target_" in col]

        X = df.drop(columns=target_cols)
        y = df[TARGET]

        # Chronological Split (70/30)
        split_index = int(len(df) * 0.7)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(f"Train size: {len(X_train)}")
        print(f"Test size : {len(X_test)}")

        # Model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        print("Training RandomForest...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae, rmse, mape = evaluate_model(y_test, y_pred)

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.4f}%")

        # Save predictions
        pred_df = pd.DataFrame({
            "Date": y_test.index,
            "Actual": y_test.values,
            "Prediction": y_pred
        })

        pred_df.to_csv(f"outputs/ml_predictions_{horizon}D.csv")

        results.append({
            "Horizon": f"{horizon}D",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/ml_overall_results.csv", index=False)

    print("\nMulti-Horizon Baseline Completed.")
    print("\nOverall Results:")
    print(results_df)


if __name__ == "__main__":
    main()