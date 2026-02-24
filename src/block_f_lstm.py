# block_f_lstm.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "data/final/master_dataset.csv"
HORIZONS = [1, 7, 30]
TIME_STEPS = 30


# -------------------------------------------------
# Create sequences
# -------------------------------------------------
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


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
        print(f"Training LSTM for {horizon}-Day Horizon")
        print("==============================")

        target_column = f"Target_{horizon}D"

        if target_column not in df.columns:
            print(f"{target_column} not found. Skipping...")
            continue

        # Separate target
        target = df[[target_column]]

        # Remove all target columns from features
        target_cols = [col for col in df.columns if "Target_" in col]
        features = df.drop(columns=target_cols)

        # Keep numeric features only
        features = features.select_dtypes(include=[np.number])

        # ---------------------
        # Scale
        # ---------------------
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target)

        # ---------------------
        # Create sequences
        # ---------------------
        X, y = create_sequences(features_scaled, target_scaled, TIME_STEPS)
        sequence_dates = df["Date"].values[TIME_STEPS:]

        # ---------------------
        # Train/Test Split
        # ---------------------
        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        dates_test = sequence_dates[split:]

        print(f"Train shape: {X_train.shape}")
        print(f"Test shape : {X_test.shape}")

        # ---------------------
        # Build Model
        # ---------------------
        model = Sequential([
            Input(shape=(TIME_STEPS, X.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(
            patience=10,
            restore_best_weights=True
        )

        print("Training...")
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )

        # ---------------------
        # Predictions
        # ---------------------
        y_pred_scaled = model.predict(X_test)

        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test_actual = target_scaler.inverse_transform(y_test)

        # ---------------------
        # Evaluation
        # ---------------------
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")

        # ---------------------
        # Save Predictions
        # ---------------------
        output_df = pd.DataFrame({
            "Date": dates_test,
            "Actual": y_test_actual.flatten(),
            "Prediction": y_pred.flatten()
        })

        output_df.to_csv(f"outputs/lstm_predictions_{horizon}D.csv", index=False)

        results.append({
            "Horizon": f"{horizon}D",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/lstm_overall_results.csv", index=False)

    print("\nMulti-Horizon LSTM Completed.")
    print(results_df)


if __name__ == "__main__":
    main()