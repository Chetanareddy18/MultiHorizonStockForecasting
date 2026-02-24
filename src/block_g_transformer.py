# block_f_tft_model.py

import pandas as pd
import numpy as np
import torch
import os

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


DATA_PATH = "data/final/master_dataset.csv"
HORIZONS = [1, 7, 30]   # Add 14, 40 if available


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["time_idx"] = np.arange(len(df))
    df["series"] = "nifty"

    os.makedirs("outputs", exist_ok=True)

    results = []

    for horizon in HORIZONS:

        print("\n==============================")
        print(f"Training TFT for {horizon}-Day Horizon")
        print("==============================")

        target = f"Target_{horizon}D"

        if target not in df.columns:
            print(f"{target} not found. Skipping...")
            continue

        # -------------------------------
        # Train/Test split (70/30)
        # -------------------------------
        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # -------------------------------
        # Feature selection
        # -------------------------------
        known_reals = ["time_idx"]

        target_cols = [col for col in df.columns if "Target_" in col]

        unknown_reals = [
            col for col in df.columns
            if col not in ["Date", "series", "time_idx"]
            and col not in target_cols
        ]

        max_encoder_length = 60
        max_prediction_length = 1   # Single-step but different horizon targets

        # -------------------------------
        # Dataset
        # -------------------------------
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target,
            group_ids=["series"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, test_df)

        train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

        # -------------------------------
        # Model
        # -------------------------------
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.0005,
            hidden_size=32,
            attention_head_size=8,
            dropout=0.2,
            hidden_continuous_size=32,
            loss=QuantileLoss(),
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=5)

        trainer = Trainer(
            max_epochs=40,
            accelerator="auto",
            callbacks=[early_stop],
            enable_progress_bar=True,
        )

        print("Training...")
        trainer.fit(tft, train_loader, val_loader)

        # -------------------------------
        # Predictions
        # -------------------------------
        raw_predictions = tft.predict(val_loader)
        y_pred = raw_predictions.numpy().flatten()

        y_true = test_df[target].values[-len(y_pred):]

        # -------------------------------
        # Evaluation
        # -------------------------------
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")

        # -------------------------------
        # Save predictions
        # -------------------------------
        pd.DataFrame({
            "Actual": y_true,
            "Prediction": y_pred
        }).to_csv(f"outputs/tft_predictions_{horizon}D.csv", index=False)

        results.append({
            "Horizon": f"{horizon}D",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    # -------------------------------
    # Save overall results
    # -------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/tft_overall_results.csv", index=False)

    print("\nMulti-Horizon TFT Completed.")
    print(results_df)


if __name__ == "__main__":
    main()