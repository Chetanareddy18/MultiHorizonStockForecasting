import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# ---------------------------------------------------
# Project Root
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
outputs_path = BASE_DIR / "outputs"


# ---------------------------------------------------
# Function to Load Data
# ---------------------------------------------------
def load_file(file_name):

    df = pd.read_csv(outputs_path / file_name)

    y_true = df["Actual"]
    y_pred = df["Prediction"]

    return y_true, y_pred


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mae, rmse, mape


def directional_accuracy(y_true, y_pred):

    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    return np.mean(actual_dir == pred_dir) * 100


# ---------------------------------------------------
# Model Files
# ---------------------------------------------------
model_files = {

    "LSTM": [
        "lstm_predictions_1D.csv",
        "lstm_predictions_7D.csv",
        "lstm_predictions_30D.csv"
    ],

    "TFT": [
        "tft_predictions_1D.csv",
        "tft_predictions_7D.csv",
        "tft_predictions_30D.csv"
    ],

    "Prophet": [
        "prophet_predictions_1D.csv",
        "prophet_predictions_7D.csv",
        "prophet_predictions_30D.csv"
    ]
}

horizons = ["1 Day", "7 Days", "30 Days"]


# ---------------------------------------------------
# Compare Models
# ---------------------------------------------------
results = []

for model_name, files in model_files.items():

    for i, file in enumerate(files):

        y_true, y_pred = load_file(file)

        mae, rmse, mape = evaluate(y_true, y_pred)
        direction = directional_accuracy(y_true, y_pred)

        results.append([
            model_name,
            horizons[i],
            mae,
            rmse,
            mape,
            direction
        ])


# ---------------------------------------------------
# Results Table
# ---------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Horizon",
        "MAE",
        "RMSE",
        "MAPE (%)",
        "Directional Accuracy (%)"
    ]
)

print("\nModel Comparison Results\n")
print(results_df)


# ---------------------------------------------------
# Save Output
# ---------------------------------------------------
results_df.to_csv(outputs_path / "model_comparison_results.csv", index=False)

print("\nResults saved to outputs/model_comparison_results.csv")