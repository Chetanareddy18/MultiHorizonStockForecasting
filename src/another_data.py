import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Load new dataset
# ----------------------------

print("Loading new dataset...")

new_data = pd.read_csv("new_data/raw/new_stock.csv")

# ensure sorted by date
new_data = new_data.sort_values("Date")

actual_prices = new_data["Close"].values


# ----------------------------
# Evaluation Metrics
# ----------------------------

def evaluate(actual, pred):

    mae = mean_absolute_error(actual, pred)

    rmse = np.sqrt(mean_squared_error(actual, pred))

    mape = np.mean(np.abs((actual - pred) / actual)) * 100

    direction = np.mean(
        np.sign(np.diff(actual)) == np.sign(np.diff(pred))
    ) * 100

    return mae, rmse, mape, direction


# ----------------------------
# Load prediction files
# (use horizon-specific files; default to 1D for the new dataset comparison)
# ----------------------------

HORIZON = "1D"

files = {
    "LSTM":    f"outputs/lstm_predictions_{HORIZON}.csv",
    "TFT":     f"outputs/tft_predictions_{HORIZON}.csv",
    "Prophet": f"outputs/prophet_predictions_{HORIZON}.csv"
}

results = []


for model, path in files.items():

    print(f"Evaluating {model}...")

    df = pd.read_csv(path)

    # Existing prediction CSVs use the column name "Prediction".
    if "Prediction" in df.columns:
        pred = df["Prediction"].values
    elif "prediction" in df.columns:
        pred = df["prediction"].values
    else:
        raise KeyError(
            f"No 'Prediction' column found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # match lengths
    min_len = min(len(actual_prices), len(pred))

    actual = actual_prices[-min_len:]
    pred = pred[-min_len:]

    mae, rmse, mape, da = evaluate(actual, pred)

    results.append({
        "Model": model,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": da
    })


# ----------------------------
# Results
# ----------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison on New Dataset\n")
print(results_df)

# save results
results_df.to_csv("outputs/new_dataset_results.csv", index=False)

print("\nResults saved to outputs/new_dataset_results.csv")