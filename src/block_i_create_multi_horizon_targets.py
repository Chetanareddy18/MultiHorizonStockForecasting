# block_e_create_multi_horizon_targets.py

import pandas as pd
import numpy as np


def create_targets(df, price_col="Close"):

    # 1-Day
    df["Target_1D"] = df[price_col].shift(-1)

    # 3-Day
    df["Target_3D"] = df[price_col].shift(-3)

    # 7-Day
    df["Target_7D"] = df[price_col].shift(-7)

    # 14-Day
    df["Target_14D"] = df[price_col].shift(-14)

    # 30-Day
    df["Target_30D"] = df[price_col].shift(-30)

    return df


def main():

    print("Loading dataset...")

    df = pd.read_csv("data/final/master_dataset.csv", parse_dates=["Date"])

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Creating multi-horizon targets...")

    df = create_targets(df, price_col="Close")

    # Drop rows where future targets are NaN
    df.dropna(inplace=True)

    print("Targets created successfully.")
    print(df[[
        "Target_1D",
        "Target_3D",
        "Target_7D",
        "Target_14D",
        "Target_30D"
    ]].head())

    # Save updated dataset
    df.to_csv("data/final/master_dataset_multi_horizon.csv", index=False)

    print("\nSaved as: data/final/master_dataset_multi_horizon.csv")


if __name__ == "__main__":
    main()