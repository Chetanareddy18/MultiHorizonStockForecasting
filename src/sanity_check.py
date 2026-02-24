import pandas as pd

df = pd.read_csv("data/final/master_dataset.csv", index_col=0, parse_dates=True)

print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe())
print("\nFirst 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())