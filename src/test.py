import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/final/master_dataset.csv", index_col=0, parse_dates=True)

# Only keep numeric columns
df_numeric = df.select_dtypes(include=['number'])

corr = df_numeric.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()