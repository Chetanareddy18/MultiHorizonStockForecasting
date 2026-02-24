import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


def compute_sentiment(news_csv="data/raw/news.csv"):

    df = pd.read_csv(news_csv)

    sia = SentimentIntensityAnalyzer()

    df["Sentiment"] = df["headline"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    return df["Sentiment"].mean()


if __name__ == "__main__":
    score = compute_sentiment()
    print("Average Sentiment:", score)