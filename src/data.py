"""Loading and preprocessing functions for the Forecasting Sticker Sales data."""
import pandas as pd

CATEGORIES = ["country", "store", "product"]
FEATURES = CATEGORIES + ["year", "month", "weekday", "dayofyear"]


def load_data(data_dir="data"):
    """Read the train and test CSV files and parse their date columns."""
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    return train, test


def preprocess(df):
    """Split the date into year, month, weekday and day-of-year columns, and mark the label columns as categories."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    for c in CATEGORIES:
        df[c] = df[c].astype("category")
    return df
