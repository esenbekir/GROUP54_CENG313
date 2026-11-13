"""Model training and prediction functions."""
import numpy as np
import lightgbm as lgb

from src.data import CATEGORIES, FEATURES


def naive_baseline(learn, exam):
    """Predict each row as the average sales of the same series and weekday in the last training year.

    Falls back to the series average over all training years when that combination is missing.
    """
    keys = CATEGORIES + ["weekday"]
    last_year = learn[learn["year"] == learn["year"].max()]
    recent = last_year.groupby(keys, observed=True)["num_sold"].mean().rename("recent").reset_index()
    overall = learn.groupby(CATEGORIES, observed=True)["num_sold"].mean().rename("overall").reset_index()
    rows = exam[keys].merge(recent, on=keys, how="left").merge(overall, on=CATEGORIES, how="left")
    return rows["recent"].fillna(rows["overall"]).to_numpy()


def train_model(df, log_target=True, n_estimators=1000, learning_rate=0.05):
    """Fit LightGBM on num_sold. With log_target=True it learns log1p(num_sold), so errors are relative, matching MAPE."""
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, verbose=-1)
    y = np.log1p(df["num_sold"]) if log_target else df["num_sold"]
    model.fit(df[FEATURES], y)
    model.log_target_ = log_target  # remembered so predict() converts back correctly
    return model


def predict(model, df):
    """Predict num_sold, converting back from log scale if the model used it."""
    guess = model.predict(df[FEATURES])
    return np.expm1(guess) if model.log_target_ else guess
