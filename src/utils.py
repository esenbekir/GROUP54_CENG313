"""Helpers: evaluation metric and submission saving."""
import os
import numpy as np


def mape(real, guess):
    """Mean absolute percentage error, in percent."""
    real, guess = np.asarray(real), np.asarray(guess)
    return np.mean(np.abs(real - guess) / real) * 100


def save_submission(test, guess, path="submissions/submission.csv"):
    """Write the id,num_sold file in Kaggle's format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = test[["id"]].copy()
    out["num_sold"] = guess
    out.to_csv(path, index=False)
    return out
