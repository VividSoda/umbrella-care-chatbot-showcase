"""Compare follow-up question strategies by simulating patients."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from train import TARGET, load_data  # reuse the cleaning train.py already does

START = 2  # symptoms the simulated patient mentions in the first message
MAX_Q = 8  # follow-up questions to simulate
FOLDS = 3
CONFIGS = ["current", "partial", "partial+IG"]


def load():
    df = load_data()
    X = df.drop(columns=TARGET).to_numpy(
        dtype=int
    )  # convert to numpy 2D array and force every value to be 0 or 1
    y = df[TARGET].str.strip().to_numpy()  # the true disease of each patient
    return X, y


def entropy(p):
    """How uncertain a probability distribution is (0 means certain)."""
    s = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1)


def softmax(log_scores):
    """Turn log-scores into probabilities that sum to 1."""
    s = np.exp(log_scores - log_scores.max())
    return s / s.sum()
