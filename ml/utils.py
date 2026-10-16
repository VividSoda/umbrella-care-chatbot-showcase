import numpy as np


def entropy(p):
    """How uncertain a distribution is (0 = certain). Works on 1D or 2D (rows)."""
    p = np.clip(p, 1e-12, 1)  # keep log() away from 0
    return -(p * np.log(p)).sum(axis=-1)


def softmax(log_scores):
    """Turn log-scores into probabilities that sum to 1."""
    s = np.exp(log_scores - log_scores.max())
    return s / s.sum()
