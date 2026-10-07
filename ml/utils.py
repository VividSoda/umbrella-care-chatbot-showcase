import numpy as np


def softmax(log_scores):
    """Turn log-scores into probabilities that sum to 1."""
    s = np.exp(log_scores - log_scores.max())
    return s / s.sum()
