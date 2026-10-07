import numpy as np


def softmax(log_scores: np.ndarray) -> np.ndarray:
    """Log-scores to probabilities that sum to 1."""
    s = np.exp(log_scores - log_scores.max())
    return s / s.sum()


def entropy(p: np.ndarray) -> np.ndarray:
    """Uncertainty of a distribution (0 = certain). Works on 1D or rows of 2D."""
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=-1)
