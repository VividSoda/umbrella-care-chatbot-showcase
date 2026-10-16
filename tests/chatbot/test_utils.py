import numpy as np
import pytest

from src.chatbot.utils import entropy, softmax


def test_softmax_sums_to_one():
    assert softmax(np.array([-1.0, -2.0, -3.0])).sum() == pytest.approx(1.0)


def test_softmax_does_not_overflow_on_large_scores():
    result = softmax(np.array([1000.0, 1001.0]))
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)


def test_entropy_of_a_certain_distribution_is_zero():
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_entropy_of_a_uniform_distribution_is_log_n():
    assert entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


def test_entropy_works_row_wise_on_2d():
    rows = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert entropy(rows) == pytest.approx([0.0, np.log(2)], abs=1e-9)
