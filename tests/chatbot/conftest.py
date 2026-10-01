import pandas as pd
import pytest
from sklearn.naive_bayes import BernoulliNB

from src.chatbot.predictor import Predictor

SYMPTOMS = ["itching", "skin_rash", "cough", "breathlessness"]


@pytest.fixture
def predictor() -> Predictor:
    """Tiny two-disease model: skin symptoms -> Acne, breathing symptoms -> Asthma.

    "Acne " keeps a trailing space, like the real dataset's labels.
    """
    rows = [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    labels = ["Acne "] * 3 + ["Asthma"] * 3
    model = BernoulliNB().fit(pd.DataFrame(rows, columns=SYMPTOMS), labels)
    return Predictor(model, SYMPTOMS)
