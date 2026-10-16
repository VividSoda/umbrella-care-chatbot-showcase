"""Info-gain follow-ups should converge in fewer turns than 'typical symptom'."""

from pathlib import Path

import numpy as np
import pytest

from src.chatbot.constants import STOP_CONFIDENCE
from src.chatbot.predictor import Predictor

MODEL_PATH = Path(__file__).parents[2] / "artifacts" / "model.joblib"
START = 2  # symptoms in the first message
MAX_QUESTIONS = 8

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")


def _patients(predictor: Predictor) -> list[tuple[str, set[str]]]:
    """One sample conversation per disease: the symptoms it usually shows."""
    model = predictor._model
    out = []
    for label, probs in zip(predictor._labels, model.feature_log_prob_):
        typical = np.flatnonzero(np.exp(probs) > 0.5)
        if len(typical) >= START + 1:
            out.append((label, {predictor.symptoms[i] for i in typical}))
    return out


def _questions_needed(predictor, disease, has, pick) -> int:
    """Questions asked before the bot is sure and right, or MAX_QUESTIONS.

    A "no" answer adds nothing: unanswered symptoms are ignored, not absent.
    """
    known = sorted(has)[:START]
    declined: set[str] = set()
    for asked in range(MAX_QUESTIONS + 1):
        top = predictor.predict(known, top_n=1)[0]
        if top.disease == disease and top.probability >= STOP_CONFIDENCE:
            return asked
        choices = [s for s in pick(known) if s not in declined]
        if not choices:
            break
        if choices[0] in has:
            known.append(choices[0])
        else:
            declined.add(choices[0])
    return MAX_QUESTIONS


def test_info_gain_needs_fewer_questions_than_typical_symptoms():
    predictor = Predictor.load(MODEL_PATH)
    patients = _patients(predictor)
    assert len(patients) > 10

    def typical(known):
        top = predictor.predict(known, top_n=1)[0].disease
        return predictor.typical_symptoms(top, known, len(predictor.symptoms))

    def smart(known):
        return predictor.next_symptoms(known, len(predictor.symptoms))

    baseline = sum(_questions_needed(predictor, d, h, typical) for d, h in patients)
    improved = sum(_questions_needed(predictor, d, h, smart) for d, h in patients)
    assert improved < baseline
