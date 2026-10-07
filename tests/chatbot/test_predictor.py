from pathlib import Path

import pytest

from src.chatbot.predictor import Predictor

MODEL_PATH = Path(__file__).parents[2] / "artifacts" / "model.joblib"


def test_returns_best_disease_first(predictor):
    result = predictor.predict(["itching", "skin_rash"])
    assert result[0].disease == "Acne"
    assert result[0].probability > result[1].probability


def test_disease_labels_have_whitespace_normalized(predictor):
    assert {p.disease for p in predictor.predict(["cough"], top_n=10)} == {
        "Acne",
        "Asthma",
    }


def test_probabilities_sum_to_one_over_all_diseases(predictor):
    result = predictor.predict(["cough"], top_n=10)
    assert sum(p.probability for p in result) == pytest.approx(1.0)


def test_top_n_limits_results(predictor):
    assert len(predictor.predict(["cough"], top_n=1)) == 1


def test_unknown_symptom_raises(predictor):
    with pytest.raises(ValueError, match="made_up"):
        predictor.predict(["cough", "made_up"])


def test_symptoms_property_keeps_training_order(predictor):
    assert predictor.symptoms == ["itching", "skin_rash", "cough", "breathlessness"]


def test_typical_symptoms_skip_the_ones_already_given(predictor):
    suggestions = predictor.typical_symptoms("Acne", exclude=["itching"], n=2)
    assert suggestions[0] == "skin_rash"
    assert "itching" not in suggestions
    assert len(suggestions) == 2


def test_next_symptoms_never_repeats_a_given_symptom(predictor):
    suggestions = predictor.next_symptoms(["itching", "cough"], n=4)
    assert "itching" not in suggestions
    assert "cough" not in suggestions
    assert len(suggestions) == 2  # only two unasked symptoms exist


def test_next_symptoms_respects_n(predictor):
    assert len(predictor.next_symptoms(["itching"], n=2)) == 2


def test_next_symptoms_works_with_no_symptoms_yet(predictor):
    assert len(predictor.next_symptoms([], n=2)) == 2


def test_next_symptoms_unknown_symptom_raises(predictor):
    with pytest.raises(ValueError, match="made_up"):
        predictor.next_symptoms(["made_up"])


def test_next_symptoms_separates_the_candidates(predictor):
    # "itching" alone leans Acne; a breathing symptom would split Acne from
    # Asthma better than another skin symptom that Acne already explains.
    assert predictor.next_symptoms(["itching"], n=1)[0] in {"cough", "breathlessness"}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model.joblib not trained")
def test_real_model_returns_ranked_top_three():
    real = Predictor.load(MODEL_PATH)
    assert len(real.symptoms) == 132
    top = real.predict(["itching", "skin_rash"])
    assert len(top) == 3
    assert top[0].probability >= top[1].probability >= top[2].probability
