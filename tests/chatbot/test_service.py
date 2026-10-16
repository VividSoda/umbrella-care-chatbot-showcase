from src.chatbot import service
from src.chatbot.schemas import ChatRequest
from src.chatbot.service import DISCLAIMER, build_reply


def reply(predictor, message, symptoms=()):
    return build_reply(ChatRequest(message=message, symptoms=list(symptoms)), predictor)


def test_too_few_symptoms_asks_for_more(predictor):
    result = reply(predictor, "I have a cough")
    assert result.specialization is None
    assert result.symptoms == ["cough"]
    assert result.message == service.ASK_MORE
    assert "cough" not in result.suggested


def test_nothing_recognized_still_asks_for_more(predictor):
    result = reply(predictor, "hello")
    assert result.symptoms == []
    assert result.specialization is None
    assert result.suggested == service.STARTER_SUGGESTIONS


def test_enough_symptoms_predicts_with_specialization_and_disclaimer(predictor):
    result = reply(predictor, "itching and a rash")
    assert result.symptoms == ["itching", "skin_rash"]
    assert "Acne" in result.message
    assert result.message.endswith(DISCLAIMER)
    assert result.specialization == "Dermatologist"


def test_earlier_symptoms_are_combined_with_new_ones(predictor):
    result = reply(predictor, "and a rash", symptoms=["itching"])
    assert result.symptoms == ["itching", "skin_rash"]
    assert result.specialization is not None


def test_repeating_a_symptom_does_not_duplicate_it(predictor):
    result = reply(predictor, "itching again", symptoms=["itching"])
    assert result.symptoms == ["itching"]
    assert result.specialization is None


def test_unknown_symptoms_from_client_are_dropped(predictor):
    result = reply(predictor, "I have a cough", symptoms=["made_up"])
    assert result.symptoms == ["cough"]


def test_suggestions_are_symptoms_not_yet_mentioned(predictor):
    result = reply(predictor, "itching and a rash")
    assert result.suggested
    assert "itching" not in result.suggested
    assert "skin rash" not in result.suggested


def test_low_confidence_asks_for_more_instead_of_guessing(predictor, monkeypatch):
    monkeypatch.setattr(service, "MIN_CONFIDENCE", 0.999)
    result = reply(predictor, "itching and a rash")
    assert result.specialization is None
    assert result.message == service.ASK_MORE_UNSURE
    assert result.symptoms == ["itching", "skin_rash"]


def test_a_suggested_chip_sent_back_is_understood(predictor):
    first = reply(predictor, "itching and a rash")
    chip = first.suggested[0]
    second = reply(predictor, chip, symptoms=first.symptoms)
    assert len(second.symptoms) == 3


def test_unrelated_message_does_not_repeat_the_old_prediction(predictor):
    result = reply(predictor, "my car won't start", symptoms=["itching", "skin_rash"])
    assert result.message == service.NOT_RECOGNIZED
    assert result.specialization is None
    assert result.symptoms == ["itching", "skin_rash"]
    assert result.suggested


def test_repeating_known_symptoms_does_not_repeat_the_prediction(predictor):
    result = reply(predictor, "itching", symptoms=["itching", "skin_rash"])
    assert result.message == service.ALREADY_NOTED
    assert result.specialization is None
    assert result.symptoms == ["itching", "skin_rash"]


def test_emergency_wins_over_earlier_symptoms(predictor):
    result = reply(
        predictor, "I have cardiac arrest", symptoms=["itching", "skin_rash"]
    )
    assert result.message == service.EMERGENCY
    assert result.specialization is None
    assert result.suggested == []
    assert result.symptoms == ["itching", "skin_rash"]


def test_injury_points_to_orthopedics_without_touching_symptoms(predictor):
    result = reply(predictor, "I have a fractured hand", symptoms=["itching"])
    assert result.message == service.INJURY
    assert result.specialization == "Orthopedic"
    assert result.symptoms == ["itching"]
    assert service.DISCLAIMER in result.message


def test_injury_is_recognized_without_any_earlier_symptoms(predictor):
    result = reply(predictor, "my toe is broken")
    assert result.specialization == "Orthopedic"
    assert result.symptoms == []


def test_a_negated_injury_is_not_treated_as_one(predictor):
    result = reply(predictor, "no fracture, just itching and a rash")
    assert result.message != service.INJURY
    assert result.specialization == "Dermatologist"
