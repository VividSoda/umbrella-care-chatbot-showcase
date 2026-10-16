from typing import ClassVar

import pytest

from src.chatbot import constants as service_constants
from src.chatbot import service
from src.chatbot.constants import EMERGENCY_PHRASES, Urgency
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


def test_suggestions_are_symptoms_not_yet_mentioned(predictor, monkeypatch):
    monkeypatch.setattr(service, "STOP_CONFIDENCE", 1.1)  # keep asking
    monkeypatch.setattr(service, "MIN_QUESTION_GAIN", 0.0)
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


def test_a_suggested_chip_sent_back_is_understood(predictor, monkeypatch):
    monkeypatch.setattr(service, "STOP_CONFIDENCE", 1.1)  # keep asking
    monkeypatch.setattr(service, "MIN_QUESTION_GAIN", 0.0)
    first = reply(predictor, "itching and a rash")
    chip = first.suggested[0]
    second = reply(predictor, chip, symptoms=first.symptoms)
    assert len(second.symptoms) == 3


def test_unrelated_message_does_not_repeat_the_old_prediction(predictor, monkeypatch):
    monkeypatch.setattr(service, "STOP_CONFIDENCE", 1.1)  # keep asking
    monkeypatch.setattr(service, "MIN_QUESTION_GAIN", 0.0)
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


def test_stops_asking_once_the_top_disease_is_clearly_ahead(predictor, monkeypatch):
    monkeypatch.setattr(service, "STOP_CONFIDENCE", 0.5)
    result = reply(predictor, "itching and a rash")
    assert result.specialization is not None
    assert result.suggested == []


def test_no_question_is_asked_when_none_would_help(predictor, monkeypatch):
    monkeypatch.setattr(service, "STOP_CONFIDENCE", 1.1)
    monkeypatch.setattr(service, "MIN_QUESTION_GAIN", 10.0)
    assert reply(predictor, "itching and a rash").suggested == []


class _NoModel:
    """Stands in for the predictor and fails if anything asks it to predict."""

    symptoms: ClassVar[list[str]] = ["itching", "skin_rash"]

    def predict(self, *args, **kwargs):
        raise AssertionError("the model must not run for an emergency")

    next_symptoms = typical_symptoms = predict


@pytest.mark.parametrize("phrase", EMERGENCY_PHRASES)
def test_every_emergency_phrase_bypasses_the_model(phrase):
    result = reply(_NoModel(), f"I think {phrase}", symptoms=["itching", "skin_rash"])
    assert result.message == service.EMERGENCY
    assert result.urgency == Urgency.EMERGENCY
    assert result.suggested == []
    assert result.specialization is None


def test_injury_is_see_a_doctor_soon(predictor):
    result = reply(predictor, "I have a fractured hand")
    assert result.urgency == Urgency.SEE_DOCTOR_SOON


def test_mild_prediction_is_self_care_with_matching_advice(predictor):
    result = reply(predictor, "itching and a rash")  # Acne: mild
    assert result.urgency == Urgency.SELF_CARE
    assert service.URGENCY_ADVICE[Urgency.SELF_CARE] in result.message
    assert result.message.endswith(DISCLAIMER)


def test_severity_raises_the_prediction_urgency(predictor):
    result = build_reply(
        ChatRequest(message="itching and a rash", severity=9), predictor
    )
    assert result.urgency == Urgency.SEE_DOCTOR_SOON
    assert service.URGENCY_ADVICE[Urgency.SEE_DOCTOR_SOON] in result.message


def test_disease_floor_can_make_the_prediction_an_emergency(predictor, monkeypatch):
    monkeypatch.setattr(
        service_constants, "_NORMALIZED_URGENCY", {"Acne": Urgency.EMERGENCY}
    )
    result = reply(predictor, "itching and a rash")
    assert result.urgency == Urgency.EMERGENCY
    assert service.URGENCY_ADVICE[Urgency.EMERGENCY] in result.message


def test_red_flag_is_not_lost_when_there_are_too_few_symptoms(predictor):
    result = reply(predictor, "I have a cough and chest pain")
    assert result.specialization is None  # too few symptoms to predict
    assert result.urgency == Urgency.SEE_DOCTOR_SOON


def test_unsure_prediction_ignores_the_disease_floor(predictor, monkeypatch):
    monkeypatch.setattr(service, "MIN_CONFIDENCE", 0.999)
    monkeypatch.setattr(
        service_constants, "_NORMALIZED_URGENCY", {"Acne": Urgency.EMERGENCY}
    )
    result = reply(predictor, "itching and a rash")
    assert result.message == service.ASK_MORE_UNSURE
    assert result.urgency == Urgency.SELF_CARE


def test_not_recognized_reply_reports_self_care(predictor):
    assert reply(predictor, "hello").urgency == Urgency.SELF_CARE


def ask_reply(predictor, message="itching and a rash", symptoms=(), **fields):
    request = ChatRequest(message=message, symptoms=list(symptoms), **fields)
    return build_reply(request, predictor)


def test_first_prediction_asks_for_duration(predictor):
    result = ask_reply(predictor)
    assert result.ask == "duration"
    assert service.TRIAGE_QUESTIONS["duration"] in result.message
    assert result.message.endswith(DISCLAIMER)
    assert result.specialization is not None  # the answer is not held back


def test_then_asks_for_severity(predictor):
    result = ask_reply(predictor, duration_days=3, asked=["duration"])
    assert result.ask == "severity"


def test_a_skipped_question_is_not_asked_again(predictor):
    result = ask_reply(predictor, asked=["duration"])
    assert result.ask == "severity"
    result = ask_reply(predictor, asked=["duration", "severity"])
    assert result.ask is None


def test_answered_questions_are_not_asked(predictor):
    result = ask_reply(predictor, duration_days=3, severity=4)
    assert result.ask is None
    assert service.TRIAGE_QUESTIONS["duration"] not in result.message


def test_no_question_after_an_emergency(predictor):
    result = ask_reply(predictor, "I think I have a seizure")
    assert result.ask is None


def test_no_question_when_the_disease_floor_is_an_emergency(predictor, monkeypatch):
    monkeypatch.setattr(
        service_constants, "_NORMALIZED_URGENCY", {"Acne": Urgency.EMERGENCY}
    )
    assert ask_reply(predictor).ask is None


def test_no_question_without_a_prediction(predictor):
    assert ask_reply(predictor, "I have a cough").ask is None
    assert ask_reply(predictor, "hello").ask is None


def test_no_question_when_the_prediction_is_too_uncertain(predictor, monkeypatch):
    monkeypatch.setattr(service, "MIN_CONFIDENCE", 0.999)
    assert ask_reply(predictor).ask is None


def test_answering_refreshes_the_reply_without_a_new_symptom(predictor):
    first = ask_reply(predictor)
    result = ask_reply(
        predictor,
        "10 days",
        symptoms=first.symptoms,
        duration_days=10,
        asked=["duration"],
        answering="duration",
    )
    assert result.message != service.NOT_RECOGNIZED
    assert result.symptoms == first.symptoms
    assert result.urgency == Urgency.SEE_DOCTOR_SOON  # long duration
    assert result.ask == "severity"


def test_skipping_still_refreshes_the_reply(predictor):
    first = ask_reply(predictor)
    result = ask_reply(
        predictor,
        "skip",
        symptoms=first.symptoms,
        asked=["duration"],
        answering="duration",
    )
    assert result.specialization is not None
    assert result.ask == "severity"


def test_unrelated_text_is_still_not_recognized_without_answering(predictor):
    result = ask_reply(
        predictor, "my car won't start", symptoms=["itching", "skin_rash"],
        duration_days=3, asked=["duration"],
    )  # fmt: skip
    assert result.message == service.NOT_RECOGNIZED


def test_answering_with_too_few_symptoms_is_not_a_prediction(predictor):
    result = ask_reply(
        predictor, "3 days", symptoms=["cough"], duration_days=3, answering="duration"
    )
    assert result.message == service.NOT_RECOGNIZED
    assert result.specialization is None
