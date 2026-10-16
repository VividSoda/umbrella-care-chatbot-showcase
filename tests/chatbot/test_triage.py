import pytest

from src.chatbot.constants import (
    EMERGENCY_PHRASES,
    HIGH_SEVERITY,
    LONG_DURATION_DAYS,
    RED_FLAG_PHRASES,
    VULNERABLE_AGE_OVER,
    VULNERABLE_AGE_UNDER,
    Urgency,
)
from src.chatbot.triage import assess

MILD = "Common Cold"
SERIOUS = "Pneumonia"
CRITICAL = "Heart attack"


def test_mild_case_is_self_care():
    assert assess("a runny nose", MILD) == Urgency.SELF_CARE


def test_no_disease_and_no_extras_is_self_care():
    assert assess("hello") == Urgency.SELF_CARE


@pytest.mark.parametrize("phrase", EMERGENCY_PHRASES)
def test_every_emergency_phrase_is_an_emergency(phrase):
    assert assess(f"I think {phrase}", MILD) == Urgency.EMERGENCY


def test_negated_emergency_phrase_is_not_an_emergency():
    assert assess("no seizure, just a cold", MILD) == Urgency.SELF_CARE


@pytest.mark.parametrize("phrase", RED_FLAG_PHRASES)
def test_every_red_flag_raises_to_see_a_doctor(phrase):
    assert assess(f"I have {phrase}", MILD) == Urgency.SEE_DOCTOR_SOON


def test_disease_floor_applies():
    assert assess("cough", SERIOUS) == Urgency.SEE_DOCTOR_SOON
    assert assess("cough", CRITICAL) == Urgency.EMERGENCY


def test_unknown_disease_is_not_self_care():
    assert assess("cough", "Made up disease") == Urgency.SEE_DOCTOR_SOON


def test_high_severity_raises():
    assert assess("cough", MILD, severity=HIGH_SEVERITY) == Urgency.SEE_DOCTOR_SOON
    assert assess("cough", MILD, severity=HIGH_SEVERITY - 1) == Urgency.SELF_CARE


def test_long_duration_raises():
    long = LONG_DURATION_DAYS
    assert assess("cough", MILD, duration_days=long) == Urgency.SEE_DOCTOR_SOON
    assert assess("cough", MILD, duration_days=long - 1) == Urgency.SELF_CARE


@pytest.mark.parametrize(
    "extra",
    [
        {"age": VULNERABLE_AGE_UNDER - 1},
        {"age": VULNERABLE_AGE_OVER + 1},
        {"pregnant": True},
        {"chronic_conditions": ["asthma"]},
    ],
)
def test_vulnerable_groups_lift_self_care(extra):
    assert assess("cough", MILD, **extra) == Urgency.SEE_DOCTOR_SOON


@pytest.mark.parametrize(
    "extra",
    [
        {"age": VULNERABLE_AGE_UNDER},
        {"age": VULNERABLE_AGE_OVER},
        {"pregnant": False},
        {"chronic_conditions": []},
    ],
)
def test_ordinary_profile_changes_nothing(extra):
    assert assess("cough", MILD, **extra) == Urgency.SELF_CARE


def test_vulnerable_alone_never_reaches_emergency():
    result = assess("cough", MILD, age=1, pregnant=True, chronic_conditions=["x"])
    assert result == Urgency.SEE_DOCTOR_SOON


def test_nothing_lowers_an_earlier_urgency():
    result = assess(
        "cough", CRITICAL, duration_days=0, severity=1, age=30, pregnant=False
    )
    assert result == Urgency.EMERGENCY


def test_emergency_phrase_beats_everything():
    assert assess("I passed out", None, severity=1) == Urgency.EMERGENCY


def test_skipped_answers_never_make_it_safer():
    assert assess("cough", SERIOUS, None, None) == assess(
        "cough", SERIOUS, duration_days=1, severity=1
    )
