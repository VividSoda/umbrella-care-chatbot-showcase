from pathlib import Path

import joblib
import pytest

from src.chatbot.symptoms import SYNONYMS, extract_symptoms

MODEL_PATH = Path(__file__).parents[2] / "artifacts" / "model.joblib"

KNOWN = [
    "headache",
    "nausea",
    "vomiting",
    "cough",
    "mild_fever",
    "high_fever",
    "skin_rash",
    "itching",
    "chest_pain",
    "stomach_pain",
]


def test_matches_column_names_as_plain_words():
    assert extract_symptoms("I have chest pain", KNOWN) == ["chest_pain"]


def test_matches_synonyms():
    assert extract_symptoms("I have a headache and feel sick", KNOWN) == [
        "headache",
        "nausea",
    ]


def test_order_follows_first_mention():
    assert extract_symptoms("vomiting then a bad cough", KNOWN) == ["vomiting", "cough"]


def test_longest_phrase_wins():
    assert extract_symptoms("I have a high fever", KNOWN) == ["high_fever"]
    assert extract_symptoms("I have a fever", KNOWN) == ["mild_fever"]


def test_negation_skips_symptom():
    assert extract_symptoms("no fever but a cough", KNOWN) == ["cough"]
    assert extract_symptoms("I don't have a headache", KNOWN) == []


def test_negated_long_phrase_does_not_fall_back_to_shorter_one():
    assert extract_symptoms("no high fever", KNOWN) == []


def test_apostrophes_and_punctuation_are_ignored():
    assert extract_symptoms("Can't stop COUGHING!!", KNOWN) == ["cough"]


def test_only_known_symptoms_are_returned():
    assert extract_symptoms("rash and itchy", ["itching"]) == ["itching"]


def test_duplicates_are_collapsed():
    assert extract_symptoms("cough, coughing, more cough", KNOWN) == ["cough"]


@pytest.mark.parametrize("text", ["", "   ", "hello there", "!!!"])
def test_no_symptoms_returns_empty(text):
    assert extract_symptoms(text, KNOWN) == []


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model.joblib not trained")
def test_every_synonym_key_is_a_real_model_symptom():
    symptoms = set(joblib.load(MODEL_PATH)["symptoms"])
    assert set(SYNONYMS) <= symptoms
