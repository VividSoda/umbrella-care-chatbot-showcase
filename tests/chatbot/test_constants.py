import csv
from pathlib import Path

import pytest

from src.chatbot.constants import (
    DEFAULT_SPECIALIZATION,
    DISEASE_TO_SPECIALIZATION,
    specialization_for,
)

TRAINING_CSV = Path(__file__).parents[2] / "ml" / "data" / "Training.csv"

# Category names from the Flutter app's doctor_filter_categories.dart.
APP_CATEGORIES = {
    "Pediatrician", "Psychiatrist", "Dermatologist", "Obs/Gynaecologist",
    "Radiologist", "Orthopedic", "General Surgeon", "Urologist", "ENT",
    "Ophthalmologist", "Endocrinologist", "Pulmonologist", "Cardiologist",
    "Breast Oncosurgeon", "Physician", "Neurologist", "Oncologist",
    "Gastroenterologist", "Allergist", "Immunologist", "Anesthesiologist",
    "Nephrologist", "Rheumatologist", "Hematologist",
}  # fmt: skip


def test_every_specialization_is_an_app_category():
    assert set(DISEASE_TO_SPECIALIZATION.values()) <= APP_CATEGORIES


@pytest.mark.skipif(not TRAINING_CSV.exists(), reason="Training.csv not downloaded")
def test_every_trained_disease_is_mapped():
    with TRAINING_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        diseases = {row["prognosis"] for row in reader}

    unmapped = {
        d
        for d in diseases
        if " ".join(d.split())
        not in {" ".join(k.split()) for k in DISEASE_TO_SPECIALIZATION}
    }
    assert not unmapped


def test_lookup_ignores_stray_whitespace():
    assert specialization_for("Diabetes ") == "Endocrinologist"
    assert specialization_for("(vertigo) Paroymsal  Positional Vertigo") == "ENT"


def test_unknown_disease_falls_back_to_default():
    assert specialization_for("Not a disease") == DEFAULT_SPECIALIZATION
