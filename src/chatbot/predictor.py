"""Wraps the trained model: symptoms in, disease probabilities out."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.chatbot.constants import TOP_N_DISEASES


@dataclass(frozen=True)
class Prediction:
    disease: str
    probability: float


class Predictor:
    """Holds the trained model and the symptom order it was trained with."""

    def __init__(self, model, symptoms: list[str]) -> None:
        self._model = model
        self._symptoms = symptoms
        self._known = frozenset(symptoms)
        # Dataset labels have stray whitespace ("Diabetes ", double spaces).
        self._labels = [" ".join(str(c).split()) for c in model.classes_]

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        """Load the bundle written by ml/train.py: {"model": ..., "symptoms": [...]}."""
        bundle = joblib.load(path)
        return cls(bundle["model"], bundle["symptoms"])

    @property
    def symptoms(self) -> list[str]:
        """All symptom names the model understands, in training order."""
        return list(self._symptoms)

    def predict(
        self, symptoms: Collection[str], top_n: int = TOP_N_DISEASES
    ) -> list[Prediction]:
        """Most likely diseases for the given symptoms, best first.

        Raises ValueError for a name that is not one of the model's symptoms.
        """
        unknown = set(symptoms) - self._known
        if unknown:
            raise ValueError(f"Unknown symptoms: {sorted(unknown)}")

        # The model was fitted on a DataFrame, so it expects named columns in
        # training order; 1 marks a symptom the user has.
        row = pd.DataFrame([[0] * len(self._symptoms)], columns=self._symptoms)
        for symptom in symptoms:
            row[symptom] = 1

        probabilities = self._model.predict_proba(row)[0]
        best = np.argsort(probabilities)[::-1][:top_n]
        return [Prediction(self._labels[i], float(probabilities[i])) for i in best]

    def typical_symptoms(
        self, disease: str, exclude: Collection[str] = (), n: int = 3
    ) -> list[str]:
        """Symptoms most associated with `disease` that are not in `exclude`.

        Reads P(symptom | disease) from the Naive Bayes model, so it is tied to
        BernoulliNB. Used to suggest what to ask the user next.
        """
        log_probs = self._model.feature_log_prob_[self._labels.index(disease)]
        skipped = set(exclude)
        ranked = np.argsort(log_probs)[::-1]
        return [self._symptoms[i] for i in ranked if self._symptoms[i] not in skipped][
            :n
        ]
