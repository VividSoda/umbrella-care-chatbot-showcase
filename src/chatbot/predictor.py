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
        return [
            Prediction(str(self._model.classes_[i]), float(probabilities[i]))
            for i in best
        ]
