"""Wraps the trained model: symptoms in, disease probabilities out."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.chatbot.constants import TOP_N_DISEASES
from src.chatbot.utils import entropy, softmax


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

        # Naive Bayes parameters for next_symptoms, computed once.
        self._index = {s: i for i, s in enumerate(symptoms)}
        self._log_prior = model.class_log_prior_  # (diseases,)
        self._log_yes = model.feature_log_prob_  # (diseases, symptoms)
        self._log_no = np.log1p(-np.exp(self._log_yes))  # log P(absent | disease)
        self._p_yes = np.exp(self._log_yes)

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

        # Symptoms the user did not mention are unknown, not absent.
        present_idx = [self._index[s] for s in symptoms]
        probabilities = self._posterior(present_idx)
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

    def _posterior(self, present_idx: list[int]) -> np.ndarray:
        """P(disease | symptoms the user has). Unanswered symptoms are ignored,
        not treated as absent."""
        scores = self._log_prior.copy()
        if present_idx:
            scores += self._log_yes[:, present_idx].sum(axis=1)
        return softmax(scores)

    def _info_gain(self, posterior: np.ndarray) -> np.ndarray:
        """Expected drop in uncertainty from asking each symptom, shape (n_symptoms,)."""
        chance_yes = posterior @ self._p_yes  # P(user says yes), per symptom

        after_yes = posterior[:, None] * self._p_yes  # (diseases, symptoms)
        after_yes /= after_yes.sum(axis=0, keepdims=True)  # belief if the answer is yes

        after_no = posterior[:, None] * np.exp(self._log_no)
        after_no /= after_no.sum(axis=0, keepdims=True)  # belief if the answer is no

        expected = chance_yes * entropy(after_yes.T) + (1 - chance_yes) * entropy(
            after_no.T
        )
        return entropy(posterior) - expected

    def next_symptoms(
        self, present: Collection[str], n: int = 3, min_gain: float = 0.0
    ) -> list[str]:
        """Symptoms worth asking next, best first.

        Ranks by expected information gain: the symptom whose answer would
        narrow the candidate diseases the most. Symptoms whose expected gain is
        below `min_gain` are left out, so the list can be empty when nothing
        separates the candidates. Reads the Naive Bayes
        parameters, so it is tied to BernoulliNB, like typical_symptoms.

        Raises ValueError for a name that is not one of the model's symptoms.
        """
        unknown = set(present) - self._known
        if unknown:
            raise ValueError(f"Unknown symptoms: {sorted(unknown)}")

        present_idx = [self._index[s] for s in present]
        gain = self._info_gain(self._posterior(present_idx))
        gain[present_idx] = -np.inf  # never ask what the user already told us

        best = np.argsort(gain)[::-1][:n]
        return [
            self._symptoms[i] for i in best if gain[i] > -np.inf and gain[i] >= min_gain
        ]
