"""Extract symptoms, predict, build the reply."""

from src.chatbot.constants import (
    MIN_CONFIDENCE,
    MIN_SYMPTOMS,
    specialization_for,
)
from src.chatbot.predictor import Predictor
from src.chatbot.schemas import ChatRequest, ChatResponse
from src.chatbot.symptoms import extract_symptoms

DISCLAIMER = "This is not a diagnosis, please consult a doctor."
ASK_MORE = "Can you tell me more? Are you having any other symptoms?"
ASK_MORE_UNSURE = (
    "I can't tell yet what this might be. Can you describe any other symptoms you have?"
)
STARTER_SUGGESTIONS = ["headache", "fever", "cough", "stomach pain"]
SUGGESTION_COUNT = 3


def _label(symptom: str) -> str:
    """Readable chip text. It re-extracts to the same symptom when sent back."""
    return symptom.replace("_", " ")


def _merge(previous: list[str], found: list[str]) -> list[str]:
    """Earlier symptoms first, then new ones, without duplicates."""
    return list(dict.fromkeys([*previous, *found]))


def build_reply(request: ChatRequest, predictor: Predictor) -> ChatResponse:
    known = set(predictor.symptoms)
    # The client's state isn't trusted: drop names the model doesn't know
    # (e.g. after a retrain) instead of failing the whole request.
    previous = [s for s in request.symptoms if s in known]
    found = extract_symptoms(request.message, predictor.symptoms)
    symptoms = _merge(previous, found)

    if len(symptoms) < MIN_SYMPTOMS:
        return ChatResponse(
            message=ASK_MORE,
            suggested=[s for s in STARTER_SUGGESTIONS if s not in set(symptoms)],
            specialization=None,
            symptoms=symptoms,
        )

    top, *others = predictor.predict(symptoms)
    suggested = [
        _label(s)
        for s in predictor.typical_symptoms(top.disease, symptoms, SUGGESTION_COUNT)
    ]

    if top.probability < MIN_CONFIDENCE:
        return ChatResponse(
            message=ASK_MORE_UNSURE,
            suggested=suggested,
            specialization=None,
            symptoms=symptoms,
        )

    message = f"Based on your symptoms this could be {top.disease}."
    if others:
        names = " or ".join(p.disease for p in others[:2])
        message += f" Other possibilities: {names}."
    return ChatResponse(
        message=f"{message} {DISCLAIMER}",
        suggested=suggested,
        specialization=specialization_for(top.disease),
        symptoms=symptoms,
    )
