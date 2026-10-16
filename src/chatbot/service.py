"""Extract symptoms, predict, build the reply."""

from src.chatbot.constants import (
    EMERGENCY_PHRASES,
    INJURY_PHRASES,
    INJURY_SPECIALIZATION,
    MIN_CONFIDENCE,
    MIN_QUESTION_GAIN,
    MIN_SYMPTOMS,
    STOP_CONFIDENCE,
    Urgency,
    specialization_for,
)
from src.chatbot.predictor import Predictor
from src.chatbot.schemas import ChatRequest, ChatResponse
from src.chatbot.symptoms import extract_symptoms, mentions_any
from src.chatbot.triage import assess

DISCLAIMER = "This is not a diagnosis, please consult a doctor."
ASK_MORE = "Can you tell me more? Are you having any other symptoms?"
ASK_MORE_UNSURE = (
    "I can't tell yet what this might be. Can you describe any other symptoms you have?"
)
NOT_RECOGNIZED = (
    "I couldn't pick out a symptom in that. Try describing how you feel, "
    "for example a headache, fever or cough. If this is a different problem, "
    "start a new chat."
)
ALREADY_NOTED = "I already have that noted. Is there anything else you are feeling?"
EMERGENCY = (
    "This could be an emergency. Call your local emergency number or go to the "
    "nearest emergency room right away; do not wait for a chat reply."
)
INJURY = (
    "I can't assess injuries like this, but an orthopedic doctor can help. If it "
    f"is very painful, the limb looks out of place, or there is heavy bleeding, "
    f"go to an emergency room right away. {DISCLAIMER}"
)
URGENCY_ADVICE = {
    Urgency.SELF_CARE: "This can often be managed at home, but see a doctor if it gets worse.",
    Urgency.SEE_DOCTOR_SOON: "Please see a doctor soon.",
    Urgency.EMERGENCY: "This could be serious. Get emergency care right away.",
}
STARTER_SUGGESTIONS = ["headache", "fever", "cough", "stomach pain"]
SUGGESTION_COUNT = 3


def _label(symptom: str) -> str:
    """Readable chip text. It re-extracts to the same symptom when sent back."""
    return symptom.replace("_", " ")


def _urgency(request: ChatRequest, disease: str | None = None) -> Urgency:
    """Triage from the message and the profile. Pass `disease` only for a
    prediction that is trusted enough to show; otherwise its floor is skipped."""
    return assess(
        request.message,
        disease,
        duration_days=request.duration_days,
        severity=request.severity,
        age=request.age,
        pregnant=request.pregnant,
        chronic_conditions=request.chronic_conditions,
    )


def _merge(previous: list[str], found: list[str]) -> list[str]:
    """Earlier symptoms first, then new ones, without duplicates."""
    return list(dict.fromkeys([*previous, *found]))


def _follow_ups(predictor: Predictor, symptoms: list[str]) -> list[str]:
    """Chips to offer when there is nothing new to predict from."""
    if len(symptoms) < MIN_SYMPTOMS:
        return [s for s in STARTER_SUGGESTIONS if s not in set(symptoms)]
    return _best_questions(predictor, symptoms)


def _best_questions(predictor: Predictor, symptoms: list[str]) -> list[str]:
    """Chips for the symptoms whose answer would separate the candidates most.

    Empty when the top disease is clearly ahead, or when no question is worth
    asking: the reply then stands on its own.
    """
    if predictor.predict(symptoms, top_n=1)[0].probability >= STOP_CONFIDENCE:
        return []
    return [
        _label(s)
        for s in predictor.next_symptoms(
            symptoms, SUGGESTION_COUNT, min_gain=MIN_QUESTION_GAIN
        )
    ]


def build_reply(request: ChatRequest, predictor: Predictor) -> ChatResponse:
    known = set(predictor.symptoms)
    # The client's state isn't trusted: drop names the model doesn't know
    # (e.g. after a retrain) instead of failing the whole request.
    previous = [s for s in request.symptoms if s in known]

    # Checked first, on the message alone: the earlier symptoms must never
    # decide what is said about a new emergency or injury.
    if mentions_any(request.message, EMERGENCY_PHRASES):
        return ChatResponse(
            message=EMERGENCY,
            suggested=[],
            specialization=None,
            symptoms=previous,
            urgency=Urgency.EMERGENCY,
        )
    if mentions_any(request.message, INJURY_PHRASES):
        return ChatResponse(
            message=INJURY,
            suggested=[],
            specialization=INJURY_SPECIALIZATION,
            symptoms=previous,
            urgency=Urgency.SEE_DOCTOR_SOON,
        )

    found = extract_symptoms(request.message, predictor.symptoms)

    # Nothing new in this message: say so instead of repeating the old
    # prediction as if it answered what was just said.
    if not found:
        return ChatResponse(
            message=NOT_RECOGNIZED,
            suggested=_follow_ups(predictor, previous),
            specialization=None,
            symptoms=previous,
            urgency=_urgency(request),
        )
    if previous and set(found) <= set(previous):
        return ChatResponse(
            message=ALREADY_NOTED,
            suggested=_follow_ups(predictor, previous),
            specialization=None,
            symptoms=previous,
            urgency=_urgency(request),
        )

    symptoms = _merge(previous, found)

    if len(symptoms) < MIN_SYMPTOMS:
        return ChatResponse(
            message=ASK_MORE,
            suggested=[s for s in STARTER_SUGGESTIONS if s not in set(symptoms)],
            specialization=None,
            symptoms=symptoms,
            urgency=_urgency(request),
        )

    top, *others = predictor.predict(symptoms)
    suggested = _best_questions(predictor, symptoms)

    if top.probability < MIN_CONFIDENCE:
        return ChatResponse(
            message=ASK_MORE_UNSURE,
            suggested=suggested,
            specialization=None,
            symptoms=symptoms,
            urgency=_urgency(request),
        )

    message = f"Based on your symptoms this could be {top.disease}."
    if others:
        names = " or ".join(p.disease for p in others[:2])
        message += f" Other possibilities: {names}."
    urgency = _urgency(request, top.disease)
    return ChatResponse(
        message=f"{message} {URGENCY_ADVICE[urgency]} {DISCLAIMER}",
        suggested=suggested,
        specialization=specialization_for(top.disease),
        symptoms=symptoms,
        urgency=urgency,
    )
