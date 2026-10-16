"""Request and response models."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_serializer

from src.chatbot.constants import Urgency

# Caps keep a single request cheap: the model has only 132 symptoms.
_MAX_MESSAGE_LENGTH = 500
_MAX_SYMPTOMS = 132
_MAX_DURATION_DAYS = 3650  # ten years; anything longer is a typo
_MAX_AGE = 120
_MAX_CHRONIC_CONDITIONS = 20

Symptom = Annotated[str, StringConstraints(strip_whitespace=True, max_length=64)]
ChronicCondition = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_LENGTH)
    # Symptoms recognized earlier in the conversation, exactly as the previous
    # response returned them. The server keeps no history.
    symptoms: list[Symptom] = Field(default_factory=list, max_length=_MAX_SYMPTOMS)

    # Triage inputs. All optional, so older app versions keep working, and the
    # app sends only what the user agreed to share.
    duration_days: int | None = Field(default=None, ge=0, le=_MAX_DURATION_DAYS)
    severity: int | None = Field(default=None, ge=1, le=10)
    age: int | None = Field(default=None, ge=0, le=_MAX_AGE)
    sex: Literal["male", "female", "other"] | None = None
    pregnant: bool | None = None
    chronic_conditions: list[ChronicCondition] = Field(
        default_factory=list, max_length=_MAX_CHRONIC_CONDITIONS
    )


class ChatResponse(BaseModel):
    message: str
    # Follow-up prompts the app can show as quick-reply chips. Sending one back
    # as the next `message` is understood as-is.
    suggested: list[str]
    # Recommended doctor category, or None while there is too little to go on.
    specialization: str | None
    # Every symptom recognized so far. The app sends these back next time.
    symptoms: list[str]

    # How soon to act. Sent as "self_care", "see_doctor_soon" or "emergency".
    urgency: Urgency = Urgency.SELF_CARE
    # The triage question this reply asks, so the app can show the matching
    # input. None when the reply asks nothing.
    ask: Literal["duration", "severity"] | None = None

    @field_serializer("urgency")
    def _urgency_as_text(self, urgency: Urgency) -> str:
        return urgency.name.lower()
