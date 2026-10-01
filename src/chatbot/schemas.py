"""Request and response models."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Caps keep a single request cheap: the model has only 132 symptoms.
_MAX_MESSAGE_LENGTH = 500
_MAX_SYMPTOMS = 132

Symptom = Annotated[str, StringConstraints(strip_whitespace=True, max_length=64)]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_LENGTH)
    # Symptoms recognized earlier in the conversation, exactly as the previous
    # response returned them. The server keeps no history.
    symptoms: list[Symptom] = Field(default_factory=list, max_length=_MAX_SYMPTOMS)


class ChatResponse(BaseModel):
    message: str
    # Follow-up prompts the app can show as quick-reply chips. Sending one back
    # as the next `message` is understood as-is.
    suggested: list[str]
    # Recommended doctor category, or None while there is too little to go on.
    specialization: str | None
    # Every symptom recognized so far. The app sends these back next time.
    symptoms: list[str]
