"""How urgent is it? Plain rules, no model and no LLM, so each decision can be audited."""

from collections.abc import Collection

from src.chatbot.constants import (
    EMERGENCY_PHRASES,
    HIGH_SEVERITY,
    LONG_DURATION_DAYS,
    RED_FLAG_PHRASES,
    VULNERABLE_AGE_OVER,
    VULNERABLE_AGE_UNDER,
    Urgency,
    urgency_floor_for,
)
from src.chatbot.symptoms import mentions_any


def _is_vulnerable(
    age: int | None, pregnant: bool | None, chronic_conditions: Collection[str]
) -> bool:
    young_or_old = age is not None and (
        age < VULNERABLE_AGE_UNDER or age > VULNERABLE_AGE_OVER
    )
    return young_or_old or bool(pregnant) or bool(chronic_conditions)


def assess(
    message: str,
    top_disease: str | None = None,
    duration_days: int | None = None,
    severity: int | None = None,
    age: int | None = None,
    pregnant: bool | None = None,
    chronic_conditions: Collection[str] = (),
) -> Urgency:
    """The most urgent level any rule calls for.

    Every rule only raises the result, and a missing input (the user skipped a
    question) means its rule stays silent, so missing data never makes the
    answer safer. Being young, old, pregnant or chronically ill lifts
    self-care to see-a-doctor-soon, but never to emergency on its own.
    """
    if mentions_any(message, EMERGENCY_PHRASES):
        return Urgency.EMERGENCY

    levels = [Urgency.SELF_CARE]
    if top_disease is not None:
        levels.append(urgency_floor_for(top_disease))
    if mentions_any(message, RED_FLAG_PHRASES):
        levels.append(Urgency.SEE_DOCTOR_SOON)
    if severity is not None and severity >= HIGH_SEVERITY:
        levels.append(Urgency.SEE_DOCTOR_SOON)
    if duration_days is not None and duration_days >= LONG_DURATION_DAYS:
        levels.append(Urgency.SEE_DOCTOR_SOON)
    if _is_vulnerable(age, pregnant, chronic_conditions):
        levels.append(Urgency.SEE_DOCTOR_SOON)
    return max(levels)
