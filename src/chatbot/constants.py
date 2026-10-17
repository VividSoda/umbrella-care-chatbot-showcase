"""Disease to specialization map and thresholds."""

from enum import IntEnum

# Specialization values are the exact category names used by the Flutter app's
# DoctorSpecialtyMatcher (doctor_filter_categories.dart), so the app can look up
# doctors from the reply without translating them. Keep both in sync.
DEFAULT_SPECIALIZATION = "Physician"

# Keys are the `prognosis` labels from Training.csv with whitespace normalized
# (see `specialization_for`). Spelling matches the dataset, typos included.
DISEASE_TO_SPECIALIZATION: dict[str, str] = {
    "(vertigo) Paroymsal Positional Vertigo": "ENT",
    "AIDS": "Immunologist",
    "Acne": "Dermatologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Allergy": "Allergist",
    "Arthritis": "Rheumatologist",
    "Bronchial Asthma": "Pulmonologist",
    "Cervical spondylosis": "Orthopedic",
    "Chicken pox": "Physician",
    "Chronic cholestasis": "Gastroenterologist",
    "Common Cold": "Physician",
    "Dengue": "Physician",
    "Diabetes": "Endocrinologist",
    "Dimorphic hemmorhoids(piles)": "General Surgeon",
    "Drug Reaction": "Allergist",
    "Fungal infection": "Dermatologist",
    "GERD": "Gastroenterologist",
    "Gastroenteritis": "Gastroenterologist",
    "Heart attack": "Cardiologist",
    "Hepatitis B": "Gastroenterologist",
    "Hepatitis C": "Gastroenterologist",
    "Hepatitis D": "Gastroenterologist",
    "Hepatitis E": "Gastroenterologist",
    "Hypertension": "Cardiologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Hypothyroidism": "Endocrinologist",
    "Impetigo": "Dermatologist",
    "Jaundice": "Gastroenterologist",
    "Malaria": "Physician",
    "Migraine": "Neurologist",
    "Osteoarthristis": "Orthopedic",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "Pneumonia": "Pulmonologist",
    "Psoriasis": "Dermatologist",
    "Tuberculosis": "Pulmonologist",
    "Typhoid": "Physician",
    "Urinary tract infection": "Urologist",
    "Varicose veins": "General Surgeon",
    "hepatitis A": "Gastroenterologist",
}

# Life-threatening situations the symptom model can't judge. The reply tells the
# user to get emergency help instead of predicting from earlier symptoms.
EMERGENCY_PHRASES: tuple[str, ...] = (
    "cardiac arrest",
    "stroke",
    "not breathing",
    "stopped breathing",
    "cant breathe at all",
    "unconscious",
    "passed out",
    "seizure",
    "severe bleeding",
    "bleeding heavily",
    "overdose",
    "poisoning",
    "suicidal",
    "want to die",
)

# Bone and joint injuries are not in the disease dataset, so they get a
# dedicated reply pointing to this specialization.
INJURY_SPECIALIZATION = "Orthopedic"
INJURY_PHRASES: tuple[str, ...] = (
    "fracture",
    "fractured",
    "broken bone",
    "broken bones",
    "broke my",
    "broken arm",
    "broken leg",
    "broken hand",
    "broken foot",
    "broken finger",
    "broken toe",
    "broken rib",
    "broken wrist",
    "broken ankle",
    "is broken",
    "are broken",
    "dislocated",
    "dislocation",
    "sprain",
    "sprained",
    "torn ligament",
)

# Fewer recognized symptoms than this: ask a follow-up instead of predicting.
MIN_SYMPTOMS = 2

# How many top diseases the predictor returns.
TOP_N_DISEASES = 3

# Top disease below this probability: treat the prediction as too uncertain and
# ask for more symptoms. Chosen from the threshold table in
# docs/model_evaluation.md: with two symptoms named, the bot answers about half
# the time and is right 93% of those (on the synthetic dataset).
MIN_CONFIDENCE = 0.3

# Top disease at or above this probability: it is clearly ahead, so stop
# asking follow-up questions and just answer. Unmentioned symptoms count as
# unknown, so confidence rarely reaches 0.9; at 0.6 the guess was right in every
# case measured with three or more symptoms (docs/model_evaluation.md).
STOP_CONFIDENCE = 0.6

# A follow-up must be expected to cut uncertainty by at least this many nats
# (see Predictor.next_symptoms), otherwise it is not worth asking.
MIN_QUESTION_GAIN = 0.05


class Urgency(IntEnum):
    """How soon the user should act. Ordered, so `max()` picks the more urgent."""

    SELF_CARE = 0
    SEE_DOCTOR_SOON = 1
    EMERGENCY = 2


# Lowest urgency a predicted disease can have. Triage rules may raise it, never
# lower it. Keys follow DISEASE_TO_SPECIALIZATION. A disease missing here gets
# DEFAULT_DISEASE_URGENCY, so "unknown" never reads as "safe".
# Needs clinician review before release.
DEFAULT_DISEASE_URGENCY = Urgency.SEE_DOCTOR_SOON
DISEASE_URGENCY: dict[str, Urgency] = {
    # Time-critical: treat as an emergency even though the model may be wrong.
    "Heart attack": Urgency.EMERGENCY,
    "Paralysis (brain hemorrhage)": Urgency.EMERGENCY,
    # Usually manageable at home when mild; severity and duration can raise it.
    "Acne": Urgency.SELF_CARE,
    "Allergy": Urgency.SELF_CARE,
    "Common Cold": Urgency.SELF_CARE,
    "Fungal infection": Urgency.SELF_CARE,
    "GERD": Urgency.SELF_CARE,
    "Migraine": Urgency.SELF_CARE,
    "Varicose veins": Urgency.SELF_CARE,
    "Cervical spondylosis": Urgency.SELF_CARE,
    "Osteoarthristis": Urgency.SELF_CARE,
    "Dimorphic hemmorhoids(piles)": Urgency.SELF_CARE,
    "Chicken pox": Urgency.SELF_CARE,
    # Everything else needs a doctor's assessment soon.
    "(vertigo) Paroymsal Positional Vertigo": Urgency.SEE_DOCTOR_SOON,
    "AIDS": Urgency.SEE_DOCTOR_SOON,
    "Alcoholic hepatitis": Urgency.SEE_DOCTOR_SOON,
    "Arthritis": Urgency.SEE_DOCTOR_SOON,
    "Bronchial Asthma": Urgency.SEE_DOCTOR_SOON,
    "Chronic cholestasis": Urgency.SEE_DOCTOR_SOON,
    "Dengue": Urgency.SEE_DOCTOR_SOON,
    "Diabetes": Urgency.SEE_DOCTOR_SOON,
    "Drug Reaction": Urgency.SEE_DOCTOR_SOON,
    "Gastroenteritis": Urgency.SEE_DOCTOR_SOON,
    "Hepatitis B": Urgency.SEE_DOCTOR_SOON,
    "Hepatitis C": Urgency.SEE_DOCTOR_SOON,
    "Hepatitis D": Urgency.SEE_DOCTOR_SOON,
    "Hepatitis E": Urgency.SEE_DOCTOR_SOON,
    "Hypertension": Urgency.SEE_DOCTOR_SOON,
    "Hyperthyroidism": Urgency.SEE_DOCTOR_SOON,
    "Hypoglycemia": Urgency.SEE_DOCTOR_SOON,
    "Hypothyroidism": Urgency.SEE_DOCTOR_SOON,
    "Impetigo": Urgency.SEE_DOCTOR_SOON,
    "Jaundice": Urgency.SEE_DOCTOR_SOON,
    "Malaria": Urgency.SEE_DOCTOR_SOON,
    "Peptic ulcer diseae": Urgency.SEE_DOCTOR_SOON,
    "Pneumonia": Urgency.SEE_DOCTOR_SOON,
    "Psoriasis": Urgency.SEE_DOCTOR_SOON,
    "Tuberculosis": Urgency.SEE_DOCTOR_SOON,
    "Typhoid": Urgency.SEE_DOCTOR_SOON,
    "Urinary tract infection": Urgency.SEE_DOCTOR_SOON,
    "hepatitis A": Urgency.SEE_DOCTOR_SOON,
}

# Worrying on their own, but not a "call an ambulance" situation: raise the
# urgency to at least SEE_DOCTOR_SOON. EMERGENCY_PHRASES always win over these.
RED_FLAG_PHRASES: tuple[str, ...] = (
    "chest pain",
    "chest tightness",
    "blood in stool",
    "blood in urine",
    "blood in vomit",
    "coughing blood",
    "vomiting blood",
    "black stool",
    "cant keep food down",
    "cant keep water down",
    "severe headache",
    "worst headache",
    "stiff neck",
    "high fever",
    "shortness of breath",
    "difficulty breathing",
    "confusion",
    "sudden vision loss",
    "severe abdominal pain",
    "severe stomach pain",
)

# Symptoms lasting at least this many days, or rated at least this high (1-10),
# raise the urgency to at least SEE_DOCTOR_SOON.
LONG_DURATION_DAYS = 7
HIGH_SEVERITY = 7

# Under this age or over that age, one step more urgent. Pregnancy and chronic
# conditions do the same.
VULNERABLE_AGE_UNDER = 5
VULNERABLE_AGE_OVER = 65


def _normalize(label: str) -> str:
    return " ".join(label.split())


_NORMALIZED_MAP = {_normalize(k): v for k, v in DISEASE_TO_SPECIALIZATION.items()}
_NORMALIZED_URGENCY = {_normalize(k): v for k, v in DISEASE_URGENCY.items()}


def specialization_for(disease: str) -> str:
    """Specialization for a model label, tolerant of stray whitespace in the dataset."""
    return _NORMALIZED_MAP.get(_normalize(disease), DEFAULT_SPECIALIZATION)


def urgency_floor_for(disease: str) -> Urgency:
    return _NORMALIZED_URGENCY.get(_normalize(disease), DEFAULT_DISEASE_URGENCY)
