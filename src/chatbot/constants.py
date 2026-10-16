"""Disease to specialization map and thresholds."""

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
# ask for more symptoms. Naive Bayes is overconfident, so tune against real use.
MIN_CONFIDENCE = 0.3


def _normalize(label: str) -> str:
    return " ".join(label.split())


_NORMALIZED_MAP = {_normalize(k): v for k, v in DISEASE_TO_SPECIALIZATION.items()}


def specialization_for(disease: str) -> str:
    """Specialization for a model label, tolerant of stray whitespace in the dataset."""
    return _NORMALIZED_MAP.get(_normalize(disease), DEFAULT_SPECIALIZATION)
