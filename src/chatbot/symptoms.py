"""Free text to symptom names."""

import re
from collections.abc import Collection

# Extra phrasings per symptom column. The column name itself ("high_fever" ->
# "high fever") is always matched too, so only add what people actually type.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "itching": ("itchy", "itch", "itchiness", "scratching"),
    "skin_rash": ("rash", "rashes"),
    "continuous_sneezing": ("sneezing", "sneeze", "sneezes"),
    "shivering": ("shivers", "trembling"),
    "chills": ("chill",),
    "joint_pain": ("joint ache", "aching joints", "painful joints", "joints hurt"),
    "stomach_pain": ("stomach ache", "stomachache", "tummy ache", "tummy pain"),
    "abdominal_pain": ("abdomen pain", "pain in abdomen", "pain in my abdomen"),
    "belly_pain": ("belly ache", "belly hurts"),
    "acidity": ("acid reflux", "heartburn", "sour stomach"),
    "vomiting": ("vomit", "vomited", "throwing up", "threw up", "puking", "puke"),
    "burning_micturition": (
        "burning urination",
        "burning when i pee",
        "burning while peeing",
        "burning sensation while urinating",
        "pain while urinating",
        "painful urination",
    ),
    "fatigue": ("tired", "tiredness", "exhausted", "exhaustion", "no energy"),
    "weight_gain": ("gaining weight", "gained weight"),
    "weight_loss": ("losing weight", "lost weight"),
    "anxiety": ("anxious", "panic", "nervous"),
    "mood_swings": ("mood swing", "moody"),
    "restlessness": ("restless",),
    "lethargy": ("lethargic", "sluggish"),
    "cough": ("coughing", "coughs", "dry cough"),
    "high_fever": ("very high fever", "high temperature", "burning up"),
    "mild_fever": (
        "fever",
        "slight fever",
        "low fever",
        "low grade fever",
        "temperature",
    ),
    "breathlessness": (
        "short of breath",
        "shortness of breath",
        "difficulty breathing",
        "trouble breathing",
        "hard to breathe",
        "cant breathe",
        "out of breath",
    ),
    "sweating": ("sweat", "sweaty", "sweats", "night sweats"),
    "dehydration": ("dehydrated",),
    "indigestion": ("upset stomach", "bad digestion"),
    "headache": ("head ache", "head hurts", "head pain", "pain in my head"),
    "yellowish_skin": ("yellow skin", "skin is yellow", "skin turned yellow"),
    "yellowing_of_eyes": ("yellow eyes", "eyes are yellow", "eyes turned yellow"),
    "dark_urine": ("dark colored urine", "dark coloured urine"),
    "nausea": ("nauseous", "nauseated", "feel sick", "feeling sick", "queasy"),
    "loss_of_appetite": (
        "no appetite",
        "not hungry",
        "dont feel like eating",
        "lost my appetite",
        "poor appetite",
    ),
    "pain_behind_the_eyes": ("pain behind my eyes", "pain behind eyes"),
    "back_pain": ("backache", "back ache", "back hurts", "sore back"),
    "constipation": ("constipated",),
    "diarrhoea": (
        "diarrhea",
        "loose motion",
        "loose motions",
        "loose stool",
        "loose stools",
    ),
    "fluid_overload": ("fluid retention",),
    "swelling_of_stomach": ("swollen stomach", "bloated stomach", "stomach swelling"),
    "swelled_lymph_nodes": ("swollen lymph nodes", "swollen glands", "swollen gland"),
    "malaise": ("unwell", "general discomfort", "feeling unwell", "not feeling well"),
    "blurred_and_distorted_vision": (
        "blurred vision",
        "blurry vision",
        "distorted vision",
    ),
    "phlegm": ("mucus", "phlegmy"),
    "throat_irritation": ("sore throat", "scratchy throat", "irritated throat"),
    "redness_of_eyes": ("red eyes", "eyes are red", "bloodshot eyes"),
    "sinus_pressure": ("sinus pain", "sinus", "sinuses"),
    "runny_nose": ("nose is running", "dripping nose"),
    "congestion": ("stuffy nose", "blocked nose", "nasal congestion", "congested"),
    "chest_pain": ("chest hurts", "pain in my chest", "chest tightness", "tight chest"),
    "weakness_in_limbs": ("weak arms", "weak legs", "limbs feel weak"),
    "fast_heart_rate": (
        "racing heart",
        "heart racing",
        "rapid heartbeat",
        "fast heartbeat",
        "heart is beating fast",
    ),
    "pain_during_bowel_movements": (
        "painful bowel movements",
        "pain while passing stool",
    ),
    "bloody_stool": ("blood in stool", "blood in my stool", "bloody stools"),
    "neck_pain": ("neck ache", "sore neck", "neck hurts"),
    "dizziness": ("dizzy", "lightheaded", "light headed", "giddy"),
    "cramps": ("cramp", "cramping", "muscle cramps"),
    "bruising": ("bruises", "bruise", "bruised easily"),
    "obesity": ("obese", "overweight"),
    "swollen_legs": (
        "legs swollen",
        "leg swelling",
        "swelling in legs",
        "swollen feet",
    ),
    "puffy_face_and_eyes": ("puffy face", "puffy eyes", "swollen face"),
    "enlarged_thyroid": ("goiter", "goitre", "swollen thyroid"),
    "brittle_nails": ("nails break easily", "weak nails"),
    "excessive_hunger": ("always hungry", "extreme hunger", "very hungry"),
    "slurred_speech": ("slurring", "slurred words", "speech is slurred"),
    "knee_pain": ("knee ache", "knee hurts", "sore knee", "knees hurt"),
    "hip_joint_pain": ("hip pain", "hip hurts"),
    "muscle_weakness": ("weak muscles", "muscles feel weak"),
    "stiff_neck": ("neck stiffness", "neck is stiff"),
    "swelling_joints": ("swollen joints", "joint swelling"),
    "movement_stiffness": ("stiffness", "stiff joints", "stiff muscles"),
    "spinning_movements": ("room spinning", "spinning", "vertigo"),
    "loss_of_balance": (
        "lose my balance",
        "losing balance",
        "off balance",
        "cant balance",
    ),
    "unsteadiness": ("unsteady", "wobbly"),
    "weakness_of_one_body_side": ("one sided weakness", "weakness on one side"),
    "loss_of_smell": ("cant smell", "no sense of smell", "lost my sense of smell"),
    "bladder_discomfort": ("bladder pain", "bladder pressure"),
    "foul_smell_of_urine": (
        "smelly urine",
        "urine smells bad",
        "strong smelling urine",
    ),
    "continuous_feel_of_urine": ("constant urge to urinate",),
    "passage_of_gases": ("gas", "gassy", "flatulence", "farting"),
    "internal_itching": ("itching inside",),
    "depression": ("depressed", "feeling low", "hopeless"),
    "irritability": ("irritable", "short tempered"),
    "muscle_pain": (
        "body ache",
        "body aches",
        "body pain",
        "muscle ache",
        "muscle aches",
        "sore muscles",
    ),
    "red_spots_over_body": ("red spots", "spots on body", "red spots on my body"),
    "abnormal_menstruation": (
        "irregular periods",
        "irregular period",
        "heavy periods",
        "missed period",
    ),
    "watering_from_eyes": ("watery eyes", "eyes watering", "teary eyes"),
    "increased_appetite": ("eating more", "increased hunger"),
    "polyuria": (
        "frequent urination",
        "urinating a lot",
        "peeing a lot",
        "pee frequently",
    ),
    "lack_of_concentration": (
        "cant concentrate",
        "poor concentration",
        "trouble concentrating",
    ),
    "visual_disturbances": ("vision problems", "seeing spots", "seeing flashes"),
    "coma": ("unconscious", "unresponsive"),
    "stomach_bleeding": ("bleeding stomach", "vomiting blood"),
    "distention_of_abdomen": ("bloating", "bloated", "swollen abdomen"),
    "history_of_alcohol_consumption": (
        "drink alcohol",
        "heavy drinker",
        "drinks a lot",
    ),
    "blood_in_sputum": ("coughing blood", "coughing up blood", "blood when i cough"),
    "prominent_veins_on_calf": ("visible veins on calf", "bulging veins"),
    "palpitations": ("palpitation", "heart pounding", "heart fluttering"),
    "painful_walking": ("pain when walking", "hurts to walk", "pain while walking"),
    "pus_filled_pimples": ("pus pimples", "pimples with pus"),
    "blackheads": ("blackhead",),
    "skin_peeling": ("peeling skin", "skin is peeling"),
    "blister": ("blisters", "blistering"),
}

# A negator within this many tokens before a match cancels it ("no fever").
_NEGATORS = frozenset({"no", "not", "without", "never", "dont", "didnt", "isnt", "nor"})
_NEGATION_WINDOW = 3
# A contrast word ends a negation's reach ("no cough but fever").
_BREAKERS = frozenset({"but", "however", "although"})

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    # Apostrophes are dropped, not split on, so "don't" becomes "dont".
    text = text.lower().replace("'", "").replace("’", "")
    return _NON_WORD.sub(" ", text).split()


def _phrases(symptom: str) -> list[tuple[str, ...]]:
    candidates = [symptom.replace("_", " "), *SYNONYMS.get(symptom, ())]
    return [tuple(tokens) for c in candidates if (tokens := _tokenize(c))]


def _is_negated(tokens: list[str], start: int) -> bool:
    for token in reversed(tokens[max(0, start - _NEGATION_WINDOW) : start]):
        if token in _BREAKERS:
            return False
        if token in _NEGATORS:
            return True
    return False


def extract_symptoms(text: str, known: Collection[str]) -> list[str]:
    """Symptom names mentioned in `text`, restricted to the model's `known` symptoms.

    Longer phrases win, so "high fever" yields `high_fever` and not `mild_fever`.
    Negated mentions ("no cough") are skipped. Order follows the first mention.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    candidates = [
        (phrase, symptom) for symptom in known for phrase in _phrases(symptom)
    ]
    candidates.sort(key=lambda c: len(c[0]), reverse=True)

    consumed: set[int] = set()
    found: dict[str, int] = {}
    for phrase, symptom in candidates:
        size = len(phrase)
        for start in range(len(tokens) - size + 1):
            if tuple(tokens[start : start + size]) != phrase:
                continue
            span = range(start, start + size)
            if any(i in consumed for i in span):
                continue
            consumed.update(
                span
            )  # even if negated: "no high fever" must not fall to "fever"
            if not _is_negated(tokens, start):
                found[symptom] = min(start, found.get(symptom, start))

    return sorted(found, key=found.__getitem__)
