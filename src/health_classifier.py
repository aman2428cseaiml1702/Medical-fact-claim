"""
Health Claim Classifier Module
Rule-based / keyword-driven domain classifier to ensure only health-related
queries proceed to the RAG retrieval pipeline.

Note: In future production iterations, this rule-based classifier can be replaced
with a fine-tuned Clinical BioBERT or MuRIL text classifier.
"""

import re

# ── HIGH-CONFIDENCE health keywords ─────────────────────────────────────────
# These alone are strong enough to flag a claim as health-related
STRONG_HEALTH_KEYWORDS = {
    # Diseases / conditions
    "cancer", "diabetes", "hypertension", "autism", "dementia", "alzheimer",
    "asthma", "arthritis", "osteoarthritis", "tuberculosis", "malaria",
    "cholera", "typhoid", "dengue", "chikungunya", "hepatitis", "hiv", "aids",
    "covid", "covid-19", "sars", "ebola", "influenza", "pneumonia",
    "stroke", "seizure", "epilepsy", "parkinson", "sclerosis",
    "cystitis", "eczema", "psoriasis", "acne", "migraine",
    "syndrome", "disorder", "tumor", "carcinoma", "lymphoma", "leukemia",
    "ulcer", "hernia", "cataract", "glaucoma", "meningitis",
    # Medical interventions & products
    "vaccine", "vaccines", "vaccination", "antibiotic", "antibiotics",
    "chemotherapy", "chemo", "dialysis", "paxlovid", "insulin",
    "medicine", "medicines", "medication", "dosage", "drug", "drugs",
    "supplement", "supplements", "vitamin", "vitamins", "probiotic",
    "therapy", "treatment", "surgery", "transplant", "physiotherapy",
    # Clinical / medical terms
    "clinical", "diagnosis", "prognosis", "biopsy", "mri", "xray",
    "ecg", "blood pressure", "cholesterol", "hemoglobin", "platelet",
    "immune", "immunity", "inflammation", "antibody", "antigen",
    "pathogen", "bacteria", "viral", "virus", "infection", "infections",
    "symptom", "symptoms", "chronic", "acute", "benign", "malignant",
    # Body systems & organs (medical context)
    "blood", "heart", "lung", "lungs", "kidney", "liver", "prostate",
    "urinary", "lymph", "thyroid", "pancreas", "spleen", "colon",
    "ovarian", "uterine", "cervical",
    # Pregnancy / reproductive
    "pregnancy", "pregnant", "prenatal", "postnatal", "miscarriage",
    "fertility", "contraceptive",
    # Professionals & facilities
    "doctor", "doctors", "physician", "surgeon", "nurse", "pharmacist",
    "hospital", "hospitals", "clinic", "icu", "emergency",
    "patient", "patients",
    # Hinglish (strong medical terms)
    "bimari", "bimaari", "ilaj", "ilaaj", "dawa", "dawai",
    "aspatal", "khansi", "bukhar", "sehat", "dard",
    # Devanagari Hindi
    "कैंसर", "वैक्सीन", "टीका", "टीकाकरण", "दवा", "दवाई", "डॉक्टर", "अस्पताल",
    "बीमारी", "मधुमेह", "डायबिटीज", "बुखार", "एंटीबायोटिक", "कोविड", "स्वास्थ्य",
    "इलाज", "उपचार", "वायरस", "खांसी", "संक्रमण", "गर्भावस्था", "रक्त", "दिल"
}

# ── AMBIGUOUS keywords ──────────────────────────────────────────────────────
# These are health-relevant BUT too generic to stand alone.
# They only count if a HEALTH_CONTEXT_VERB or second strong keyword is also present.
AMBIGUOUS_HEALTH_KEYWORDS = {
    "sugar", "pain", "cold", "cough", "sore", "sores", "blister", "blisters",
    "skin", "gut", "stomach", "muscle", "joint", "joints", "knee", "back",
    "mask", "masks", "soap", "hygiene", "sanitizer", "sanitize", "handwash",
    "herbal", "remedy", "remedies", "heal", "healing", "prevent", "prevention",
    "preventive", "relieve", "alleviate", "fever", "health", "disease", "diseases",
    "herpes", "lysine", "cbd", "cannabis", "cannabidiol",
    "tobacco", "smoking", "cigarette",
    "haldi", "doodh", "chot",  # Hinglish
}

# ── Context verbs that signal a health claim ────────────────────────────────
# If an ambiguous keyword appears with one of these, the claim is health-related
HEALTH_CONTEXT_VERBS = {
    "cause", "causes", "caused", "causing",
    "treat", "treats", "treated", "treating", "treatment",
    "cure", "cures", "cured", "curing",
    "prevent", "prevents", "prevented", "preventing",
    "reduce", "reduces", "reduced", "reducing",
    "increase", "increases", "increased", "increasing",
    "diagnose", "diagnosed", "diagnosis",
    "affect", "affects", "affected",
    "help", "helps", "helped", "helping",
    "fight", "fights", "fighting",
    "risk", "risks",
    "effective", "ineffective", "beneficial", "harmful",
    "safe", "unsafe", "dangerous",
    "improve", "improves", "improved",
    "spread", "spreads",
    "kill", "kills",
    "absorb", "absorbs",
    "boost", "boosts",
    "lower", "lowers",
    "raise", "raises",
    "protect", "protects",
    "linked", "associated", "correlation",
    "symptom", "symptoms",
}

# ── Hard nonsense / non-health signal words ─────────────────────────────────
# If the claim contains these words alongside ambiguous keywords, it's not health-related
NONSENSE_SIGNALS = {
    "nasa", "rocket", "spaceship", "alien", "ufo", "planet", "galaxy",
    "sports", "football", "cricket", "basketball", "baseball",
    "movie", "film", "cinema", "actor", "actress",
    "bank", "money", "loan", "finance", "stock", "crypto", "bitcoin",
    "recipe", "cook", "cooking", "bake", "baking", "restaurant",
    "weather", "rain", "snow", "hurricane", "tornado",
    "politics", "government", "election", "vote", "parliament",
    "car", "drive", "driving", "traffic", "accident",
    "school", "college", "university", "exam", "study",
    "music", "song", "concert", "dance",
}


def is_health_related(text: str) -> bool:
    """
    Evaluates whether the input claim is related to health or medicine.

    Logic:
    1. Extract tokens from the input.
    2. If any STRONG_HEALTH_KEYWORD is matched → True (high confidence).
    3. If an AMBIGUOUS_HEALTH_KEYWORD is matched:
       a. But a NONSENSE_SIGNAL word is also present → False (incoherent, non-health context).
       b. And a HEALTH_CONTEXT_VERB is present → True (genuine health claim with ambiguous word).
       c. Otherwise → False (too vague to be a health claim).
    4. No keywords matched → False.
    """
    if not text or not isinstance(text, str):
        return False

    text_lower = text.lower().strip()

    # Extract alphanumeric and Devanagari tokens
    tokens = set(re.findall(r"[\w\u0900-\u097F]+", text_lower))

    # ── Step 1: Check for strong health keywords (standalone sufficient) ────
    strong_matches = tokens.intersection(STRONG_HEALTH_KEYWORDS)

    # Also check multi-word strong keywords (e.g., "blood pressure", "cold sore")
    for kw in STRONG_HEALTH_KEYWORDS:
        if " " in kw and kw in text_lower:
            strong_matches.add(kw)

    if strong_matches:
        # Even for strong matches, if there are nonsense signals and no
        # health context verb, it might be a garbled/nonsense query
        nonsense_matches = tokens.intersection(NONSENSE_SIGNALS)
        health_verb_matches = tokens.intersection(HEALTH_CONTEXT_VERBS)
        # Allow if strong match is a disease/condition name (not just generic "health")
        generic_strong = {"health", "disease", "diseases", "fever", "cough", "infection", "infections"}
        non_generic_strong = strong_matches - generic_strong
        if non_generic_strong:
            return True
        if nonsense_matches and not health_verb_matches:
            return False
        return True

    # ── Step 2: Check ambiguous health keywords ─────────────────────────────
    ambiguous_matches = tokens.intersection(AMBIGUOUS_HEALTH_KEYWORDS)
    if ambiguous_matches:
        nonsense_matches = tokens.intersection(NONSENSE_SIGNALS)
        health_verb_matches = tokens.intersection(HEALTH_CONTEXT_VERBS)

        # If nonsense context detected → reject
        if nonsense_matches:
            return False

        # Need a health context verb to confirm genuine health intent
        if health_verb_matches:
            return True

        # Ambiguous keyword alone, no health verb, no nonsense → check for
        # at least 2 ambiguous keywords (more likely a health discussion)
        if len(ambiguous_matches) >= 2:
            return True

        return False

    return False
