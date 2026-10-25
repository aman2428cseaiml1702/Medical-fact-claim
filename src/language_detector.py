"""
Language Detector Module
Identifies Hindi (Devanagari script), Hinglish (Romanized Hindi),
and English using a hybrid rule-based and langdetect approach.
"""

import re

# Common Hinglish stopwords and keywords written in Latin script
HINGLISH_KEYWORDS = {
    "hai", "hain", "kya", "karta", "karti", "karte", "ke", "ki", "ka", "ko",
    "mein", "me", "se", "par", "bhi", "nahi", "nahin", "dawai", "dawa",
    "theek", "thik", "bimari", "ilaj", "kare", "karo", "peena", "peene",
    "khatam", "hota", "hoti", "hote", "hoga", "hogi", "baat", "aur", "ya",
    "doodh", "haldi", "ghee", "paani", "khana", "fayda", "nuksan", "wala",
    "wali", "apna", "apne", "dard", "garam", "thand"
}


def detect_language(text: str) -> str:
    """
    Detects the primary language of the input query:
    1. If Devanagari characters (\u0900-\u097F) are present -> 'Hindi'
    2. If Latin script contains common Hinglish markers -> 'Hinglish'
    3. Uses langdetect if available, defaulting to 'English'
    """
    if not text or not isinstance(text, str):
        return "English"

    # 1. Hindi Devanagari Script check
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    # 2. Hinglish check (Latin script with Hindi romanized vocabulary)
    clean_words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    hinglish_matches = clean_words.intersection(HINGLISH_KEYWORDS)

    # If at least one distinct Hinglish function/content word is detected
    if len(hinglish_matches) >= 1:
        return "Hinglish"

    # 3. Fallback to langdetect for other languages / standard English
    try:
        from langdetect import detect
        detected = detect(text)
        if detected == "hi":
            return "Hindi"
        elif detected == "en":
            return "English"
        else:
            return "English"
    except Exception:
        return "English"
