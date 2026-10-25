"""
Text Preprocessing Module
Handles text cleaning, URL removal, punctuation normalization,
Devanagari Unicode preservation, and Hinglish variant normalization.
"""

import re
import unicodedata

# Common Hinglish variant mappings to canonical forms
HINGLISH_NORMALIZATION_MAP = {
    r"\bthik\b": "theek",
    r"\bteek\b": "theek",
    r"\bilaaj\b": "ilaj",
    r"\bdawa\b": "dawai",
    r"\bdawaai\b": "dawai",
    r"\bdawayi\b": "dawai",
    r"\bh\b": "hai",
    r"\bkartey\b": "karte",
    r"\bkarti\b": "karta",
    r"\bkarta\b": "karta",
    r"\bpeeney\b": "peena",
    r"\bpeene\b": "peena",
    r"\bkhatm\b": "khatam",
    r"\bmadhumeh\b": "diabetes",
    r"\bbimaari\b": "bimari",
    r"\bkaran\b": "cause",
    r"\bsehat\b": "health",
    r"\bfaida\b": "benefit",
    r"\bfayda\b": "benefit",
    r"\bnuksan\b": "harm",
}


def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes text for retrieval:
    1. Converts text to lowercase.
    2. Removes URLs, @mentions, hashtags, and excess whitespaces.
    3. Retains Devanagari Hindi characters and standard alphanumeric tokens.
    4. Applies Hinglish spelling normalization.
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization (NFKC ensures proper character representation)
    text = unicodedata.normalize("NFKC", text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove mentions (@username) and hashtags (#hashtag)
    text = re.sub(r"[@#]\S+", " ", text)

    # 3. Keep Hindi Devanagari range (\u0900-\u097F), Latin letters, numbers, and basic punctuation
    # Strip unnecessary symbols like emojis or noise while keeping Devanagari
    text = re.sub(r"[^\w\s\u0900-\u097F\.,?!-]", " ", text)

    # 4. Normalize common Hinglish spelling variants
    for pattern, replacement in HINGLISH_NORMALIZATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Collapse multiple whitespaces and strip boundaries
    text = re.sub(r"\s+", " ", text).strip()

    return text
