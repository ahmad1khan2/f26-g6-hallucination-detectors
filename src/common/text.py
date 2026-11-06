import unicodedata


def normalize(text: str) -> str:
    """NFKC, collapsed whitespace, lower-cased (a no-op for scripts without case)."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


def matches(answer: str, gold_list) -> bool:
    """True iff any non-empty gold answer is a substring of the answer (the base paper's rule)."""
    answer = normalize(answer)
    return any(g and g in answer for g in map(normalize, gold_list))


# NLTK Snowball stemmers for the languages we use that have one; the others use the crude fallback.
SNOWBALL_NAMES = {"en": "english", "de": "german", "fi": "finnish", "ar": "arabic"}
FALLBACK_MIN_KEEP = 5
FALLBACK_MAX_CUT = 3


def _stem_word(word: str, language: str) -> str:
    if language in SNOWBALL_NAMES:
        from nltk.stem.snowball import SnowballStemmer

        return SnowballStemmer(SNOWBALL_NAMES[language]).stem(word)
    # No stemmer: cut up to 3 final characters, never leaving fewer than 5 (ponytail: crude suffix guess).
    return word[: max(FALLBACK_MIN_KEEP, len(word) - FALLBACK_MAX_CUT)] if len(word) > FALLBACK_MIN_KEEP else word


def _stem_text(text: str, language: str) -> list:
    return [_stem_word(w, language) for w in normalize(text).split()]


def stem_match(answer: str, gold_list, language: str) -> bool:
    """Second label (used if a language's hand check is under 90%): answer equals a gold alias after stemming."""
    stemmed = _stem_text(answer, language)
    return any(stemmed and stemmed == _stem_text(g, language) for g in gold_list)
