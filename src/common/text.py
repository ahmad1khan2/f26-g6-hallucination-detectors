import unicodedata


def normalize(text: str) -> str:
    """NFKC, collapsed whitespace, lower-cased (a no-op for scripts without case)."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


def matches(answer: str, gold_list) -> bool:
    """True iff any non-empty gold answer is a substring of the answer (the base paper's rule)."""
    answer = normalize(answer)
    return any(g and g in answer for g in map(normalize, gold_list))
