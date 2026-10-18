import pytest

from src.common.text import matches, normalize


@pytest.mark.parametrize("answer, gold, expected", [
    ("The capital is Paris.", ["paris"], True),                 # English, case
    ("Die Hauptstadt ist  München", ["München"], True),          # German, spaces
    ("राजधानी नई दिल्ली है", ["नई दिल्ली"], True),                # Devanagari
    ("রাজধানী ঢাকা", ["ঢাকা"], True),                            # Bengali
    ("دارالحکومت لاہور ہے", ["لاہور"], True),                    # Urdu
    ("首都是東京", ["東京"], True),                                # Chinese, no spaces
    ("ＴＯＫＹＯ", ["tokyo"], True),                             # full-width via NFKC
    ("ਰਾਜਧਾਨੀ ਚੰਡੀਗੜ੍ਹ ਹੈ", ["ਚੰਡੀਗੜ੍ਹ"], True),                # Gurmukhi
    ("Nairobi ni mji mkuu", ["Dar es Salaam", "Dodoma"], False),
    ("anything", [], False),
    ("anything", [""], False),                                   # empty gold never matches
])
def test_matches(answer, gold, expected):
    assert matches(answer, gold) is expected


def test_normalize_collapses_whitespace():
    assert normalize("  A \n b\t") == "a b"
