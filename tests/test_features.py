"""
Tests for src/data/features.py brand/model extraction.

These cover the behaviours the module's own docstrings call out as
deliberate design decisions (longest-brand-first, accent-insensitive
matching, model-only fallback, brand-scoped model search, typo
correction) -- the things most likely to silently regress if the
MANUFACTURERS / BRAND_MODELS dictionaries are edited later.
"""

from data.features import extract_brand, extract_model


def test_extract_brand_full_template():
    assert extract_brand("Dacia Logan Diesel Manuelle 2018 à Casablanca") == "Dacia"


def test_extract_brand_longest_match_wins():
    # "Mercedes" is a substring of "Mercedes-Benz" -- the more specific
    # name must win, not the shorter one that happens to match first.
    assert extract_brand("Mercedes-Benz GLC 220 2019") == "Mercedes-Benz"


def test_extract_brand_bare_mercedes_canonicalizes():
    # A seller writing just "Mercedes" should still collapse to the same
    # canonical brand as "Mercedes-Benz", not a separate category.
    assert extract_brand("Mercedes Classe C 2017") == "Mercedes-Benz"


def test_extract_brand_is_case_insensitive():
    assert extract_brand("VENDS bmw serie 3 2016") == "BMW"


def test_extract_brand_strips_accents():
    # Accented seller spelling ('Citroën') must still match the
    # plain-ASCII dictionary entry.
    assert extract_brand("Citroën C3 2015 à vendre") == "Citroen"


def test_extract_brand_falls_back_to_model_name():
    # No brand word appears at all -- only a known model name ("Golf"),
    # which should infer the brand.
    assert extract_brand("Golf 7 GTD 2017 impeccable") == "Volkswagen"


def test_extract_brand_corrects_known_typo():
    assert extract_brand("Wolkswagen Polo 2015") == "Volkswagen"


def test_extract_brand_returns_none_when_nothing_matches():
    assert extract_brand("Belle voiture familiale, bon état, prix négociable") is None


def test_extract_brand_returns_none_for_non_string():
    assert extract_brand(None) is None


def test_extract_model_requires_known_brand():
    # "Golf" is a Volkswagen model but the row's brand is Renault --
    # the search is brand-scoped, so this must not match.
    assert extract_model("Golf 7 GTD 2017", "Renault") is None


def test_extract_model_matches_within_correct_brand():
    assert extract_model("Volkswagen Golf 7 GTD 2017", "Volkswagen") == "Golf"


def test_extract_model_canonicalizes_case():
    assert extract_model("mercedes classe c 220 2020", "Mercedes-Benz") == "Classe C"


def test_extract_model_returns_none_when_brand_unknown():
    assert extract_model("Golf 7 2017", None) is None


def test_extract_model_returns_none_when_no_model_word_present():
    assert extract_model("Volkswagen 2017 bon état", "Volkswagen") is None
