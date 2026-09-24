"""
Integration tests for src/models/predict.py against the real trained
artifact (models/price_model.joblib, committed to the repo).

These don't check exact price numbers -- the model can be retrained
with different numbers at any time. They check the CONTRACT predict.py
promises: a sane range around the point estimate, and a confidence
flag that actually reflects how much was known about the car, which is
the whole reason predict.py returns a range instead of a bare number.
"""

import pytest

from predict import predict_price

FULL_INFO = {
    "title": "Dacia Logan Diesel Manuelle 2018 à Casablanca",
    "year": 2018,
    "mileage_km": 120_000,
    "fuel": "Diesel",
    "transmission": "Manuelle",
    "city": "Casablanca",
}

NO_INFO = {
    "title": None,
    "year": 2015,
    "mileage_km": 180_000,
    "fuel": "Essence",
    "transmission": "Manuelle",
    "city": "Fès",
}


def test_predict_price_returns_expected_keys():
    result = predict_price(**FULL_INFO)
    assert set(result) == {
        "estimate_mad", "range_low_mad", "range_high_mad", "confidence",
        "detected_brand", "detected_model", "typical_error_pct",
    }


def test_range_brackets_the_point_estimate():
    result = predict_price(**FULL_INFO)
    assert result["range_low_mad"] < result["estimate_mad"] < result["range_high_mad"]


def test_range_and_estimate_are_positive():
    result = predict_price(**FULL_INFO)
    assert result["estimate_mad"] > 0
    assert result["range_low_mad"] > 0


def test_confidence_is_a_known_label():
    result = predict_price(**FULL_INFO)
    assert result["confidence"] in {"high", "medium", "low"}


def test_full_title_detects_brand_and_model():
    result = predict_price(**FULL_INFO)
    assert result["detected_brand"] == "Dacia"
    assert result["detected_model"] == "Logan"


def test_missing_title_yields_low_confidence_and_no_detection():
    # No title given -> brand/model can't be extracted -> the confidence
    # flag must say so rather than showing a falsely narrow range.
    result = predict_price(**NO_INFO)
    assert result["confidence"] == "low"
    assert result["detected_brand"] is None
    assert result["detected_model"] is None


def test_known_car_has_narrower_or_equal_range_than_unknown_car():
    # This is the core promise of the confidence flag: identifying the
    # car should never make the model's reported uncertainty go UP.
    known = predict_price(**FULL_INFO)
    unknown = predict_price(**NO_INFO)
    assert known["typical_error_pct"] <= unknown["typical_error_pct"]


@pytest.mark.parametrize("year", [1970, 2000, 2027])
def test_predict_price_handles_year_boundaries(year):
    result = predict_price(**{**FULL_INFO, "year": year})
    assert result["estimate_mad"] > 0
