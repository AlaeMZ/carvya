"""
One test per documented rule in src/data/clean.py, run through the real
clean() pipeline on the hand-made rows in tests/fixtures/raw_listings.csv.
Each fixture row's listing_id names the case it exercises.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.clean as clean_module
from src.data.clean import parse_mileage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cleaned():
    raw = pd.read_csv(FIXTURES / "raw_listings.csv", dtype={"year": str})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clean_module, "load_raw", lambda: raw.copy())
        df = clean_module.clean(verbose=False)
    return df.set_index("listing_id")


def test_no_price_rows_dropped(cleaned):
    assert "no_price" not in cleaned.index


def test_price_below_5000_dropped(cleaned):
    assert "price_1dh" not in cleaned.index
    assert "price_4999" not in cleaned.index
    assert "price_5000" in cleaned.index


def test_price_above_5_million_dropped(cleaned):
    assert "price_phone" not in cleaned.index
    assert "price_5m" in cleaned.index


def test_placeholder_rule_is_age_aware(cleaned):
    # age <= 8 and price < 30,000 -> placeholder, dropped
    assert "placeholder_new" not in cleaned.index
    assert "placeholder_age8" not in cleaned.index
    # a recent car exactly at the threshold is a real price
    assert "recent_at_30000" in cleaned.index
    # an old cheap car is a real price, not a placeholder
    assert "old_cheap_1987" in cleaned.index
    assert "age9_cheap" in cleaned.index


def test_peer_ratio_outlier_dropped_when_group_has_5_rows(cleaned):
    # Golf 2018: 4 normal rows + 1 at ~17x the group median -> dropped
    assert "golf_outlier" not in cleaned.index
    assert {"golf_1", "golf_2", "golf_3", "golf_4"} <= set(cleaned.index)


def test_peer_ratio_outlier_kept_when_group_too_small(cleaned):
    # Logan 2016: only 4 rows in the group, so the median is not trusted
    assert "logan_outlier" in cleaned.index


def test_mileage_parsed_from_full_value():
    # The numeric value said 120; the formatted text is unambiguous.
    assert parse_mileage("120,000 km") == 120_000
    assert parse_mileage("1 500 km") == 1_500


def test_implausibly_low_mileage_becomes_nan(cleaned):
    assert np.isnan(cleaned.loc["low_km_old", "mileage_km"])
    assert cleaned.loc["low_km_old", "mileage_was_implausible"] == 1
    # a nearly new car can genuinely have 113 km
    assert cleaned.loc["low_km_new", "mileage_km"] == 113
    assert cleaned.loc["low_km_new", "mileage_was_implausible"] == 0


def test_implausibly_high_mileage_becomes_nan(cleaned):
    assert np.isnan(cleaned.loc["high_km", "mileage_km"])
    assert cleaned.loc["high_km", "mileage_was_implausible"] == 1


def test_is_premium_null_becomes_zero(cleaned):
    assert cleaned["is_premium"].notna().all()
    assert cleaned.loc["golf_1", "is_premium"] == 0
    assert cleaned.loc["low_km_new", "is_premium"] == 1


def test_is_dealer_from_seller_type(cleaned):
    assert cleaned.loc["low_km_new", "is_dealer"] == 1
    assert cleaned.loc["golf_1", "is_dealer"] == 0
