"""
src/models/predict.py

Loads the saved artifact and turns car details into a price RANGE.

Why a range, not a point estimate:
    "147,382 DH" is false precision. The model's median error is ~15%,
    so a single number implies an accuracy that does not exist. A range
    derived from measured out-of-fold residuals is honest about what the
    model actually knows.

Why the confidence flag:
    Error analysis showed error depends heavily on how much is known
    about the car: 21.5% MAPE with brand+model+year, 42.4% with neither.
    A car the model can barely identify should say so, rather than
    returning a confident-looking range of the same width.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append("src")
sys.path.append("src/models")

from data.features import extract_brand, extract_model
from feature_builder import build_matrix

ARTIFACT_PATH = Path("models/price_model.joblib")

_artifact = None


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    """Loads once and caches -- the API should not reload per request."""
    global _artifact
    if _artifact is None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: uv run python "
                f"src/models/train_final.py"
            )
        _artifact = joblib.load(path)
    return _artifact


CURRENT_YEAR = 2026


def _row_from_input(
    year: int,
    mileage_km: float | None,
    fuel: str | None,
    transmission: str | None,
    city: str | None,
    title: str | None,
    is_dealer: int,
    n_photos: int,
) -> pd.DataFrame:
    """
    Builds a one-row dataframe matching the training schema.

    brand/model are extracted from the title with the same functions used
    in training. If no title is given, both are None -- which is a valid
    state the model saw plenty of during training (37% of rows had no
    model), and which the confidence flag will report honestly.
    """
    brand = extract_brand(title) if title else None
    model_name = extract_model(title, brand) if title and brand else None

    age = CURRENT_YEAR - year
    km_per_year = (
        mileage_km / max(age, 1) if mileage_km is not None else np.nan
    )

    return pd.DataFrame([{
        "brand": brand,
        "model": model_name,
        "year": year,
        "age": age,
        "mileage_km": mileage_km if mileage_km is not None else np.nan,
        "km_per_year": km_per_year,
        "fuel": fuel,
        "transmission": transmission,
        "city": city,
        "city_freq": np.nan,   # unknown at serve time; imputed
        "n_photos": n_photos,
        "is_dealer": is_dealer,
        "is_premium": 0,
        "is_urgent": 0,
        "is_highlighted": 0,
        "is_car_checked": 0,
        "year_is_capped": 1 if year <= 1980 else 0,
        "mileage_was_implausible": 0,
    }])


def predict_price(
    year: int,
    mileage_km: float | None = None,
    fuel: str | None = None,
    transmission: str | None = None,
    city: str | None = None,
    title: str | None = None,
    is_dealer: int = 0,
    n_photos: int = 5,
) -> dict:
    """
    Returns a dict with the point estimate, a range, and a confidence
    flag. The range half-width is the measured p68 out-of-fold APE for
    the relevant segment -- roughly "about two thirds of the time, the
    true price falls in here".
    """
    art = load_artifact()

    row = _row_from_input(
        year, mileage_km, fuel, transmission, city, title,
        is_dealer, n_photos,
    )

    row = art["feature_builder"].transform(row)
    X = build_matrix(row)

    # force the frozen training schema: add missing one-hot columns as 0,
    # drop anything unseen, reorder to match
    for col in art["columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[art["columns"]]

    X_i = art["imputer"].transform(X)
    X_s = art["scaler"].transform(X_i)

    point = float(np.expm1(art["model"].predict(X_s)[0]))
    point = max(point, 1000.0)

    has_full_info = (
        row["brand"].notna().iloc[0] and row["model"].notna().iloc[0]
    )
    segment = "full_info" if has_full_info else "partial_info"
    stats = art["residual_quantiles"].get(segment) or \
        art["residual_quantiles"]["overall"]

    half_width_pct = stats["p68"]
    low = point * (1 - half_width_pct / 100)
    high = point * (1 + half_width_pct / 100)

    if has_full_info and half_width_pct < 25:
        confidence = "high"
    elif has_full_info:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "estimate_mad": round(point, -2),
        "range_low_mad": round(max(low, 1000), -2),
        "range_high_mad": round(high, -2),
        "confidence": confidence,
        "detected_brand": row["brand"].iloc[0],
        "detected_model": row["model"].iloc[0],
        "typical_error_pct": round(half_width_pct, 1),
    }


if __name__ == "__main__":
    examples = [
        dict(title="Dacia Logan Diesel Manuelle 2018 à Casablanca",
             year=2018, mileage_km=120_000, fuel="Diesel",
             transmission="Manuelle", city="Casablanca"),
        dict(title="Mercedes-Benz Classe C 2020", year=2020,
             mileage_km=60_000, fuel="Diesel",
             transmission="Automatique", city="Rabat"),
        dict(title=None, year=2015, mileage_km=180_000, fuel="Essence",
             transmission="Manuelle", city="Fès"),
    ]
    for ex in examples:
        result = predict_price(**ex)
        print(f"input: {ex.get('title') or '(no title)'}  {ex['year']}")
        print(f"  {result['range_low_mad']:,.0f} - "
              f"{result['range_high_mad']:,.0f} DH "
              f"(est {result['estimate_mad']:,.0f}) "
              f"confidence={result['confidence']} "
              f"brand={result['detected_brand']}")
        print()