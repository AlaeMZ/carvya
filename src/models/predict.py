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
_explainer = None


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


def _get_explainer(art: dict):
    """
    Builds the SHAP TreeExplainer once and caches it, same reasoning as
    load_artifact: constructing it walks the whole booster, which is
    wasted work to repeat on every request.
    """
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(art["model"])
    return _explainer


CURRENT_YEAR = 2026

# Human-readable labels for the "why this price" breakdown. Deliberately
# excludes columns the form never actually lets a user set (is_premium,
# is_urgent, is_highlighted, is_car_checked are hardcoded to 0 in
# _row_from_input; year_is_capped/mileage_was_implausible are data-quality
# flags from scraped rows) -- showing those in a per-request explanation
# would look like a hidden factor the user has no way to act on.
FEATURE_LABELS: dict[str, str] = {
    "age": "Âge du véhicule",
    "mileage_km": "Kilométrage",
    "km_per_year": "Kilométrage annuel moyen",
    "n_photos": "Nombre de photos",
    "city_freq": "Popularité de la ville",
    "brand_tier": "Gamme de la marque",
    "city_tier": "Gamme de la ville",
    "model_frequency": "Popularité du modèle",
    "brand_known": "Marque identifiée",
    "model_known": "Modèle identifié",
    "is_dealer": "Type de vendeur",
}
CATEGORICAL_LABELS: dict[str, str] = {
    "fuel": "Carburant",
    "transmission": "Transmission",
}


def _humanize_feature(col: str) -> str | None:
    """
    Maps a raw model column name to a French label for display, or None
    if this column should not be shown to the end user (see FEATURE_LABELS
    docstring above).
    """
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]

    for prefix, label in CATEGORICAL_LABELS.items():
        if col.startswith(prefix + "_"):
            value = col[len(prefix) + 1:]
            if value == "nan":
                return None
            return f"{label} : {value}"

    return None


def _build_explanation(
    art: dict, X: pd.DataFrame, X_s: np.ndarray, columns: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Per-request SHAP breakdown of the top contributing factors, so the
    estimate isn't a black box. SHAP values live in log-price space (the
    model is trained on log1p(price)), so they aren't converted to a DH
    amount -- doing that accurately would require undoing a nonlinear
    transform per feature, which is exactly the kind of false precision
    this project avoids elsewhere. Instead each factor gets a direction
    and a relative magnitude bar, normalized against the strongest driver
    for THIS prediction.

    One-hot dummy columns need an extra filter X (the pre-scaling 0/1
    matrix) does not: a categorical feature's OTHER dummy columns (e.g.
    "transmission_Automatique" when the car is Manuelle) are 0 for this
    row and can still carry a nonzero SHAP value -- that's the model
    correctly using "not automatic" as information, but surfacing it
    under the label "Transmission : Automatique" would read as if the
    car WAS automatic. Only the dummy that is actually 1 for this row is
    eligible to represent that categorical feature.
    """
    explainer = _get_explainer(art)
    shap_values = np.asarray(explainer.shap_values(X_s))[0]

    items = []
    for col, value in zip(columns, shap_values):
        label = _humanize_feature(col)
        if label is None or value == 0:
            continue
        is_categorical_dummy = any(
            col.startswith(prefix + "_") for prefix in CATEGORICAL_LABELS
        )
        if is_categorical_dummy and X[col].iloc[0] != 1:
            continue
        items.append({"feature": label, "shap": float(value)})

    items.sort(key=lambda item: abs(item["shap"]), reverse=True)
    top = items[:top_n]
    if not top:
        return []

    max_abs = max(abs(item["shap"]) for item in top)
    return [
        {
            "feature": item["feature"],
            "direction": "up" if item["shap"] > 0 else "down",
            "magnitude": round(abs(item["shap"]) / max_abs, 3),
        }
        for item in top
    ]


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

    explanation = _build_explanation(art, X, X_s, art["columns"])

    return {
        "estimate_mad": round(point, -2),
        "range_low_mad": round(max(low, 1000), -2),
        "range_high_mad": round(high, -2),
        "confidence": confidence,
        "detected_brand": row["brand"].iloc[0],
        "detected_model": row["model"].iloc[0],
        "typical_error_pct": round(half_width_pct, 1),
        "explanation": explanation,
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