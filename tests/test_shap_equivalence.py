"""
Pins the claim in predict._build_explanation: LightGBM's built-in TreeSHAP
(booster.predict(..., pred_contrib=True)) gives the same per-feature
values as shap.TreeExplainer on this model.

The serving path dropped the shap package to keep the deployed bundle
small (shap pulls in numba + llvmlite, ~190 MB). If a LightGBM or shap
upgrade ever makes the two disagree, the "why this price" breakdown
would silently change meaning -- this test is what catches that.

shap lives in the `training` dependency group, so the comparison test
is skipped when only runtime deps are installed. CI installs all groups.
"""

import subprocess
import sys

import numpy as np
import pytest
from feature_builder import build_matrix
from predict import _row_from_input, load_artifact

FUELS = ["Diesel", "Essence", "Hybride", "Electrique", "Lpg", None]
TRANSMISSIONS = ["Manuelle", "Automatique", None]
CITIES = ["Casablanca", "Rabat", "Fès", "Nowhere", None]
TITLES = [
    "Renault Clio Diesel 2019", "Dacia Logan", "Mercedes-Benz Classe C",
    "Peugeot 208", "Volkswagen Golf 7", "voiture a vendre", None,
]


def _scaled_matrix(n_rows: int = 200) -> np.ndarray:
    """Random but reproducible inputs, pushed through the serving transform."""
    art = load_artifact()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_rows):
        row = _row_from_input(
            year=int(rng.integers(1978, 2027)),
            mileage_km=None if i % 11 == 0 else float(rng.integers(0, 450_000)),
            fuel=FUELS[i % len(FUELS)],
            transmission=TRANSMISSIONS[i % len(TRANSMISSIONS)],
            city=CITIES[i % len(CITIES)],
            title=TITLES[i % len(TITLES)],
            is_dealer=i % 2,
            n_photos=int(rng.integers(0, 25)),
        )
        X = build_matrix(art["feature_builder"].transform(row))
        for col in art["columns"]:
            if col not in X.columns:
                X[col] = 0
        rows.append(X[art["columns"]])
    X_all = np.vstack([r.to_numpy(dtype=float) for r in rows])
    return art["scaler"].transform(art["imputer"].transform(X_all))


def test_lightgbm_contribs_match_shap_tree_explainer():
    shap = pytest.importorskip("shap")
    art = load_artifact()
    X_s = _scaled_matrix()

    from_shap = np.asarray(shap.TreeExplainer(art["model"]).shap_values(X_s))
    # last column of pred_contrib is the bias (expected value), not a feature
    from_lgbm = art["model"].predict(X_s, pred_contrib=True)[:, :-1]

    assert from_lgbm.shape == from_shap.shape
    np.testing.assert_allclose(from_lgbm, from_shap, rtol=0, atol=1e-9)


def test_serving_path_does_not_import_shap():
    """
    Runs in a fresh interpreter so shap imported by the test above (or by
    anything else in the session) can't mask a regression.
    """
    code = (
        "import sys; sys.path[:0] = ['src', 'src/models'];"
        "from predict import predict_price;"
        "predict_price(year=2019, title='Renault Clio 2019');"
        "assert 'shap' not in sys.modules, 'serving path imported shap'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
