"""
src/models/train.py

Trains Ridge -> RandomForest -> LightGBM on IDENTICAL GroupKFold folds
and reports MAPE against the baseline floor.

Why GroupKFold here (vs GroupShuffleSplit in baseline.py):
    This is the model-comparison stage. All three models must see exactly
    the same folds or the comparison means nothing -- a difference in
    score could just be a difference in which rows landed where. Groups
    are the leakage fingerprints, so a relisted car never sits in train
    and validation simultaneously.

Why train on log_price but score in DH:
    EDA confirmed raw price is heavily right-skewed. Training on the log
    target makes the model optimise PROPORTIONAL error, which is what
    MAPE measures and what a user actually experiences. Predictions are
    converted back with expm1() before scoring, so every number reported
    is in real dirhams.

Why report mean AND median MAPE:
    The baseline diagnostic showed a huge gap (62.3% mean vs 17.5%
    median) -- a minority of badly-mispriced cheap cars drags the mean
    up. Quoting only the mean misrepresents typical performance; quoting
    only the median hides the tail. Both, always.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.append("src")
sys.path.append("src/models")

from data.features import add_model_column, build_fingerprint
from feature_builder import FeatureBuilder, build_matrix, align_columns

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (mean MAPE, median APE), both as percentages."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs((y_true - y_pred) / y_true) * 100
    return float(np.mean(ape)), float(np.median(ape))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds brand, model and fingerprint columns if missing."""
    if "model" not in df.columns:
        df = add_model_column(df)
    if "fingerprint" not in df.columns:
        df = build_fingerprint(df)
    return df


def run_cv(
    df: pd.DataFrame, model_name: str, make_model, n_splits: int = 5
) -> dict:
    """
    Runs GroupKFold CV for one model. Feature fitting happens INSIDE each
    fold, on that fold's training rows only -- fitting FeatureBuilder
    once outside the loop would leak validation prices into the features
    for every fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_means, fold_medians = [], []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, groups=df["fingerprint"]), start=1
    ):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        builder = FeatureBuilder()
        train_df = builder.fit_transform(train_df)
        val_df = builder.transform(val_df)

        X_train = build_matrix(train_df)
        X_val = build_matrix(val_df)
        X_train, X_val = align_columns(X_train, X_val)

        # median imputation, fitted on train only, for the same reason
        # everything else here is fitted on train only
        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)

        # Ridge needs scaled input; trees do not care. Scaling everything
        # keeps the pipeline uniform and does not hurt the tree models.
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_i)
        X_val_s = scaler.transform(X_val_i)

        y_train_log = np.log1p(train_df["price_mad"].values)
        y_val_true = val_df["price_mad"].values

        model = make_model()
        model.fit(X_train_s, y_train_log)
        y_pred_log = model.predict(X_val_s)
        y_pred = np.expm1(y_pred_log)

        # guard against nonsensical negative predictions leaking into MAPE
        y_pred = np.clip(y_pred, 1000, None)

        m, med = mape(y_val_true, y_pred)
        fold_means.append(m)
        fold_medians.append(med)
        print(f"  fold {fold}: mean MAPE {m:.1f}%  median APE {med:.1f}%")

    return {
        "model": model_name,
        "mean_mape": float(np.mean(fold_means)),
        "median_ape": float(np.mean(fold_medians)),
        "mean_mape_std": float(np.std(fold_means)),
    }


def main() -> None:
    df = pd.read_parquet("data/processed/listings_clean.parquet")
    df = prepare(df)
    print(f"Rows: {len(df)}   unique fingerprints: "
          f"{df['fingerprint'].nunique()}")
    print()

    specs = [
        ("Ridge", lambda: Ridge(alpha=1.0)),
        ("RandomForest", lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )),
    ]
    if HAS_LIGHTGBM:
        specs.append(("LightGBM", lambda: LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            random_state=42, n_jobs=-1, verbose=-1,
        )))
    else:
        print("lightgbm not installed -- skipping. Install with: "
              "uv add lightgbm")
        print()

    results = []
    for name, factory in specs:
        print(f"{name}:")
        results.append(run_cv(df, name, factory))
        print()

    print("=" * 70)
    print()
    print("BASELINE FLOOR (group median, from baseline.py):")
    print("  mean MAPE 62.3%   median APE 17.5%")
    print()
    print("MODEL RESULTS (5-fold GroupKFold, identical folds):")
    for r in results:
        print(f"  {r['model']:<14} mean MAPE {r['mean_mape']:.1f}% "
              f"(+/- {r['mean_mape_std']:.1f})   "
              f"median APE {r['median_ape']:.1f}%")
    print()
    print("A model only earns its complexity if it beats the group-median "
          "floor on BOTH numbers.")


if __name__ == "__main__":
    main()