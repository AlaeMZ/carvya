"""
src/models/train_final.py

Trains the production model on ALL data and saves everything the API
needs to serve predictions.

Different from train.py on purpose:
    train.py runs GroupKFold to COMPARE models -- that question is
    answered (tree ensembles beat Ridge; LightGBM and RandomForest are
    within fold noise). This trains one model on all rows and persists it.

What gets saved and why each piece is needed:
    model          the trained LightGBM regressor
    feature_builder the FITTED lookups (brand_tier, model_frequency...).
                   The API must use the exact same lookups the model was
                   trained with -- refitting on new data at serve time
                   would silently change what every feature means.
    imputer/scaler same reasoning: fitted transforms, not refitted ones
    columns        the exact column order build_matrix produced. One-hot
                   encoding at serve time on a single row will not
                   naturally produce the training schema, so the order is
                   frozen here and reapplied.
    residual_quantiles  out-of-fold |APE| quantiles, used to turn a point
                   estimate into a RANGE. Measured, not guessed: the
                   interval width comes from how wrong the model actually
                   was on data it had not seen.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

sys.path.append("src")
sys.path.append("src/models")

from data.features import add_model_column, build_fingerprint
from feature_builder import FeatureBuilder, build_matrix, align_columns

MODEL_DIR = Path("models")
ARTIFACT_PATH = MODEL_DIR / "price_model.joblib"

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def compute_residual_quantiles(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Out-of-fold absolute percentage errors, summarised as quantiles.

    Must be out-of-fold: in-sample residuals are far too optimistic
    (the model has seen those rows), which would produce confidence
    intervals that are far too narrow and quietly dishonest.

    Also computed per prediction-source segment, because error analysis
    showed the spread is very different for a fully-identified car
    (21.5% MAPE) versus one with no brand or model (42.4%). One global
    interval width would be too wide for the former and too narrow for
    the latter.
    """
    gkf = GroupKFold(n_splits=n_splits)
    df = df.reset_index(drop=True)
    oof_ape = np.zeros(len(df))
    has_full_info = np.zeros(len(df), dtype=bool)

    for train_idx, val_idx in gkf.split(df, groups=df["fingerprint"]):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        builder = FeatureBuilder()
        train_df = builder.fit_transform(train_df)
        val_df = builder.transform(val_df)

        X_train = build_matrix(train_df)
        X_val = build_matrix(val_df)
        X_train, X_val = align_columns(X_train, X_val)

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_i)
        X_val_s = scaler.transform(X_val_i)

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train_s, np.log1p(train_df["price_mad"].values))
        pred = np.clip(np.expm1(model.predict(X_val_s)), 1000, None)

        oof_ape[val_idx] = np.abs(
            (val_df["price_mad"].values - pred) / val_df["price_mad"].values
        ) * 100
        has_full_info[val_idx] = (
            val_df["brand"].notna() & val_df["model"].notna()
        ).values

    def quantiles_for(mask) -> dict:
        subset = oof_ape[mask]
        if len(subset) == 0:
            return {}
        return {
            "p50": float(np.percentile(subset, 50)),
            "p68": float(np.percentile(subset, 68)),
            "p80": float(np.percentile(subset, 80)),
            "n": int(len(subset)),
        }

    return {
        "full_info": quantiles_for(has_full_info),
        "partial_info": quantiles_for(~has_full_info),
        "overall": quantiles_for(np.ones(len(df), dtype=bool)),
    }


def main() -> None:
    df = pd.read_parquet("data/processed/listings_clean.parquet")
    if "model" not in df.columns:
        df = add_model_column(df)
    if "fingerprint" not in df.columns:
        df = build_fingerprint(df)

    print(f"Training on {len(df):,} rows")
    print()

    print("Computing out-of-fold residual quantiles (for prediction "
          "ranges)...")
    residuals = compute_residual_quantiles(df)
    for segment, stats in residuals.items():
        if stats:
            print(f"  {segment:<14} n={stats['n']:>6}  "
                  f"p50={stats['p50']:.1f}%  p68={stats['p68']:.1f}%  "
                  f"p80={stats['p80']:.1f}%")
    print()

    print("Fitting final model on all rows...")
    builder = FeatureBuilder()
    train_df = builder.fit_transform(df)

    X = build_matrix(train_df)
    imputer = SimpleImputer(strategy="median")
    X_i = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_i)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_s, np.log1p(train_df["price_mad"].values))

    artifact = {
        "model": model,
        "feature_builder": builder,
        "imputer": imputer,
        "scaler": scaler,
        "columns": list(X.columns),
        "residual_quantiles": residuals,
        "n_training_rows": len(df),
        "lgbm_params": LGBM_PARAMS,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    size_mb = ARTIFACT_PATH.stat().st_size / 1e6
    print(f"Saved {ARTIFACT_PATH}  ({size_mb:.1f} MB)")
    print()
    print("Artifact contains: model, fitted feature_builder, imputer, "
          "scaler, column order, residual quantiles.")


if __name__ == "__main__":
    main()