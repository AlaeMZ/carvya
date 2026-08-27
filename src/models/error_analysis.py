"""
src/models/error_analysis.py

Error analysis for the trained model. Answers WHERE the model fails, not
just how much -- a single MAPE number hides everything that matters.

Structure mirrors the questions actually worth asking:
  1. Did the model fix the segments the BASELINE was worst at?
     (baseline.py showed 'brand only' at 107% and 'global fallback' at
     120% -- the stated expectation was that age/mileage/fuel would
     rescue exactly those rows. This tests that claim rather than
     assuming it.)
  2. MAPE by price band -- cheap cars are mechanically punished harder by
     percentage error, so an overall number hides which band is failing.
  3. MAPE by seller type -- the Finding 7 (MNAR) test: dealers omit
     prices 2.7x more often, so the model is fitted mostly to
     private-seller pricing. If dealer MAPE is much worse, that
     selection bias is showing up in performance, not just in theory.
  4. The worst 30 predictions, dumped to CSV for MANUAL inspection.
     Reading actual bad rows finds bugs that aggregate metrics cannot.
  5. SHAP -- what the model actually leaned on, vs what you assumed.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def prediction_source(row, by_bmy, by_bm, by_brand) -> str:
    """
    Reproduces the baseline's fallback chain purely to LABEL each row with
    how much group information was available for it. Used to compare the
    model's per-segment performance against the baseline's on the same
    segments.
    """
    if pd.isna(row.get("brand")) or pd.isna(row.get("model")):
        if pd.isna(row.get("brand")):
            return "global fallback (no brand/model)"
        if row["brand"] in by_brand.index:
            return "brand only"
        return "global fallback (no brand/model)"

    if (row["brand"], row["model"], row["year"]) in by_bmy.index:
        return "brand+model+year"
    if (row["brand"], row["model"]) in by_bm.index:
        return "brand+model"
    if row["brand"] in by_brand.index:
        return "brand only"
    return "global fallback (no brand/model)"


def get_oof_predictions(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Out-of-fold predictions: every row is predicted by a model that never
    saw it in training. This is what makes per-segment analysis honest --
    using in-sample predictions would flatter every segment equally and
    tell you nothing.
    """
    gkf = GroupKFold(n_splits=n_splits)
    df = df.reset_index(drop=True)
    oof_pred = np.zeros(len(df))
    oof_source = np.empty(len(df), dtype=object)

    for train_idx, val_idx in gkf.split(df, groups=df["fingerprint"]):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # label validation rows by how much group info train had for them
        by_bmy = train_df.groupby(["brand", "model", "year"])["price_mad"].median()
        by_bm = train_df.groupby(["brand", "model"])["price_mad"].median()
        by_brand = train_df.groupby("brand")["price_mad"].median()
        oof_source[val_idx] = val_df.apply(
            lambda r: prediction_source(r, by_bmy, by_bm, by_brand), axis=1
        ).values

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

        model = LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train_s, np.log1p(train_df["price_mad"].values))
        pred = np.expm1(model.predict(X_val_s))
        oof_pred[val_idx] = np.clip(pred, 1000, None)

    out = df.copy()
    out["predicted"] = oof_pred
    out["prediction_source"] = oof_source
    out["ape"] = np.abs(
        (out["price_mad"] - out["predicted"]) / out["price_mad"]
    ) * 100
    out["error_dh"] = out["predicted"] - out["price_mad"]
    return out


def segment_report(oof: pd.DataFrame) -> None:
    """Prints MAPE broken down every way that matters."""
    print("1. MAPE BY PREDICTION SOURCE")
    print("   (baseline numbers for comparison: brand+model+year 25.9%, "
          "brand+model 77.0%, brand only 107.1%, global fallback 120.0%)")
    seg = oof.groupby("prediction_source")["ape"].agg(
        ["count", "mean", "median"]
    ).round(1)
    print(seg)
    print()

    print("2. MAPE BY PRICE BAND")
    bands = pd.cut(
        oof["price_mad"],
        bins=[0, 50_000, 150_000, 400_000, np.inf],
        labels=["0-50k", "50-150k", "150-400k", "400k+"],
    )
    print(oof.groupby(bands, observed=True)["ape"].agg(
        ["count", "mean", "median"]
    ).round(1))
    print()

    print("3. MAPE BY SELLER TYPE (the Finding 7 / MNAR test)")
    if "is_dealer" in oof.columns:
        print(oof.groupby("is_dealer")["ape"].agg(
            ["count", "mean", "median"]
        ).round(1))
        print("   is_dealer 0 = private, 1 = dealer. A large gap means "
              "the dealer price-hiding bias affects performance, not "
              "just theory.")
    print()

    print("4. MAPE BY FUEL TYPE")
    if "fuel" in oof.columns:
        print(oof.groupby("fuel")["ape"].agg(
            ["count", "mean", "median"]
        ).round(1).sort_values("mean"))
    print()


def worst_predictions(oof: pd.DataFrame, n: int = 30,
                      out_path: str = "data/processed/worst_predictions.csv") -> None:
    """
    Dumps the n worst predictions to CSV. This is the step that finds
    data bugs -- aggregate metrics never will. Read these by hand.
    """
    cols = [c for c in [
        "listing_id", "title", "brand", "model", "year", "age",
        "mileage_km", "fuel", "transmission", "city", "is_dealer",
        "price_mad", "predicted", "error_dh", "ape", "prediction_source",
    ] if c in oof.columns]

    worst = oof.nlargest(n, "ape")[cols]
    worst.to_csv(out_path, index=False)
    print(f"5. WORST {n} PREDICTIONS -> {out_path}")
    print("   Read these by hand. Look for: prices that are actually "
          "phone numbers, mileage that survived cleaning but is wrong, "
          "titles that reveal a damaged/salvage car the features cannot "
          "see, or genuine market outliers.")
    print()
    preview_cols = [c for c in ["title", "price_mad", "predicted", "ape"]
                    if c in worst.columns]
    print(worst[preview_cols].head(10).to_string(index=False))
    print()


def shap_report(df: pd.DataFrame) -> None:
    """
    SHAP importances on a single train/validation split. Answers "what
    did the model actually use", which is worth checking against what you
    assumed it would use.
    """
    if not HAS_SHAP:
        print("6. SHAP -- skipped, shap not installed (uv add shap)")
        return

    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(df, groups=df["fingerprint"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    builder = FeatureBuilder()
    train_df = builder.fit_transform(train_df)
    val_df = builder.transform(val_df)

    X_train = build_matrix(train_df)
    X_val = build_matrix(val_df)
    X_train, X_val = align_columns(X_train, X_val)

    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_val_i = pd.DataFrame(
        imputer.transform(X_val), columns=X_val.columns
    )

    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train_i, np.log1p(train_df["price_mad"].values))

    explainer = shap.TreeExplainer(model)
    sample = X_val_i.sample(min(2000, len(X_val_i)), random_state=42)
    shap_values = explainer.shap_values(sample)

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=sample.columns
    ).sort_values(ascending=False)

    print("6. SHAP FEATURE IMPORTANCE (mean |SHAP|, top 15)")
    print(importance.head(15).round(4).to_string())
    print()
    print("   Sanity checks worth making: is mileage_km near zero? That "
          "would signal the unit bug returned. Is age doing the work you "
          "expect? Did brand_tier earn its place?")
    print()


def main() -> None:
    df = pd.read_parquet("data/processed/listings_clean.parquet")
    if "model" not in df.columns:
        df = add_model_column(df)
    if "fingerprint" not in df.columns:
        df = build_fingerprint(df)

    print("Computing out-of-fold predictions...")
    print()
    oof = get_oof_predictions(df)

    print(f"Overall: mean MAPE {oof['ape'].mean():.1f}%   "
          f"median APE {oof['ape'].median():.1f}%")
    print("=" * 70)
    print()

    segment_report(oof)
    worst_predictions(oof)
    shap_report(df)


if __name__ == "__main__":
    main()