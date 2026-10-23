"""
src/models/baseline.py

Leakage-safe train/test split + two baseline models, to establish the
floor any real model (Ridge/RF/LightGBM) has to beat.

Two evaluation modes:
    cv_baselines() scores both baselines with the same 5-fold GroupKFold
    that train.py uses for Ridge/RF/LightGBM, so the floor and the models
    are compared on identical folds. This is the number to report.

    baseline_report() keeps the original single GroupShuffleSplit plus the
    per-source diagnostic. Useful for reading WHY the group median fails,
    but its MAPE comes from a different split and is not comparable with
    the CV table.

Why two baselines, not one:
    Global median alone is nearly useless (predicts the same number for
    every car) but is the true floor -- any real feature-using model MUST
    beat it, or something is badly wrong. Group median is a much stronger,
    more honest baseline: "predict the typical price of similar cars."
    A LightGBM model that barely beats group-median isn't earning its
    complexity.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.append("src")
from data.features import add_model_column, build_fingerprint


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, as a percentage (e.g. 18.4, not 0.184).
    Matches the metric decision from the project notes: percentage error,
    not raw DH, so a 10,000 DH miss is judged relative to the car's price.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def leakage_safe_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits on the fingerprint groups from build_fingerprint, so the same
    physical car (or its relistings) can never appear in both train and
    test. Returns (train_df, test_df).
    """
    if "model" not in df.columns:
        df = add_model_column(df)
    df = build_fingerprint(df)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(df, groups=df["fingerprint"])
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Hard check, not a trust exercise: confirms zero fingerprint overlap
    between train and test. Prints PASS/FAIL rather than silently assuming
    the split worked.
    """
    train_fps = set(train_df["fingerprint"])
    test_fps = set(test_df["fingerprint"])
    overlap = train_fps & test_fps
    if overlap:
        print(f"FAIL: {len(overlap)} fingerprints appear in BOTH train "
              f"and test. Leakage check did not work.")
    else:
        print(f"PASS: zero fingerprint overlap between train "
              f"({len(train_df)} rows) and test ({len(test_df)} rows).")


def global_median_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Dumbest possible model: predict the same number (train median price)
    for every row in test, regardless of any feature. This is the true
    floor -- any real model that can't beat this is worse than useless.
    """
    global_median = train_df["price_mad"].median()
    predictions = np.full(len(test_df), global_median)
    return mape(test_df["price_mad"].values, predictions)


def group_median_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Predicts the median price of similar cars: same (brand, model, year)
    if available in train, falling back to (brand, model), then brand
    alone, then the global median -- so every test row gets SOME
    prediction even if its exact combo never appeared in train.
    """
    predictions = group_median_predict(train_df, test_df)
    return mape(test_df["price_mad"].values, predictions)


def group_median_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """The group-median predictions behind group_median_baseline."""
    global_median = train_df["price_mad"].median()

    by_bmy = train_df.groupby(["brand", "model", "year"])["price_mad"].median()
    by_bm = train_df.groupby(["brand", "model"])["price_mad"].median()
    by_brand = train_df.groupby("brand")["price_mad"].median()

    def predict_one(row) -> float:
        key_bmy = (row["brand"], row["model"], row["year"])
        if key_bmy in by_bmy.index:
            return by_bmy.loc[key_bmy]

        key_bm = (row["brand"], row["model"])
        if key_bm in by_bm.index:
            return by_bm.loc[key_bm]

        if row["brand"] in by_brand.index:
            return by_brand.loc[row["brand"]]

        return global_median

    return test_df.apply(predict_one, axis=1).values.astype(float)


def group_median_diagnostic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Breaks the group-median baseline apart to explain WHY its MAPE is
    what it is, instead of trusting the single number:
      - what fraction of test rows actually got a real group match vs
        silently fell back to the global median
      - mean vs median APE, to check whether a long tail of badly-priced
        cheap cars is dragging the MEAN up disproportionately
    """
    global_median = train_df["price_mad"].median()
    by_bmy = train_df.groupby(["brand", "model", "year"])["price_mad"].median()
    by_bm = train_df.groupby(["brand", "model"])["price_mad"].median()
    by_brand = train_df.groupby("brand")["price_mad"].median()

    def predict_with_source(row) -> tuple[float, str]:
        key_bmy = (row["brand"], row["model"], row["year"])
        if key_bmy in by_bmy.index:
            return by_bmy.loc[key_bmy], "brand+model+year"
        key_bm = (row["brand"], row["model"])
        if key_bm in by_bm.index:
            return by_bm.loc[key_bm], "brand+model"
        if row["brand"] in by_brand.index:
            return by_brand.loc[row["brand"]], "brand only"
        return global_median, "global fallback (no brand/model)"

    results = test_df.apply(predict_with_source, axis=1)
    test_df = test_df.copy()
    test_df["pred"] = [r[0] for r in results]
    test_df["source"] = [r[1] for r in results]
    test_df["ape"] = np.abs(
        (test_df["price_mad"] - test_df["pred"]) / test_df["price_mad"]
    ) * 100

    print("Prediction source breakdown (what fraction used real group info "
          "vs fell back to the global median):")
    print(test_df["source"].value_counts())
    print()
    print("MAPE by prediction source:")
    print(test_df.groupby("source")["ape"].mean().round(1))
    print()
    print(f"Overall MEAN APE:   {test_df['ape'].mean():.1f}%")
    print(f"Overall MEDIAN APE: {test_df['ape'].median():.1f}%  "
          f"(robust to outliers -- compare to the mean above)")


def cv_baselines(df: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    """
    Both baselines on 5-fold GroupKFold over the leakage fingerprint --
    the same folds train.py uses, because GroupKFold is deterministic for
    a given row order and grouping. Returns one dict per baseline in the
    same shape as train.run_cv.
    """
    if "model" not in df.columns:
        df = add_model_column(df)
    if "fingerprint" not in df.columns:
        df = build_fingerprint(df)

    scores = {"Global median": ([], []), "Group median": ([], [])}
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df, groups=df["fingerprint"]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        y_true = val_df["price_mad"].values.astype(float)

        preds = {
            "Global median": np.full(len(val_df), train_df["price_mad"].median()),
            "Group median": group_median_predict(train_df, val_df),
        }
        for name, y_pred in preds.items():
            ape = np.abs((y_true - y_pred) / y_true) * 100
            scores[name][0].append(float(np.mean(ape)))
            scores[name][1].append(float(np.median(ape)))

    return [
        {
            "model": name,
            "mean_mape": float(np.mean(means)),
            "median_ape": float(np.mean(medians)),
            "mean_mape_std": float(np.std(means)),
        }
        for name, (means, medians) in scores.items()
    ]


def baseline_report(df: pd.DataFrame) -> None:
    train_df, test_df = leakage_safe_split(df)
    verify_no_leakage(train_df, test_df)
    print()

    global_mape = global_median_baseline(train_df, test_df)
    group_mape = group_median_baseline(train_df, test_df)

    print(f"Global median baseline MAPE: {global_mape:.1f}%")
    print(f"Group median baseline MAPE:  {group_mape:.1f}%")
    print()
    print("Any real model (Ridge/RF/LightGBM) must beat the group-median "
          "number to justify its complexity.")
    print()
    print("=" * 70)
    print()
    group_median_diagnostic(train_df, test_df)


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/listings_clean.parquet")

    print("5-fold GroupKFold (same folds as train.py):")
    for r in cv_baselines(df):
        print(f"  {r['model']:<14} mean MAPE {r['mean_mape']:.1f}% "
              f"(+/- {r['mean_mape_std']:.1f})   "
              f"median APE {r['median_ape']:.1f}%")
    print()
    print("=" * 70)
    print()
    print("Single GroupShuffleSplit diagnostic (not comparable with the "
          "CV numbers above):")
    print()
    baseline_report(df)