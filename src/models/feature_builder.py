"""
src/models/feature_builder.py

Leakage-safe feature engineering.

Why a fit/transform class and not a plain add_features(df) function:
    brand_tier and city_tier are derived from AVERAGE PRICE per group.
    If those averages are computed over the full dataset, test-set prices
    leak into a training feature -- the model indirectly sees what test
    prices look like, and the score inflates for the same reason
    duplicate listings inflate it. Fitting on train only and applying
    those frozen lookups to test is the fix.

    This is the same class of bug as the fingerprint/GroupKFold work,
    arriving through a different door: there the leak was the same CAR in
    both splits; here it would be the same PRICE INFORMATION in both.
"""

import numpy as np
import pandas as pd


class FeatureBuilder:
    """
    Fit on train, transform train and test with the SAME frozen lookups.

    Features produced:
      brand_tier       ordinal 0-3, from train median price per brand
      model_frequency  how many train rows share this model (rarity signal)
      city_tier        ordinal 0-2, from train median price per city
      brand_known      1/0 -- was brand extractable at all
      model_known      1/0 -- was model extractable at all

    brand_known / model_known exist because the baseline diagnostic showed
    rows with no brand/model are the worst-predicted segment (107-120%
    MAPE). Making "we don't know this" an explicit feature lets the model
    learn to lean on age/mileage/fuel for exactly those rows, instead of
    treating a missing category as just another category.
    """

    def __init__(self, n_brand_tiers: int = 4, n_city_tiers: int = 3):
        self.n_brand_tiers = n_brand_tiers
        self.n_city_tiers = n_city_tiers
        self.brand_tier_map: dict = {}
        self.model_freq_map: dict = {}
        self.city_tier_map: dict = {}
        self.default_brand_tier: float = 0.0
        self.default_city_tier: float = 0.0
        self.default_model_freq: float = 0.0

    @staticmethod
    def _rank_to_tiers(medians: pd.Series, n_tiers: int) -> dict:
        """
        Assign ordinal tiers by rank, not by pd.qcut.

        qcut silently collapses everything into one bucket when it can't
        form clean quantile edges (few distinct groups, or ties), which
        would make the tier column a useless constant with no error
        raised. Ranking and slicing by position always produces the
        intended spread, and degrades predictably when there are fewer
        groups than tiers.
        """
        if len(medians) == 0:
            return {}

        ranked = medians.sort_values()
        n_groups = len(ranked)
        effective_tiers = min(n_tiers, n_groups)

        tier_assignments = {}
        for position, name in enumerate(ranked.index):
            tier = int(position * effective_tiers / n_groups)
            tier = min(tier, effective_tiers - 1)
            tier_assignments[name] = tier
        return tier_assignments

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """
        Learn all lookups from TRAIN ONLY. Never call this on test data
        or on the full dataset.
        """
        # brand tier: bucket brands by their median price into ordinal tiers
        brand_medians = train_df.groupby("brand")["price_mad"].median()
        self.brand_tier_map = self._rank_to_tiers(
            brand_medians, self.n_brand_tiers
        )

        # city tier: same idea, fewer buckets (EDA showed city separates
        # price only weakly -- ~0.3 log-spread vs fuel's ~1.4 -- so a
        # coarse 3-tier split is all this variable can support)
        city_medians = train_df.groupby("city")["price_mad"].median()
        self.city_tier_map = self._rank_to_tiers(
            city_medians, self.n_city_tiers
        )

        # model frequency: raw count in train. NOT a price -- a rarity
        # signal. Rare models have thin evidence behind their price, and
        # the model can learn to trust them less.
        self.model_freq_map = train_df["model"].value_counts().to_dict()

        # Defaults for categories never seen in train. Middle tier, not 0:
        # 0 would falsely tell the model "this is the cheapest tier",
        # which is a claim we have no evidence for. Middle = "no signal".
        self.default_brand_tier = (self.n_brand_tiers - 1) / 2
        self.default_city_tier = (self.n_city_tiers - 1) / 2
        self.default_model_freq = 0.0  # genuinely unseen == genuinely rare

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the frozen train-fitted lookups. Safe to call on train or
        test -- the lookups do not change here.
        """
        df = df.copy()

        df["brand_tier"] = (
            df["brand"].map(self.brand_tier_map).fillna(self.default_brand_tier)
        )
        df["city_tier"] = (
            df["city"].map(self.city_tier_map).fillna(self.default_city_tier)
        )
        df["model_frequency"] = (
            df["model"].map(self.model_freq_map).fillna(self.default_model_freq)
        )

        df["brand_known"] = df["brand"].notna().astype(int)
        df["model_known"] = df["model"].notna().astype(int)

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


# Columns fed to the models. Kept in one place so Ridge, RandomForest and
# LightGBM all train on an identical feature set -- otherwise the
# comparison between them is meaningless.
NUMERIC_FEATURES = [
    "age",
    "mileage_km",
    "km_per_year",
    "n_photos",
    "city_freq",
    "brand_tier",
    "city_tier",
    "model_frequency",
    "brand_known",
    "model_known",
    "year_is_capped",
    "mileage_was_implausible",
    "is_dealer",
    "is_premium",
    "is_urgent",
    "is_highlighted",
    "is_car_checked",
]

CATEGORICAL_FEATURES = [
    "fuel",
    "transmission",
]


def build_matrix(
    df: pd.DataFrame, categorical_dummies: bool = True
) -> pd.DataFrame:
    """
    Selects the model input columns and one-hot encodes the categoricals.

    categorical_dummies=True for Ridge (needs numeric input).
    LightGBM can consume raw categoricals natively, but keeping one
    encoding for all three models means the comparison is apples-to-apples
    -- which is the whole point of holding the feature set constant.
    """
    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[available_numeric].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    if categorical_dummies:
        available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if available_cat:
            dummies = pd.get_dummies(
                df[available_cat], prefix=available_cat, dummy_na=True
            )
            X = pd.concat([X, dummies.astype(int)], axis=1)

    return X


def align_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encoding can produce different columns for train vs test (a
    fuel type present in one but not the other). Align to the TRAIN
    columns: drop test-only columns, add missing ones as zeros. Train
    defines the schema, because that is what the model was fitted on.
    """
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    for col in missing_in_test:
        X_test[col] = 0

    extra_in_test = set(X_test.columns) - set(X_train.columns)
    X_test = X_test.drop(columns=list(extra_in_test))

    X_test = X_test[X_train.columns]
    return X_train, X_test