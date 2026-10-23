"""Clean raw search listings into an analysis-ready table.

Reads  : search_listings (raw, exactly as scraped)
Writes : data/processed/listings_clean.parquet

Every rule here is a judgement call, documented inline. The raw table is never
modified, so rules can be revised and this script re-run in seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.scraper.store import connect

OUT_PATH = Path("data/processed/listings_clean.parquet")
CURRENT_YEAR = 2026

# --- thresholds -------------------------------------------------------------
# Below this, listings are contact-bait: sellers put 1 DH to win the
# "sort by price ascending" filter. Same intent as "prix à débattre".
PRICE_MIN = 5_000
# Above this, values are Moroccan phone numbers typed into the price field
# (e.g. title "06XXXXXXXX", price 6XXXXXXXX) or extra-zero typos.
# A genuine Moroccan car ceiling is ~3-5M DH.
PRICE_MAX = 5_000_000

# Age-aware placeholder rule. Found via error analysis: the model's worst
# predictions were recent cars at absurd prices — a 2024 Tiguan R.LINE at
# 5,555 DH, a 2026 GWM Tank 500 at 6,300 DH, a 2022 Kia Picanto at 11,500 DH.
# Same contact-bait intent as the 1 DH listings above, just using less obvious
# numbers (5,555 / 10,101 / 11,111) that cleared the flat PRICE_MIN floor.
#
# Deliberately age-aware, not a higher flat floor: raising PRICE_MIN to 30,000
# would delete genuinely cheap old cars (a 1987 Peugeot 205 at 10,000 DH and a
# 1998 Renault R11 at 8,500 DH are real prices). The signal is price relative
# to age, not price alone.
#
# Cost: 130 rows = 0.57% of the dataset, and widening the band barely moves
# that (age<=8 & price<25,000 catches 119; price<40,000 catches 138), which
# says the cutoff sits in empty territory rather than slicing real listings.
PLACEHOLDER_MAX_AGE = 8
PLACEHOLDER_MIN_PRICE = 30_000

# Peer-ratio rule for absurdly HIGH prices. The mirror image of the
# placeholder rule above, and found the same way: a 2018 Golf appeared in
# the data at 2,700,000 DH, which the model duly learned from.
#
# The mechanism is Moroccan-specific and worth naming. One flagged row
# reads "Dacia Logan à vendre, prix convenable 13 millions" with a price
# of 1,300,000 — the seller means 13 million CENTIMES (1 DH = 100
# centimes), i.e. 130,000 DH, which is normal Moroccan speech for car
# prices. Entered into a dirham field, it inflates by ~10x. That is why
# the flagged ratios cluster tightly around 10-11x rather than spreading
# out.
#
# PRICE_MAX cannot catch these: 780,000 DH for a 2011 Dacia Logan is
# absurd but nowhere near the 5,000,000 phone-number ceiling. The signal
# is price relative to the same brand+model+year PEER GROUP, not price
# in absolute terms.
#
# Cost: 105 rows = 0.46%. Inspection showed zero legitimate cars in the
# flagged set. The threshold is safe because the distribution is bimodal:
# >3x flags 110 rows and >8x flags 91, so almost nothing sits in the
# "moderately expensive" zone — a genuine high-trim variant runs 2-3x its
# peer median, never 10x.
PEER_RATIO_MAX = 5.0
# Below this many peers, the group median is too noisy to judge against.
PEER_MIN_COUNT = 5

# A car older than this with sub-1000 km is not credible -> treat as missing.
IMPLAUSIBLE_MILEAGE_KM = 1_000
IMPLAUSIBLE_AGE_YEARS = 2
MILEAGE_MAX_KM = 800_000


def load_raw() -> pd.DataFrame:
    conn = connect()
    df = pd.read_sql("SELECT * FROM search_listings", conn)
    conn.close()
    return df


def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    """`year` is NOT numeric: 81 rows hold the string "1980 ou plus ancien".

    Naive pd.to_numeric would silently NaN exactly the oldest cars, where age
    matters most. Map to 1980 and flag it, so the model can learn that these
    are censored rather than precise.
    """
    raw = df["year"].astype("string")
    df["year_is_capped"] = raw.str.contains("ancien", case=False, na=False).astype(int)
    df["year"] = pd.to_numeric(raw.where(df["year_is_capped"] == 0, "1980"),
                               errors="coerce")
    df["age"] = CURRENT_YEAR - df["year"]
    return df


def clean_price(
    df: pd.DataFrame, placeholder_rule: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Drop rows with no usable target. You cannot impute y.

    Runs after clean_year because the placeholder rule below needs `age`.
    placeholder_rule=False reproduces the table from before that rule
    existed (used by cleaning_ablation.py).
    """
    n0 = len(df)
    dropped = {}

    df = df[df["price_mad"].notna()]
    dropped["no_price"] = n0 - len(df)

    n1 = len(df)
    df = df[df["price_mad"] >= PRICE_MIN]
    dropped["price_too_low"] = n1 - len(df)

    n2 = len(df)
    df = df[df["price_mad"] <= PRICE_MAX]
    dropped["price_too_high"] = n2 - len(df)

    # Recent car at an impossible price -> placeholder, not a real listing.
    # NaN age compares False, so rows with an unparseable year are kept
    # rather than silently dropped by a rule that cannot evaluate them.
    if not placeholder_rule:
        return df, dropped

    n3 = len(df)
    placeholder = (
        (df["age"] <= PLACEHOLDER_MAX_AGE)
        & (df["price_mad"] < PLACEHOLDER_MIN_PRICE)
    )
    df = df[~placeholder]
    dropped["placeholder_price"] = n3 - len(df)

    return df, dropped


def drop_peer_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop prices absurdly high relative to their brand+model+year peers.

    Needs brand/model, so it imports the extractors rather than relying on
    those columns already existing. They are computed here and dropped
    again -- clean.py's contract is to emit the raw-ish analysis table,
    and features.py owns brand/model as a feature concern.

    Rows in a peer group smaller than PEER_MIN_COUNT are never dropped:
    a median over 2 cars is not evidence, and dropping against it would
    be guessing rather than measuring.
    """
    from src.data.features import extract_brand, extract_model

    brand = df["title"].apply(extract_brand)
    model = [
        extract_model(t, b) if b else None
        for t, b in zip(df["title"], brand)
    ]
    tmp = df.assign(_brand=brand.values, _model=model)

    grp = tmp.groupby(["_brand", "_model", "year"])["price_mad"]
    peer_median = grp.transform("median")
    peer_count = grp.transform("count")

    outlier = (
        (peer_count >= PEER_MIN_COUNT)
        & (df["price_mad"] > PEER_RATIO_MAX * peer_median)
    ).fillna(False)

    return df[~outlier], int(outlier.sum())


def clean_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Implausible mileage becomes NaN, not an imputed value.

    NOTE: we do NOT multiply sub-1000 values by 1000. Inspection showed a 2001
    Bora at 336 km and a 2000 car at 17 km — these are blank/junk entries, not
    values expressed in thousands. Treating them as missing is honest; guessing
    a magnitude would fabricate data. LightGBM handles NaN natively.
    """
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

    implausible_low = (
        (df["mileage_km"] < IMPLAUSIBLE_MILEAGE_KM)
        & (df["age"] > IMPLAUSIBLE_AGE_YEARS)
    )
    implausible_high = df["mileage_km"] > MILEAGE_MAX_KM

    df["mileage_was_implausible"] = (implausible_low | implausible_high).astype(int)
    df.loc[implausible_low | implausible_high, "mileage_km"] = np.nan

    # Key feature: usage intensity. A 2015 car at 60k km is a different object
    # from a 2015 car at 300k km.
    df["km_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    return df


def clean_flags(df: pd.DataFrame) -> pd.DataFrame:
    """is_premium is 98.9% NULL — but there are zero explicit 0s.

    The source omits the key when false rather than sending false. So NULL means
    "not premium", not "unknown". Filling with 0 recovers a real feature
    instead of dropping the column.
    """
    for col in ("is_premium", "is_urgent", "is_highlighted",
                "is_shop", "is_car_checked"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df["is_dealer"] = (df["seller_type"].str.upper() == "STORE").astype(int)
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("fuel", "transmission", "city"):
        df[col] = df[col].astype("string").str.strip().str.title()

    # Frequency encoding: how liquid is this city's market?
    df["city_freq"] = df["city"].map(df["city"].value_counts())
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Train on log(price): car prices are heavily right-skewed, and errors
    should be proportional, not absolute. Exponentiate at predict time."""
    df["log_price"] = np.log1p(df["price_mad"])
    return df


def clean(
    verbose: bool = True, placeholder_rule: bool = True, peer_rule: bool = True
) -> pd.DataFrame:
    """Run every step. The two rule switches exist only so
    cleaning_ablation.py can rebuild the pre-fix table; leave them on."""
    df = load_raw()
    n_raw = len(df)

    df = clean_year(df)
    df, dropped = clean_price(df, placeholder_rule=placeholder_rule)
    if peer_rule:
        df, n_peer = drop_peer_outliers(df)
        dropped["peer_price_outlier"] = n_peer
    df = clean_mileage(df)
    df = clean_flags(df)
    df = clean_categoricals(df)
    df = add_targets(df)

    keep = [
        "listing_id", "url", "title",
        "price_mad", "log_price",
        "year", "age", "year_is_capped",
        "mileage_km", "km_per_year", "mileage_was_implausible",
        "fuel", "transmission",
        "city", "city_id", "city_freq",
        "seller_type", "is_dealer", "is_shop",
        "is_premium", "is_urgent", "is_highlighted", "is_car_checked",
        "n_photos", "search_page", "scraped_at",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    if verbose:
        print(f"raw rows                 {n_raw:>7,}")
        for reason, n in dropped.items():
            print(f"  dropped {reason:<18} {n:>7,}")
        print(f"clean rows               {len(df):>7,}  "
              f"({len(df)/n_raw:.1%} of raw)")
        print()
        print(f"price   median {df.price_mad.median():>10,.0f} DH   "
              f"IQR {df.price_mad.quantile(.25):,.0f}–{df.price_mad.quantile(.75):,.0f}")
        print(f"age     median {df.age.median():>10.0f} yrs")
        print(f"mileage median {df.mileage_km.median():>10,.0f} km   "
              f"missing {df.mileage_km.isna().mean():.1%}")
        print(f"dealers        {df.is_dealer.mean():>10.1%}")

    return df


if __name__ == "__main__":
    df = clean()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(df):,} rows, {len(df.columns)} cols)")