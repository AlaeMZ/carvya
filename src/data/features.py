"""
src/data/features.py

Brand extraction from the `title` column.

Why title only, not title + url slug:
    Inspection in Data Wrangler showed the url slug and title are the same
    underlying string — the slug is `title_with_underscores_LISTINGID.htm`.
    Matching against `title` alone captures everything the slug would.

Why substring match is enough, not just the "full template" pattern:
    Even seller-written titles ("Vends BYD Seal U Hybride en excellent état")
    still contain the brand name as a normal word. The match doesn't need
    the full {brand} {model} {fuel} {transmission} {year} à {city} template —
    it just needs the brand word to appear anywhere in the string.

Why longest-brand-first matching:
    "Mercedes" is a substring of "Mercedes-Benz". If "Mercedes" is checked
    first, "Mercedes-Benz GLC 220" would be tagged as brand="Mercedes",
    losing information and creating a spurious near-duplicate category next
    to any row that already says exactly "Mercedes". Sorting brands longest
    -> shortest before matching ensures the more specific name wins.
"""

import re
import unicodedata
import pandas as pd


def _strip_accents(s: str) -> str:
    """
    'Mégane' -> 'Megane', 'Citroën' -> 'Citroen'.

    re.IGNORECASE normalizes letter CASE (M <-> m) but does NOT strip
    accents -- 'é' is a genuinely different character from 'e', not a
    cased variant of it. Without this, accented seller spelling
    ('Mégane', 'à vendre') silently fails to match plain-ASCII brand/model
    entries in the dictionaries below. Applied to both the title being
    searched and the dictionary keys at pattern-build time, so accents
    never need to be hand-duplicated in MANUFACTURERS or MODEL_TO_BRAND.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

# Starter list — manufacturers actually present in the Moroccan used-car
# market. Treat this as a first pass, not gospel: measure coverage, then
# look at UNMATCHED titles to find what's missing (see coverage_report below).
MANUFACTURERS = [
    "Mercedes-Benz", "Mercedes", "Volkswagen", "Land Rover", "Range Rover",
    "Alfa Romeo", "Great Wall", "MG", "BMW", "Audi", "Renault", "Peugeot",
    "Citroen", "Citroën", "Dacia", "Fiat", "Ford", "Opel", "Skoda", "Seat",
    "Toyota", "Nissan", "Honda", "Hyundai", "Kia", "Mazda", "Mitsubishi",
    "Suzuki", "Subaru", "Volvo", "Jeep", "Chevrolet", "Chrysler", "Dodge",
    "Jaguar", "Mini", "Porsche", "Lexus", "Infiniti", "Acura", "Buick",
    "Cadillac", "GMC", "Lincoln", "Maserati", "Ferrari", "Lamborghini",
    "Bentley", "Rolls-Royce", "Aston Martin", "Bugatti", "Tesla", "BYD",
    "Geely", "Chery", "DFSK", "Changan", "JAC", "Haval", "Isuzu", "Daihatsu",
    "SsangYong", "Lada", "Iveco", "Smart", "DS", "Abarth", "Lancia",
]

# sort longest-first so "Mercedes-Benz" is tried before "Mercedes"
_MANUFACTURERS_SORTED = sorted(MANUFACTURERS, key=len, reverse=True)

# canonicalize spelling AND case variants to one label. Keyed by
# accent-stripped, LOWERCASED brand text so "Citroen"/"citroen"/"CITROEN"
# all resolve to one consistent output string -- without this, extract_brand
# returns whatever case the seller happened to type, which silently
# fragments a single brand into multiple categories in any downstream
# groupby (e.g. average price per brand).
_BRAND_CANONICAL: dict[str, str] = {}
for _b in MANUFACTURERS:
    _key = _strip_accents(_b).lower()
    _BRAND_CANONICAL.setdefault(_key, _b)  # first entry in list wins ties

# explicit overrides: variants that should collapse into ONE brand name
_BRAND_CANONICAL["mercedes"] = "Mercedes-Benz"
_BRAND_CANONICAL["mercedes-benz"] = "Mercedes-Benz"
_BRAND_CANONICAL["range rover"] = "Land Rover"
_BRAND_CANONICAL["rolls-royce"] = "Rolls-Royce"

# Second pass, used ONLY when no brand word itself was found in the title.
# These are model names common enough in Morocco that sellers write just the
# model, dropping the brand entirely (e.g. "Golf 7 GTD", "Classe A 180").
# Built from the unmatched-title sample, not guessed up front — extend this
# by rerunning coverage_report() and reading new unmatched samples.
MODEL_TO_BRAND = {
    "golf": "Volkswagen",
    "polo": "Volkswagen",
    "tiguan": "Volkswagen",
    "passat": "Volkswagen",
    "touran": "Volkswagen",
    "touareg": "Volkswagen",
    "classe a": "Mercedes-Benz",
    "classe b": "Mercedes-Benz",
    "classe c": "Mercedes-Benz",
    "classe e": "Mercedes-Benz",
    "classe s": "Mercedes-Benz",
    "classe": "Mercedes-Benz",  # fallback if no letter follows
    "ix35": "Hyundai",
    "santa fe": "Hyundai",
    "santafe": "Hyundai",
    "tucson": "Hyundai",
    "corolla": "Toyota",
    "yaris": "Toyota",
    "rav4": "Toyota",
    "clio": "Renault",
    "megane": "Renault",
    "captur": "Renault",
    "logan": "Dacia",
    "duster": "Dacia",
    "sandero": "Dacia",
    "308": "Peugeot",
    "208": "Peugeot",
    "3008": "Peugeot",
    "partner": "Peugeot",
}

# longest-first again, same reasoning as brand matching: "classe a" must be
# tried before the bare "classe" fallback
_MODELS_SORTED = sorted(
    (_strip_accents(m) for m in MODEL_TO_BRAND.keys()), key=len, reverse=True
)
_MODEL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _MODELS_SORTED) + r")\b",
    flags=re.IGNORECASE,
)

# Third pass, used ONLY when neither brand nor model matched. Explicit,
# observed typos -- not fuzzy matching. Fuzzy matching (edit distance) would
# catch more typos automatically but risks false-positive matches on
# unrelated words; an explicit list is slower to grow but has zero false
# positives. Extend this only with typos you've actually seen in unmatched
# samples, not guessed ones.
TYPO_TO_BRAND = {
    "wolkswagen": "Volkswagen",
    "dasia": "Dacia",
    "mercedess": "Mercedes-Benz",
    "hauday": "Hyundai",
}
_TYPO_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TYPO_TO_BRAND) + r")\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Model extraction — a SEPARATE, second feature from brand.
#
# Deliberately brand-scoped, not global: model names collide across
# unrelated meanings ("Classe" only means something for Mercedes-Benz;
# a bare "500" could be a Fiat 500 or just a mileage/price fragment).
# Searching only within the models list of the brand ALREADY detected for
# that row avoids these collisions entirely, at the cost of inheriting
# brand's coverage ceiling: no brand -> no model, by design.
#
# Coverage here will be built from actual real Moroccan-market volume
# (see the brand value_counts from the coverage report) — top models for
# the highest-volume brands first, extended the same way MANUFACTURERS
# was: measure, look at real unmatched titles, add what's missing.
# ---------------------------------------------------------------------------
BRAND_MODELS: dict[str, list[str]] = {
    "Volkswagen": ["Golf", "Polo", "Tiguan", "Passat", "Touran", "Touareg",
                   "Jetta", "Caddy", "Up", "Amarok", "T-Roc", "Sharan"],
    "Renault": ["Clio", "Megane", "Captur", "Symbol", "Kadjar", "Talisman",
                "Kangoo", "Twingo", "Scenic", "Fluence", "Koleos"],
    "Peugeot": ["308", "208", "3008", "2008", "508", "Partner", "5008",
                "206", "207", "306", "406", "Boxer", "301", "205"],
    "Mercedes-Benz": ["Classe A", "Classe B", "Classe C", "Classe E",
                      "Classe S", "GLC", "GLA", "GLE", "ML", "Vito",
                      "Sprinter", "CLA", "C220", "C180", "C200", "E220",
                      "E200", "Classe"],
    "Dacia": ["Logan", "Duster", "Sandero", "Lodgy", "Dokker", "Stepway"],
    "Hyundai": ["Tucson", "Santa Fe", "Accent", "i10", "i20", "i30",
                "ix35", "Elantra", "Creta", "Getz", "H100"],
    "Ford": ["Fiesta", "Focus", "Kuga", "EcoSport", "Mondeo", "Ranger",
             "Transit", "Puma", "Fusion", "Everest"],
    "Fiat": ["Doblo", "Punto", "500", "Tipo", "Panda", "Uno", "Palio"],
    "Audi": ["A3", "A4", "A5", "A6", "Q3", "Q5", "Q7", "A1", "TT"],
    "BMW": ["Serie 1", "Serie 2", "Serie 3", "Serie 4", "Serie 5",
            "Serie 7", "X1", "X3", "X5", "X6",
            "E46", "E60", "E90", "F30", "F10"],
    "Kia": ["Picanto", "Rio", "Sportage", "Sorento", "Cerato", "Ceed",
            "Stonic"],
    "Citroen": ["C3", "C4", "C5", "Berlingo", "Xsara", "Jumpy", "Nemo",
                "C-Elysee", "C1", "Picasso"],
    "Opel": ["Corsa", "Astra", "Insignia", "Mokka", "Vectra", "Zafira",
             "Crossland"],
    "Toyota": ["Corolla", "Yaris", "RAV4", "RAV-4", "Land Cruiser", "Hilux",
               "Camry", "Auris", "Prado"],
    "Land Rover": ["Range Rover", "Discovery", "Defender", "Evoque",
                   "Freelander", "Velar"],
    "Nissan": ["Qashqai", "Micra", "Juke", "Note", "X-Trail", "Navara"],
    "Seat": ["Ibiza", "Leon", "Arona", "Ateca"],
    "Jeep": ["Renegade", "Compass", "Grand Cherokee", "Cherokee",
             "Wrangler"],
    "Skoda": ["Fabia", "Octavia", "Superb", "Rapid", "Kodiaq"],
    "Honda": ["Civic", "CR-V", "Accord", "Jazz"],
    "Suzuki": ["Maruti", "Alto", "Celerio", "Swift", "Vitara", "Jimny",
               "Baleno"],
    "Volvo": ["V40", "XC90", "XC60", "S60", "V60"],
    "Porsche": ["Cayenne", "Macan", "Panamera", "911"],
    "Mini": ["Cooper", "Countryman"],
}

_BRAND_MODEL_PATTERNS: dict[str, re.Pattern] = {}
for _brand, _models in BRAND_MODELS.items():
    _sorted_models = sorted(
        (_strip_accents(m) for m in _models), key=len, reverse=True
    )
    _escaped = [re.escape(m) for m in _sorted_models]
    _BRAND_MODEL_PATTERNS[_brand] = re.compile(
        r"\b(" + "|".join(_escaped) + r")\b", flags=re.IGNORECASE
    )


def extract_model(title: str, brand: str | None) -> str | None:
    """
    Return the matched model name for a KNOWN brand, or None.

    Requires brand to already be known (see extract_brand) -- this is
    intentional. Searching a brand-scoped list avoids the cross-brand
    collisions a global model search would hit (e.g. "Classe" meaning
    nothing outside Mercedes-Benz).
    """
    if not isinstance(title, str) or brand not in _BRAND_MODEL_PATTERNS:
        return None

    stripped_title = _strip_accents(title)
    match = _BRAND_MODEL_PATTERNS[brand].search(stripped_title)
    return match.group(1) if match else None


def _build_pattern() -> re.Pattern:
    """
    One compiled regex, brand names as alternatives, longest first,
    word-boundaried so 'Kia' doesn't match inside 'Skia' or similar.
    Names are accent-stripped so the pattern matches both accented and
    unaccented seller spelling once the title is also stripped before search.
    """
    stripped_sorted = sorted(
        (_strip_accents(b) for b in MANUFACTURERS), key=len, reverse=True
    )
    escaped = [re.escape(brand) for brand in stripped_sorted]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


_BRAND_PATTERN = _build_pattern()


def extract_brand(title: str) -> str | None:
    """
    Return the matched manufacturer name (canonicalized), or None if no
    known brand or known model appears in the title.

    Pass 1: does a brand name itself appear? (unambiguous, always trusted)
    Pass 2: only if pass 1 found nothing, does a known MODEL name appear,
            inferring the brand from it. Deliberately second-priority:
            model names are more likely to collide with unrelated words
            than brand names are, so they're only trusted as a fallback.
    """
    if not isinstance(title, str):
        return None

    stripped_title = _strip_accents(title)

    match = _BRAND_PATTERN.search(stripped_title)
    if match:
        key = match.group(1).lower()
        return _BRAND_CANONICAL.get(key, match.group(1))

    model_match = _MODEL_PATTERN.search(stripped_title)
    if model_match:
        return MODEL_TO_BRAND[model_match.group(1).lower()]

    typo_match = _TYPO_PATTERN.search(stripped_title)
    if typo_match:
        return TYPO_TO_BRAND[typo_match.group(1).lower()]

    return None


def add_brand_column(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """
    Adds a `brand` column to df. Does not mutate the input in place.
    """
    df = df.copy()
    df["brand"] = df[title_col].apply(extract_brand)
    return df


def add_model_column(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """
    Adds a `model` column to df. Requires `brand` to already exist --
    calls add_brand_column first if it's missing. Does not mutate input.
    """
    df = df.copy()
    if "brand" not in df.columns:
        df = add_brand_column(df, title_col=title_col)
    df["model"] = df.apply(
        lambda row: extract_model(row[title_col], row["brand"]), axis=1
    )
    return df


def model_coverage_report(df: pd.DataFrame, title_col: str = "title") -> None:
    """
    Coverage is reported two ways, because they answer different questions:
      - overall: model_found / all_rows -- the number that matters for
        "how useful is this feature across my whole dataset"
      - of_known_brand: model_found / brand_known -- the number that
        isolates model-matching quality itself from brand's ceiling,
        useful for judging whether BRAND_MODELS needs more entries.
    """
    df = add_model_column(df, title_col=title_col)
    total = len(df)
    brand_known = df["brand"].notna().sum()
    model_found = df["model"].notna().sum()

    print(f"Model coverage (of all rows):        "
          f"{model_found}/{total} = {model_found/total*100:.1f}%")
    print(f"Model coverage (of brand-known rows): "
          f"{model_found}/{brand_known} = {model_found/brand_known*100:.1f}%")
    print()
    print("Model value counts (top 20):")
    print(df["model"].value_counts().head(20))
    print()

    # unmatched but brand IS known -- these are the rows worth reading,
    # since they're where BRAND_MODELS is missing an entry
    unmatched_known_brand = df.loc[
        df["brand"].notna() & df["model"].isna(), [title_col, "brand"]
    ]
    n = min(20, len(unmatched_known_brand))
    print(f"Sample of {n} titles with a KNOWN brand but no model match "
          f"(these tell you what to add to BRAND_MODELS):")
    for _, row in unmatched_known_brand.sample(n, random_state=42).iterrows():
        print(f"  - [{row['brand']}] {row[title_col]}")


def coverage_report(df: pd.DataFrame, title_col: str = "title") -> None:
    """
    Prints coverage stats and a sample of unmatched titles, so you can
    judge the extraction against the notes' thresholds:
        > 80% coverage -> strong feature
        < 50% coverage -> may add more noise than signal
    and so you can eyeball what's missing from MANUFACTURERS.
    """
    df = add_brand_column(df, title_col=title_col)
    total = len(df)
    matched = df["brand"].notna().sum()
    coverage = matched / total * 100

    print(f"Coverage: {matched}/{total} = {coverage:.1f}%")
    print()
    print("Brand value counts (top 20):")
    print(df["brand"].value_counts().head(20))
    print()

    unmatched = df.loc[df["brand"].isna(), title_col]
    print(f"Sample of {min(20, len(unmatched))} UNMATCHED titles "
          f"(look for missing brands or genuine seller-junk):")
    for t in unmatched.sample(min(20, len(unmatched)), random_state=42):
        print(f"  - {t}")


def build_fingerprint(
    df: pd.DataFrame, mileage_bucket_size: int = 10_000
) -> pd.DataFrame:
    """
    Adds a `fingerprint` column meant to identify the same PHYSICAL car
    across multiple listings (relistings), for GroupKFold splitting --
    so a car never appears in train and test simultaneously.

    fingerprint = brand|model|year|mileage_bucket|city

    Mileage is bucketed (rounded to mileage_bucket_size) rather than used
    exactly, because the same car relisted weeks apart will show a
    slightly different mileage each time -- exact matching would miss
    real duplicates. 10,000 km chosen deliberately over a tighter 5,000:
    missing a real duplicate (inflates test score, the actual leakage
    risk this exists to prevent) is worse than occasionally grouping two
    different cars with similar mileage together (costs a bit of
    within-group training diversity, not a correctness problem).

    Rows missing brand OR model get a fingerprint built from their own
    listing_id instead of the shared scheme. Without this, every
    brand-less/model-less row would collapse into the same handful of
    fingerprints and get falsely treated as duplicates of each other --
    a worse error than the one this function exists to fix.
    """
    df = df.copy()

    def _make_fp(row) -> str:
        if pd.isna(row.get("brand")) or pd.isna(row.get("model")):
            return f"unique_{row['listing_id']}"

        mileage = row.get("mileage_km")
        if pd.isna(mileage):
            bucket = "unknown_mileage"
        else:
            bucket = int(round(mileage / mileage_bucket_size) * mileage_bucket_size)

        return f"{row['brand']}|{row['model']}|{row.get('year')}|{bucket}|{row.get('city')}"

    df["fingerprint"] = df.apply(_make_fp, axis=1)
    return df


def leakage_report(df: pd.DataFrame) -> None:
    """
    Prints how much duplication the fingerprint actually catches, plus a
    sample of the biggest duplicate groups so you can eyeball whether
    they look like genuine relistings (same year/mileage/city, just
    reposted) or a false merge (bucketing/missing-brand accidentally
    lumping unrelated cars together).
    """
    if "model" not in df.columns:
        df = add_model_column(df)
    df = build_fingerprint(df)
    total = len(df)
    unique_fingerprints = df["fingerprint"].nunique()
    duplicated_rows = total - unique_fingerprints

    print(f"Total rows:            {total}")
    print(f"Unique fingerprints:   {unique_fingerprints}")
    print(f"Rows in a duplicate group (would-be leakage without "
          f"GroupKFold): {duplicated_rows} ({duplicated_rows/total*100:.1f}%)")
    print()

    group_sizes = df.groupby("fingerprint").size().sort_values(ascending=False)
    real_dupe_groups = group_sizes[
        (group_sizes > 1) & (~group_sizes.index.str.startswith("unique_"))
    ]
    print(f"Largest duplicate groups (top 10) -- inspect these for "
          f"plausibility:")
    for fp, size in real_dupe_groups.head(10).items():
        print(f"  x{size}  {fp}")


if __name__ == "__main__":
    # Adjust the path if your parquet lives elsewhere.
    df = pd.read_parquet("data/processed/listings_clean.parquet")
    coverage_report(df)
    print()
    print("=" * 70)
    print()
    model_coverage_report(df)
    print()
    print("=" * 70)
    print()
    leakage_report(df)