# Moroccan Used Car Price Prediction

I built a website that estimates what a used car is worth in Morocco, trained on about 22,500 public used-car listings from Moroccan classifieds that I collected myself.

Half of its estimates land within about 17,000 DH of the seller's asking price; the typical car in the data is listed at 138,000 DH.

Cleaning the data helped, but less than it first appeared: most of the early improvement came from removing junk listings no model could price, not from the model getting smarter.

**Live demo:** https://moroccan-car-prices.vercel.app — hosted on Vercel's free tier; the first request after a quiet period can take a few seconds while the model loads.

[![CI](https://github.com/AlaeMZ/moroccan-car-prices/actions/workflows/ci.yml/badge.svg)](https://github.com/AlaeMZ/moroccan-car-prices/actions/workflows/ci.yml)

**Typical error: 13.5% median APE** (25.9% mean) across 22,564 listings.

The data is self-collected — no Kaggle CSV. Collecting and cleaning it took the
most work. Three cleaning fixes affecting ~1% of rows took reported mean MAPE
from 33.6% to 25.9%, but **most of that drop came from removing unpredictable
junk rows from the test set. Scored on the same rows, the cleaner training data
is worth 1.3 points of mean MAPE**
([details](#how-much-did-the-cleaning-fixes-help)).

---

## Quickstart

```bash
uv sync
uv run uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

Or with Docker:

```bash
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api
```

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2019,
    "mileage_km": 100000,
    "fuel": "Diesel",
    "transmission": "Manuelle",
    "city": "Casablanca",
    "title": "Renault Clio Diesel Manuelle 2019"
  }'
```

```json
{
  "estimate_mad": 123300,
  "range_low_mad": 103000,
  "range_high_mad": 143500,
  "confidence": "high",
  "detected_brand": "Renault",
  "detected_model": "Clio",
  "typical_error_pct": 16.5
}
```

**The API returns a range, not a point estimate.** A single number like
"123,847 DH" implies a precision the model does not have. The range width comes
from measured out-of-fold error, and it widens automatically when the car cannot
be identified from its title (p68 error is 16.5% for identified cars vs 33.8%
otherwise).

---

## Pipeline

```
collection  public listings →  SQLite (private; not in this repository)
clean.py    SQLite        →  analysis parquet
features.py brand/model extraction + leakage fingerprint
train.py    Ridge / RandomForest / LightGBM on identical folds
app/main.py FastAPI service
```

Each boundary separates something expensive from something volatile. Collection
is rate-limited and slow; parsing is cheap and got rewritten six times. Because
the raw pages were stored first, all six rewrites cost zero extra requests. The
same principle repeats at the cleaning step: the raw table is never modified, so
a revised cleaning threshold is a seconds-long re-run rather than a re-collection.

---

## Data collection

**29,269 listings**, collected once, from a random sample of search-result pages.

The first plan was two passes: list the search pages, then open every listing.
Looking at the pages showed that each search page already carries complete,
structured records for the listings on it (price, year, mileage, fuel, gearbox,
city, seller type). One pass was enough: **about 40× fewer requests, so the
considerate approach turned out to be the efficient one.**

Pages were sampled at random rather than taken from the top. Search results are
ordered, and the order correlates with price: page 1 is dominated by promoted
listings that hide their price far more often, so taking the first pages would
have biased the data.

Parsing reads that structured data rather than the page layout. The layout is
redesigned often; the data model changes far less.

### Data handling

- Single-threaded collection with 2–3.5 s delays and an identifying User-Agent.
- Only allowlisted fields were read. Seller names and phone numbers were never
  extracted, and phone numbers were scrubbed from free-text descriptions at
  parse time (Moroccan personal-data law 09-08).
- No listing data is published here: no rows, titles, URLs or IDs. The
  collection code is kept private; I'm happy to walk through it in an interview.
- The live site shows only model estimates, never listings.

---

## What the data actually contained

None of this was visible from the schema. All of it came from reading values.

**1. Mileage units are inconsistent.** The same field holds `113` (a 2023 car
genuinely at 113 km), `200000`, and `120` meaning 120,000 — with no flag
distinguishing them. *Fix:* parse the formatted `fullValue` string rather than
the numeric field, because the formatting is unambiguous. Counterintuitively,
the display text carried more information than the clean number.

*If missed:* this destroys the mileage signal rather than inverting it.
Identical cars get values three orders of magnitude apart, no split separates
cheap from expensive, and the model quietly shifts weight onto year and brand.
SHAP would show mileage near zero — which reads like a finding ("mileage matters
less than expected in Morocco") but is a parsing bug.

**2. `1 DH` means "call me."** Every rock-bottom listing — Range Rovers, a 2024
VW — sits at exactly 1 DH, gaming the sort-by-price-ascending filter. Same
intent as "prix à débattre", different expression.

**3. Prices contain phone numbers.** e.g. price `6XXXXXXXX` where the title is
itself a phone number. Sellers embed numbers in the price field to dodge
the site's contact rules.

**4. Implausibly low mileage is junk, not thousands.** `"200 km"` on a 2013 car,
`"0 km"` on a 2000 car. The initial assumption was "expressed in thousands";
inspection disproved it. Treated as NaN — guessing a magnitude would fabricate
data, and LightGBM handles NaN natively.

**5. `is_premium` NULL means false, not unknown.** 98.9% NULL, and **zero**
explicit `0`s — the source omits the key entirely when false. So `fillna(0)` recovers
a real feature. Contrast with a NULL price, which is genuinely unknown and gets
the row dropped. Same NULL, opposite treatment; the difference comes from
understanding the source encoding, not from the statistics.

**6. Dealers hide prices 2.7× more often.** 48.7% of dealer listings omit a
price vs 18.2% for private sellers. This is **missing not at random** — the
missingness mechanism correlates with the target. Dropping unpriced rows takes
the dealer share from 8.2% to 5.4%, so the model is calibrated mainly to
private-seller pricing. Cannot be fixed by imputation; the prices do not exist
in the source.

**7. Sellers mix centimes and dirhams.** One flagged row reads *"Dacia Logan à
vendre, prix convenable 13 millions"* with a price of 1,300,000. The seller
means 13 million **centimes** — 130,000 DH — which is ordinary Moroccan speech
for car prices. Entered into a dirham field, it inflates by ~10×. This is why
the flagged outlier ratios cluster tightly around 10–11× their peer median
rather than spreading out.

---

## Cleaning

```
29,269   collected
-6,046   no price ("prix à débattre" — cannot impute a target)
-  313   price < 5,000 DH (contact-bait)
-  102   price > 5,000,000 DH (phone numbers, extra-zero typos)
-  139   placeholder prices (age-aware rule, below)
-  105   peer-ratio outliers (centimes/dirham confusion, below)
──────
22,564   usable (77.1% of raw)
```

**Every rule is measured, not reasoned about.** The price filters cost 1.8% of
priced rows — a diagnostic, not a budget. 1.8% means the cutoffs sit in empty
territory and catch only inspected garbage; 15% would mean they are slicing
through real listings and need moving.

### Two rules that came out of error analysis

Reading the model's worst predictions found data bugs that aggregate metrics
never would.

**Placeholder prices (age-aware).** The worst predictions were recent cars at
absurd prices: a 2024 Tiguan R.LINE at 5,555 DH, a 2026 GWM Tank 500 at 6,300
DH, a Maserati at 11,111 DH. Note the digit patterns. Same contact-bait as
finding 2, but with numbers just high enough to clear the flat 5,000 DH floor.
*The model was being penalised for correctly pricing a 2024 Tiguan at ~466,000
DH.*

The rule is `age <= 8 AND price < 30,000` — deliberately age-aware, because a
flat 30,000 floor would delete genuinely cheap old cars (a 1987 Peugeot 205 at
10,000 DH and a 1998 Renault R11 at 8,500 DH are real prices). The signal is
price *relative to age*, not price alone.

**Peer-ratio outliers.** The mirror image: a 2018 Golf in the data at 2,700,000
DH, which `PRICE_MAX` could not catch because it is nowhere near the 5,000,000
phone-number ceiling. The rule compares each price to the median of its own
brand+model+year group and drops anything above 5×, only where that group has at
least 5 members. Safe because the distribution is bimodal — `>3×` flags 110 rows
and `>8×` flags 91, so almost nothing sits in the "moderately expensive" zone. A
genuine high-trim variant runs 2–3× its peer median, never 10×.

### A rule that was measured and rejected

A sliding price floor (`floor = base − decay × age`) was tried to catch survivors
like a 2011 Golf at 12,000 DH. At `base=100,000, decay=6,000, min=15,000` it
removed 199 rows — 0.88%, a cost that looked as safe as every other rule.

Reading what it would delete killed it: a **2025 Dacia Logan at 80,000 DH**, a
**2026 Dacia Logan at 52,000 DH**, a **2022 DFSK K01s at 69,500 DH** — all real
prices. `base=100,000` wrongly assumes every new car costs at least 100,000 DH,
false for Morocco's budget segment (Dacia, DFSK, Chery). A cheap *new* Dacia and
a mispriced *new* Mercedes occupy the same price range; separating them needs
price relative to **brand**, not to age.

**Measuring the cost of a cleaning rule is necessary but not sufficient. You
also have to read what it removes.**

---

## Brand and model extraction

No brand or model column exists — both live only in the listing title. Extraction
is regex against a manufacturer list, longest-name-first so `Mercedes-Benz` wins
over `Mercedes`, with two fallbacks: a model→brand lookup for titles that drop
the brand (`"Golf 7 GTD"` → Volkswagen) and a small dictionary of observed typos.

**Coverage: 82.0% brand, 63.1% model** (76.9% of brand-known rows).

Three bugs found by inspecting output rather than trusting it:

- **Accent mismatch.** `re.IGNORECASE` normalises case but not accents — `Mégane`
  never matched `megane`. Fixed by stripping diacritics from both sides before
  matching.
- **Case fragmentation.** `extract_brand` returned whatever case the seller
  typed, so `Citroen` / `citroen` / `CITROEN` became three categories, silently
  splitting every brand in the dataset. Found by diffing a raw `str.contains`
  count against the extraction output. The same bug was later found in
  `extract_model` — `Classe C` / `CLASSE C` / `classe c` — where it was
  understating `model_frequency` and preventing test cars from matching their own
  model group. Fixing it moved 800+ rows into the best-predicted segment.
- **No-space concatenation.** `CitroënBerlingo`, `DaciaLogan`, `HyundaiTucson`
  fail the `\b` word boundary. Documented, not fixed — small n, and loosening the
  boundary would weaken the protection against false matches.

Model extraction is **brand-scoped**: it only searches the model list of the
brand already detected for that row. A global search would collide (`"Classe"`
means nothing outside Mercedes-Benz; a bare `"500"` could be a Fiat or a price
fragment). The cost is inheriting brand's coverage ceiling by design.

---

## Leakage

The same physical car is relisted repeatedly — a 3-page test sample found 15 of
114 listings (13%) appearing on more than one page. A naive `train_test_split`
would put one copy in train and another in test, and test MAPE would look
excellent while the model had memorised rather than learned.

**Fix:** a fingerprint of `brand|model|year|mileage_bucket|city`, with
`GroupKFold` grouped on it so all copies land on the same side of every split.
Mileage is bucketed to 10,000 km because the same car relisted weeks later shows
slightly different mileage; exact matching would miss real duplicates. The looser
bucket was chosen deliberately — missing a real duplicate inflates the test score,
which is the exact risk this exists to prevent, while over-grouping two different
cars only costs a little within-group diversity.

Rows missing brand or model get a fingerprint built from their own `listing_id`.
Without that, thousands of brand-less rows would share a `None|None|...` prefix
and collapse into giant fake duplicate groups — a worse error than the one being
fixed.

**Measured duplication: 5.7%** (1,275 rows). Treat as a rough floor: 37% of rows
have no model and can never be fingerprint-matched, while some flagged groups
(`Peugeot|3008|2018|90000|Casablanca` ×8) may be coincidence rather than
relisting.

*The first run reported 0.0% duplication and was correctly distrusted — with 537
Clios in the data, exactly zero collisions is implausible. The cause was a
dataframe passed without its brand column, so every row silently took the
"unique" fallback branch.*

---

## Modelling

**Metric: MAPE primary, median APE reported alongside, MAE secondary.** R²
easily reaches 0.85 and tells a user nothing. RMSE punishes large errors
quadratically, so a 600k DH error would dominate a 30k DH error and the model
would optimise for a small luxury segment — but being 10,000 DH wrong on a
30,000 DH Logan is a disaster, while the same absolute error on a 600,000 DH
Cayenne is fine. Percentage error matches how a user experiences being wrong.

The metric and split strategy were frozen **before** baselines were built, so
neither could be chosen to flatter a result.

Trained on `log1p(price)` and scored in dirhams via `expm1()`: log-space
training optimises proportional error, which is what MAPE measures, while every
reported number stays in real currency.

### Results

5-fold `GroupKFold`, identical folds for every model, all scored on the final
cleaned data (22,564 rows). ± is the standard deviation across folds.

| Model | mean MAPE | median APE | mean MAPE before cleaning fixes¹ |
|---|---|---|---|
| Global median baseline | 80.1% (±1.9) | 42.0% | 85.0% |
| Group median baseline | 54.9% (±2.3) | 15.5% | 62.9% |
| Ridge | 29.8% (±0.5) | 18.4% | 36.1% |
| RandomForest | 26.3% (±0.6) | 14.0% | 33.8% |
| **LightGBM** | **25.9% (±0.5)** | **13.5%** | 33.6% |

¹ The same models and folds-by-fingerprint protocol, run on the table as it was
before the three cleaning fixes (22,808 rows, rebuilt from the raw SQLite table
by `src/models/cleaning_ablation.py`). The 244 extra rows are mostly junk prices
no model can predict, so **this column is scored on a harder test set and is not
directly comparable with the others** — see the next section. (An earlier
version of this table showed 85.3% / 62.3% for the baselines; those came from a
single `GroupShuffleSplit`, not the 5-fold CV used for the models.)

**LightGBM and RandomForest are statistically indistinguishable** — the fold
standard deviations overlap. This is "tree ensembles beat linear
models", not "LightGBM won". Ridge underperforms both, consistent with the
non-linear depreciation curves in the EDA.

### How much did the cleaning fixes help?

The fixes (age-aware placeholder rule, canonical model casing, peer-ratio
outlier rule) were each found by reading the worst predictions, and together
they took LightGBM from 33.6% to 25.9% mean MAPE. That comparison is unfair to
the "before" model: two of the fixes delete rows, from the test folds as well
as the training folds, and the deleted rows are exactly the ones it got most
wrong.

To separate the two effects, `cleaning_ablation.py` builds one set of 5 folds
over the pre-fix rows and scores both pipelines on the same 22,564 rows that
survive the fixes:

| | Trained on | Scored on | mean MAPE | median APE |
|---|---|---|---|---|
| A | pre-fix data | all 22,808 pre-fix rows | 33.7% (±1.4) | 15.7% |
| B | pre-fix data | the 22,564 surviving rows | 27.4% (±0.5) | 15.5% |
| C | post-fix data | the 22,564 surviving rows | 26.1% (±0.2) | 14.0% |

- **Easier test set (A → B): 6.3 points.** Most of the headline gain is
  measurement — junk rows leaving the test folds.
- **Better training data (B → C): 1.3 points of mean MAPE, 1.5 of median APE.**
  Smaller, but real: C beats B on every fold (by 0.5 to 2.0 points).

(A reads 33.7% rather than 33.6% because these folds group on the fixed
fingerprint, so they differ slightly from the original run's.)

For scale, on the same final data Ridge → LightGBM is worth 3.9 points of mean
MAPE and RandomForest → LightGBM 0.4. So the cleaning fixes improved the model
more than choosing between tree ensembles did, but less than choosing trees over
a linear model.

### Where the models actually add value

The mean improved far more than the median over the group-median baseline
(54.9% → 25.9% vs 15.5% → 13.5%), and Ridge actually *lost* to the baseline on
median (18.4% vs 15.5%). Per-segment analysis explains why:

| Segment | Baseline MAPE | Model MAPE |
|---|---|---|
| brand + model + year known | 19.0% | 18.6% |
| brand + model | 101.7% | 29.3% |
| brand only | 102.8% | 32.2% |
| neither | 114.8% | 42.6% |

**The models rescue catastrophically mispriced rows rather than improving typical
predictions.** A group-median lookup has nothing to say about a car it cannot
identify; the model still has age, mileage, fuel and transmission to work with.
This was stated as a testable expectation before training and held.

### Feature engineering is fit on train only

`brand_tier` and `city_tier` are derived from average price per group. Computing
them across the full dataset would leak test prices into a training feature — the
same class of bug as the fingerprint work, arriving through a different door.
`FeatureBuilder` follows scikit-learn's fit/transform contract, and is fitted
**inside** each CV fold rather than once outside it.

Unseen categories at serve time get the *middle* tier, not 0 — zero would falsely
assert "cheapest tier", a claim with no evidence behind it.

### SHAP

`age` (0.415) dominates, then `transmission_Automatique` (0.162) and
`brand_tier` (0.137). `mileage_km` sits at 0.032 — present and contributing,
which is the specific check that finding 1's unit bug has not returned.

---

## Known limitations

1. **These are asking prices, not transaction prices.** Sellers list above
   what they accept. Spot-checks against live listings suggest predictions are
   reasonable but skew high for some segments.
2. **No trim or engine variant.** A `Golf GTD` and a base `Golf` are the same car
   to this model. For a Golf, published valuations span 298k–489k DH depending on
   configuration — most of that variance is invisible here. This is the single
   largest source of irreducible error.
3. **Placeholder prices remain in the 0–50k band** (85.7% mean vs 29.3% median —
   the gap is the fingerprint of remaining junk). Contact-bait on older cars
   survives the age-8 cutoff.
4. **Non-cars in the dataset.** Worst-prediction dumps surfaced an electric
   wheelchair, a cargo triporteur, a Kymco quad, spare parts, and apartment
   listings, all posted under `voitures_d_occasion`. These need a title-based
   category filter, not a price rule.
5. **Dealer pricing is under-sampled** (finding 6). Dealer MAPE is *better* than
   private (18.6% vs 26.3%), but these are only the 51% of dealers who disclosed
   a price.
6. **Arabic-script listings are never matched.** Titles like `بيجو 2008 ديزل`
   ("Peugeot 2008 diesel") are invisible to a Latin-alphabet regex. A meaningful
   share of the unmatched 18%.
7. **`Electrique` at 40.2% MAPE on 73 rows** — too thin to trust.

---

## Stack

Python 3.12, uv, SQLite, pandas, pyarrow, scikit-learn,
LightGBM, SHAP, FastAPI, Docker.

Dependencies are split so the served app stays small:

| Group | Packages | Needed for |
|---|---|---|
| default | fastapi, uvicorn, pandas, scikit-learn, lightgbm, joblib | serving the API |
| `training` | pyarrow, shap | reading the parquet data, `error_analysis.py` |
| `dev` | pytest, ruff, jupyter, matplotlib, seaborn | tests, notebooks |

The API's per-prediction "why this price" breakdown uses LightGBM's built-in
TreeSHAP (`pred_contrib=True`), not the shap package. The values are identical,
which `tests/test_shap_equivalence.py` checks.

## Layout

```
src/data/        clean.py, features.py
src/models/      baseline, feature_builder, train, train_final, predict,
                 error_analysis, cleaning_ablation
app/             FastAPI service (main.py) and the static frontend
tests/           pytest suite; fixtures/ holds synthetic cleaning data
notebooks/       01_eda.ipynb
models/          price_model.joblib
vendor/          libgomp.so.1, which LightGBM needs on Vercel
docs/            README media
```

## Reproducing

The trained model is in `models/price_model.joblib`, so the API and the site run
without any data:

```bash
uv sync
uv run uvicorn app.main:app --reload     # site at /, English at /en, API docs at /docs
uv run pytest                            # tests run on synthetic fixtures
```

The training scripts (`src/data/clean.py`, `src/models/train_final.py` and the
others under `src/models/`) document every step, but they need the raw listing
database, which is not published.

---

## License

MIT — see [LICENSE](LICENSE).
