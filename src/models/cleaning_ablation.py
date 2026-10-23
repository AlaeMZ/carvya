"""
src/models/cleaning_ablation.py

How much of the "cleaning fixes improved mean MAPE" result is real?

Three fixes landed between the first LightGBM run and the final one:
    1. the age-aware placeholder-price rule      (clean.py)
    2. canonical casing in extract_model         (features.py)
    3. the peer-ratio outlier rule               (clean.py)

Rules 1 and 3 delete rows, from the test folds as well as the training
folds, and the rows they delete are the ones the model gets most wrong.
So comparing "before" and "after" MAPE mixes two effects:
    (a) the model learned from better data
    (b) the hardest rows were removed from the test set

This script separates them by scoring two pipelines on the SAME rows:

    A  train on pre-fix data,  score on pre-fix val rows      (as reported)
    B  train on pre-fix data,  score on val rows that survive the fixes
    C  train on post-fix data, score on val rows that survive the fixes

A -> B is effect (b), B -> C is effect (a). Folds are one 5-fold
GroupKFold over the pre-fix rows, grouped on the fixed fingerprint so a
relisted car never straddles train and validation in either pipeline.

It also reruns every model on the pre-fix data with its own folds, which
reproduces the README's original "before cleaning fixes" numbers.

Run from the repo root (needs the raw SQLite DB):
    uv run python -m src.models.cleaning_ablation
"""

import sys

import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.append("src")
sys.path.append("src/models")

from baseline import cv_baselines
from train import fit_predict, mape, model_specs, run_cv

from data.features import (
    _BRAND_MODEL_PATTERNS,
    _strip_accents,
    add_brand_column,
    add_model_column,
    build_fingerprint,
)
from src.data.clean import clean


def extract_model_case_preserving(title, brand):
    """
    extract_model as it was before fix 2: returns the matched text in
    whatever case the seller typed, so "Clio" and "CLIO" are different
    models. Kept only to rebuild the pre-fix table.
    """
    if not isinstance(title, str) or brand not in _BRAND_MODEL_PATTERNS:
        return None
    match = _BRAND_MODEL_PATTERNS[brand].search(_strip_accents(title))
    return match.group(1) if match else None


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pre, post), both with brand/model/fingerprint columns.

    pre carries two extra columns: model_fixed and fingerprint_fixed, the
    post-fix extraction applied to the pre-fix rows, used for grouping.
    """
    post = pd.read_parquet("data/processed/listings_clean.parquet")
    post = build_fingerprint(add_model_column(post))

    pre = clean(verbose=False, placeholder_rule=False, peer_rule=False)
    pre = add_brand_column(pre)
    pre["model"] = [
        extract_model_case_preserving(t, b)
        for t, b in zip(pre["title"], pre["brand"])
    ]
    pre = build_fingerprint(pre)

    fixed = build_fingerprint(add_model_column(pre.drop(columns=["model"])))
    pre["model_fixed"] = fixed["model"].values
    pre["fingerprint_fixed"] = fixed["fingerprint"].values

    missing = set(post["listing_id"]) - set(pre["listing_id"])
    assert not missing, f"{len(missing)} post-fix rows absent from pre-fix rebuild"
    return pre, post


def decompose(pre: pd.DataFrame, post: pd.DataFrame, make_model,
              n_splits: int = 5) -> pd.DataFrame:
    """Per-fold mean/median APE for pipelines A, B and C."""
    post_by_id = post.set_index("listing_id")
    survives = pre["listing_id"].isin(post_by_id.index).values

    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(pre, groups=pre["fingerprint_fixed"]), start=1
    ):
        pre_train, pre_val = pre.iloc[train_idx], pre.iloc[val_idx]
        val_survives = survives[val_idx]

        post_train = post_by_id.loc[
            pre_train["listing_id"][survives[train_idx]]
        ].reset_index()
        post_val = post_by_id.loc[
            pre_val["listing_id"][val_survives]
        ].reset_index()

        # the post-fix pipeline's own fingerprints must not leak either
        assert not set(post_train["fingerprint"]) & set(post_val["fingerprint"])

        pred_pre = fit_predict(pre_train, pre_val, make_model)
        pred_post = fit_predict(post_train, post_val, make_model)
        y_pre = pre_val["price_mad"].values

        for name, y, p in [
            ("A", y_pre, pred_pre),
            ("B", y_pre[val_survives], pred_pre[val_survives]),
            ("C", post_val["price_mad"].values, pred_post),
        ]:
            m, med = mape(y, p)
            rows.append({"fold": fold, "pipeline": name,
                         "mean_mape": m, "median_ape": med, "n": len(y)})
    return pd.DataFrame(rows)


def main() -> None:
    pre, post = build_tables()
    print(f"pre-fix rows {len(pre):,}   post-fix rows {len(post):,}   "
          f"removed by fixes {len(pre) - len(post):,}")
    print()

    print("BEFORE CLEANING FIXES -- every model on the pre-fix table,")
    print("its own fingerprints and folds (the original setup):")
    before = cv_baselines(pre)
    for name, factory in model_specs():
        print(f"{name}:")
        before.append(run_cv(pre, name, factory))
    print()
    for r in before:
        print(f"  {r['model']:<14} mean MAPE {r['mean_mape']:.1f}% "
              f"(+/- {r['mean_mape_std']:.1f})   "
              f"median APE {r['median_ape']:.1f}%")
    print()
    print("=" * 70)
    print()

    lgbm = dict(model_specs())["LightGBM"]
    folds = decompose(pre, post, lgbm)
    summary = folds.groupby("pipeline").agg(
        mean_mape=("mean_mape", "mean"),
        mean_mape_std=("mean_mape", "std"),
        median_ape=("median_ape", "mean"),
        val_rows=("n", "sum"),
    )
    print("DECOMPOSITION (LightGBM, shared folds):")
    print("  A  pre-fix train,  all pre-fix val rows")
    print("  B  pre-fix train,  surviving val rows only")
    print("  C  post-fix train, surviving val rows only")
    print(summary.round(2).to_string())
    print()

    wide = folds.pivot(index="fold", columns="pipeline", values="mean_mape")
    test_effect = wide["A"] - wide["B"]
    train_effect = wide["B"] - wide["C"]
    print(f"  (b) easier test set   A-B = {test_effect.mean():.2f} pts "
          f"(per fold {', '.join(f'{x:.1f}' for x in test_effect)})")
    print(f"  (a) better train data B-C = {train_effect.mean():.2f} pts "
          f"(per fold {', '.join(f'{x:.1f}' for x in train_effect)})")
    print(f"  total                 A-C = {(wide['A'] - wide['C']).mean():.2f} pts")


if __name__ == "__main__":
    main()
