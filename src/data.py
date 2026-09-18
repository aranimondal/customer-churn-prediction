"""Dataset loading and cleaning.

Cleaning here is deliberately limited to operations that are *row-local* and
therefore cannot leak information between the train and test sets: type
coercion, whitespace stripping and duplicate removal. Anything that learns a
statistic from the data (imputation values, category vocabularies) lives inside
the fitted sklearn pipeline in src/pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Telco CSV exactly as supplied, with no coercion applied."""
    return pd.read_csv(path or config.RAW_DATA_FILE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the row-local cleaning steps.

    Two dataset-specific issues are handled:

    1. ``TotalCharges`` ships as an object/string column because 11 rows carry a
       single blank space instead of a number. Every one of those rows has
       ``tenure == 0``, i.e. a customer who signed up but has not yet been
       billed for a full cycle. Their true lifetime spend is therefore 0, not
       "unknown", so we coerce to numeric and fill those rows with 0.0. Median
       imputation would be actively wrong here: it would credit brand-new
       customers with the spend history of an average long-tenured one and blur
       the strongest churn signal in the dataset (short tenure).

    2. Categorical columns are whitespace-stripped so that a stray " Yes" in
       future data does not become an unseen category at inference time.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    # Guard the assumption rather than assuming it silently.
    blank_rows = out["TotalCharges"].isna()
    if blank_rows.any():
        assert (out.loc[blank_rows, "tenure"] == 0).all(), (
            "TotalCharges is blank for a row with tenure > 0; the "
            "fill-with-zero rule no longer holds and must be revisited."
        )
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].str.strip()

    # Full-row duplicates. customerID is unique in the source file so this is a
    # safety net for re-runs on appended data, not a fix for a known defect.
    out = out.drop_duplicates()

    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with the identifier and the target removed from X.

    ``customerID`` is dropped because it is a surrogate key: it carries no
    generalisable signal and a tree would happily memorise it.
    """
    y = (df[config.TARGET] == config.POSITIVE_CLASS_LABEL).astype(int)
    X = df.drop(columns=[config.TARGET, config.ID_COLUMN], errors="ignore")
    return X[config.MODEL_INPUT_COLUMNS], y
