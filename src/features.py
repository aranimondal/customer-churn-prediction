"""Feature engineering as a stateless sklearn transformer.

Implementing this as a transformer (rather than a notebook-only helper) means
the notebook and the API run byte-identical feature logic: both simply call the
persisted pipeline. There is no second copy of this code to drift.

The derived feature is computed from a single customer's own row, so it is
stateless and cannot leak information across the train/test boundary.

Two earlier engineered features were removed after measuring their importance
on the fitted trees -- see the class docstring below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds one domain-motivated feature to the raw Telco columns.

    ``monthly_charge_delta`` -- ``MonthlyCharges - TotalCharges / tenure``,
        i.e. the gap between what the customer pays today and their historical
        average monthly spend. A large positive value means the bill has crept
        up relative to what they are used to paying, which is a classic trigger
        for shopping around. Guarded against tenure == 0, where no billing
        history exists and the delta is defined as 0.

    Removed features
    ----------------
    ``num_addon_services`` (count of the six optional add-ons) and
    ``tenure_bucket`` (tenure discretised into 0-12 / 13-24 / 25-48 / 49+)
    were both dropped. Measured on the fitted model, each scored an importance
    of exactly 0.0 in all three pruned configurations -- the tree never chose to
    split on either one. The hypotheses behind them were reasonable (add-ons
    raise switching cost; churn risk is non-linear in tenure) but the tree
    already captures both effects directly: it splits on the raw add-on columns
    and finds its own tenure thresholds, which is strictly more flexible than a
    fixed bucketing. Keeping them would have added encoded columns that carry no
    signal, so no information was lost by removing them.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureBuilder":
        # Stateless: nothing is learned from the data.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        tenure = out["tenure"].to_numpy(dtype=float)
        avg_monthly = np.divide(
            out["TotalCharges"].to_numpy(dtype=float),
            tenure,
            out=np.zeros_like(tenure),
            where=tenure > 0,
        )
        delta = out["MonthlyCharges"].to_numpy(dtype=float) - avg_monthly
        # tenure == 0 => no history to compare against; neutral value.
        out["monthly_charge_delta"] = np.where(tenure > 0, delta, 0.0)

        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(
            config.MODEL_INPUT_COLUMNS
        )
        return np.asarray(
            base
            + config.ENGINEERED_NUMERIC_FEATURES
            + config.ENGINEERED_CATEGORICAL_FEATURES
        )
