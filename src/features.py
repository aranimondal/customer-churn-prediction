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
    """Add domain features without learning state from the dataset.

    monthly_charge_delta compares today's charge with the customer's
    historical average monthly spend. It is a simple price-change signal.
    num_addon_services captures service breadth, while tenure_bucket
    exposes broad customer lifecycle stages alongside raw tenure.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureBuilder":
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
        out["monthly_charge_delta"] = np.where(tenure > 0, delta, 0.0)

        out["num_addon_services"] = (
            out[config.ADDON_SERVICES].eq("Yes").sum(axis=1).astype(float)
        )

        out["tenure_bucket"] = pd.cut(
            out["tenure"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(str)

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
