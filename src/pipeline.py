"""The end-to-end model pipeline: features -> encoding -> Decision Tree.

Everything that learns from data (imputer statistics, one-hot vocabularies, the
tree itself) sits inside this single Pipeline object. It is fitted on the
training split only and then persisted, so test-set evaluation and live API
inference both go through exactly the transformations learned in training.

Note there is no scaler: a Decision Tree splits on thresholds and is invariant
to monotone rescaling, so standardising would add a dependency and buy nothing.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import ChurnFeatureBuilder


def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at inference time.

    ``handle_unknown="ignore"`` keeps the API from crashing on an unexpected
    category. The request schema still rejects unknown values with a 422, so
    this is defence in depth rather than the primary guard.
    """
    try:  # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing applied after feature engineering."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(**tree_params) -> Pipeline:
    """Assemble the full pipeline with the given Decision Tree hyperparameters."""
    params = {"random_state": config.RANDOM_STATE, **tree_params}
    return Pipeline(
        [
            ("features", ChurnFeatureBuilder()),
            ("preprocess", build_preprocessor()),
            ("model", DecisionTreeClassifier(**params)),
        ]
    )


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Post-encoding feature names, aligned with ``model.feature_importances_``."""
    return list(
        fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    )
