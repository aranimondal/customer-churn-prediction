"""Bonus model comparison kept separate from the required Decision Tree path."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config
from .pipeline import build_preprocessor
from .features import ChurnFeatureBuilder


def build_logistic_pipeline() -> Pipeline:
    """Build a leakage-safe Logistic Regression baseline using the same feature contract."""
    return Pipeline([
        ("features", ChurnFeatureBuilder()),
        ("preprocess", build_preprocessor()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE)),
    ])


def compare_bonus_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compare Logistic Regression with the selected Decision Tree on the holdout.

    This comparison is descriptive/bonus-only; the assignment's Decision Tree
    remains the persisted production model.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    logit = build_logistic_pipeline()
    cv_scores = cross_validate(logit, X_train, y_train, cv=cv, scoring=("accuracy", "precision", "recall", "f1", "roc_auc"))
    logit.fit(X_train, y_train)
    proba = logit.predict_proba(X_test)[:, 1]
    pred = logit.predict(X_test)
    return {
        "logistic_regression": {
            "cv": {f"{k}_mean": float(v.mean()) for k, v in ((m, cv_scores[f"test_{m}"]) for m in ("accuracy", "precision", "recall", "f1", "roc_auc"))},
            "test": {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            },
        }
    }
