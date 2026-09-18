"""Business-facing threshold analysis for churn campaign capacity.

This module is deliberately separate from the trained pipeline and API. It does
not change the deployed model or its 0.5 prediction threshold. It provides a
reproducible way to inspect how a retention team's contact capacity changes as
the probability cut-off changes.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def threshold_sensitivity(
    y_true: Sequence[int],
    churn_probability: Sequence[float],
    thresholds: Sequence[float] = (0.30, 0.40, 0.50, 0.60, 0.70),
) -> pd.DataFrame:
    """Return an operational precision/recall table without refitting a model.

    A row represents the customers that would be contacted at a given
    probability threshold. This is an analysis aid only; no threshold is
    selected automatically because the appropriate cut-off depends on the
    retention team's available capacity and the relative cost of errors.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(churn_probability, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and churn_probability must be non-empty 1-D arrays of equal length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("churn_probability values must be finite and between 0 and 1.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels.")

    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("thresholds must be between 0 and 1.")

        predicted = (p >= threshold).astype(int)
        contacted = int(predicted.sum())
        rows.append(
            {
                "threshold": threshold,
                "customers_contacted": contacted,
                "contact_rate": contacted / len(y),
                "true_positives": int(((predicted == 1) & (y == 1)).sum()),
                "false_positives": int(((predicted == 1) & (y == 0)).sum()),
                "false_negatives": int(((predicted == 0) & (y == 1)).sum()),
                "precision": float(precision_score(y, predicted, zero_division=0)),
                "recall": float(recall_score(y, predicted, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)
