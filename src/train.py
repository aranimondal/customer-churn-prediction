"""Training entry point.

Run as a module from the project root:

    python -m src.train

Compares four materially different Decision Tree configurations by 5-fold
stratified cross-validated recall on the training split, refits the selected
configuration on the full training split, evaluates once on the held-out test
split, and persists the fitted pipeline plus a metrics report.

The test split is touched exactly once, at the end. Model selection uses
cross-validation *inside* the training data only.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config, data
from .pipeline import build_pipeline, feature_names
from .persistence import save_pipeline

# --------------------------------------------------------------------------- #
# Candidate configurations
# --------------------------------------------------------------------------- #
# Chosen to differ in kind, not just in degree:
#   baseline_unpruned -> establishes the overfitting reference point
#   pruned_gini       -> regularised depth/leaf constraints
#   pruned_entropy    -> same capacity, different split criterion
#   balanced_pruned   -> pruned + class_weight to counter the ~26.5% churn rate
CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline_unpruned": {
        "criterion": "gini",
    },
    "pruned_gini": {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 40,
        "min_samples_leaf": 20,
    },
    "pruned_entropy": {
        "criterion": "entropy",
        "max_depth": 6,
        "min_samples_split": 40,
        "min_samples_leaf": 20,
    },
    "balanced_pruned": {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 40,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
    },
}

# Recall is the selection metric: a missed churner is an irrecoverable lost
# subscriber, whereas a false positive costs one retention contact. See README
# and the notebook for the full cost argument.
SELECTION_METRIC = "recall"


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute the metrics required by the assignment, plus ROC-AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def main() -> None:
    df = data.clean(data.load_raw())
    X, y = data.split_features_target(df)

    # Stratified because churn is imbalanced (~26.5% positive). Without
    # stratification the 30% test split could drift several points away from the
    # population churn rate, making the recall estimate unreliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    comparison = {}
    for name, params in CANDIDATES.items():
        pipe = build_pipeline(**params)
        scores = {
            metric: cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric)
            for metric in ("accuracy", "precision", "recall", "f1")
        }
        comparison[name] = {
            f"cv_{metric}": float(vals.mean()) for metric, vals in scores.items()
        }
        comparison[name]["params"] = params
        print(
            f"{name:<20} "
            + "  ".join(
                f"{m}={comparison[name][f'cv_{m}']:.3f}"
                for m in ("accuracy", "precision", "recall", "f1")
            )
        )

    best_name = max(
        comparison, key=lambda n: comparison[n][f"cv_{SELECTION_METRIC}"]
    )
    print(f"\nSelected configuration: {best_name} (by CV {SELECTION_METRIC})")

    final = build_pipeline(**CANDIDATES[best_name])
    final.fit(X_train, y_train)

    y_pred = final.predict(X_test)
    y_proba = final.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, y_pred, y_proba)

    print("\nHeld-out test performance")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10} {test_metrics[key]:.4f}")
    print(f"  confusion  {test_metrics['confusion_matrix']}")

    importances = (
        pd.Series(
            final.named_steps["model"].feature_importances_,
            index=feature_names(final),
        )
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nTop features")
    print(importances.to_string())

    save_pipeline(final)

    config.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(
        json.dumps(
            {
                "selected_configuration": best_name,
                "selection_metric": SELECTION_METRIC,
                "configuration_comparison": comparison,
                "test_metrics": test_metrics,
                "n_rows": int(len(df)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "train_churn_rate": float(y_train.mean()),
                "test_churn_rate": float(y_test.mean()),
                "top_features": importances.round(4).to_dict(),
            },
            indent=2,
        )
    )
    print(f"\nSaved model -> {config.MODEL_FILE}")
    print(f"Saved metrics -> {config.METRICS_FILE}")


if __name__ == "__main__":
    main()
