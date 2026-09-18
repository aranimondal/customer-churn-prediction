"""Model artifact persistence.

The whole Pipeline (feature builder + encoders + tree) is pickled as one object,
so loading it restores the exact transformations fitted during training. The API
never reimplements preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from . import config


def save_pipeline(pipeline: Pipeline, path: str | Path | None = None) -> Path:
    target = Path(path or config.MODEL_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    return target


def load_pipeline(path: str | Path | None = None) -> Pipeline:
    target = Path(path or config.MODEL_FILE)
    if not target.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {target}. "
            "Train it first with: python -m src.train"
        )
    return joblib.load(target)
