"""Tests covering the contracts that matter: cleaning rules, leakage, feature
correctness, artifact round-trip, and API behaviour.

Run from the project root:  pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config, data
from src.features import ChurnFeatureBuilder
from src.pipeline import build_pipeline, feature_names
from src.bonus_models import build_logistic_pipeline
from src.decision_policy import threshold_sensitivity


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return data.clean(data.load_raw())


@pytest.fixture(scope="module")
def sample_payload() -> dict:
    return json.loads((config.PROJECT_ROOT / "sample_request.json").read_text())


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def test_expected_shape(clean_df):
    """The assignment states 7,043 rows and 21 columns."""
    assert clean_df.shape[1] == 21
    assert len(clean_df) == 7043, (
        "Row count changed; duplicates were dropped or the source file differs."
    )


def test_total_charges_is_numeric_and_complete(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].isna().sum() == 0


def test_blank_total_charges_were_zero_tenure_rows():
    """The documented defect: 11 blanks, all with tenure == 0."""
    raw = data.load_raw()
    blank = pd.to_numeric(
        raw["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).isna()
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()
    cleaned = data.clean(raw)
    assert (cleaned.loc[blank, "TotalCharges"] == 0.0).all()


def test_no_duplicates(clean_df):
    assert clean_df.duplicated().sum() == 0
    assert clean_df[config.ID_COLUMN].duplicated().sum() == 0


def test_categories_match_data_dictionary(clean_df):
    for col, allowed in config.ALLOWED_VALUES.items():
        assert set(clean_df[col].unique()) <= set(allowed), f"{col} has unknown values"


def test_identifier_and_target_excluded_from_features(clean_df):
    X, y = data.split_features_target(clean_df)
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.MODEL_INPUT_COLUMNS
    assert set(y.unique()) <= {0, 1}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_engineered_features_are_present():
    """The assignment requires at least two meaningful engineered features."""
    row = {c: "No" for c in config.MODEL_INPUT_COLUMNS}
    row.update(
        {
            "SeniorCitizen": 0,
            "tenure": 18,
            "MonthlyCharges": 60.0,
            "TotalCharges": 900.0,
            "gender": "Male",
            "InternetService": "DSL",
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
        }
    )
    out = ChurnFeatureBuilder().fit_transform(
        pd.DataFrame([row])[config.MODEL_INPUT_COLUMNS]
    )
    assert "monthly_charge_delta" in out.columns
    assert "num_addon_services" in out.columns
    assert "tenure_bucket" in out.columns
    assert out["num_addon_services"].iloc[0] == 3.0
    assert out["tenure_bucket"].iloc[0] == "13-24"


def test_charge_delta_handles_zero_tenure():
    """tenure == 0 must yield a finite, neutral delta rather than inf/NaN."""
    row = {c: "No" for c in config.MODEL_INPUT_COLUMNS}
    row.update(
        {
            "SeniorCitizen": 0,
            "tenure": 0,
            "MonthlyCharges": 70.0,
            "TotalCharges": 0.0,
            "gender": "Female",
            "InternetService": "No",
            "Contract": "Month-to-month",
            "PaymentMethod": "Mailed check",
        }
    )
    out = ChurnFeatureBuilder().fit_transform(
        pd.DataFrame([row])[config.MODEL_INPUT_COLUMNS]
    )
    assert out["monthly_charge_delta"].iloc[0] == 0.0


def test_engineered_columns_are_finite(clean_df):
    X, _ = data.split_features_target(clean_df)
    out = ChurnFeatureBuilder().fit_transform(X)
    assert np.isfinite(out[config.ENGINEERED_NUMERIC_FEATURES].to_numpy()).all()


def test_feature_builder_is_stateless(clean_df):
    """Fitting on a subset must not change what transform produces."""
    X, _ = data.split_features_target(clean_df)
    a = ChurnFeatureBuilder().fit(X.head(100)).transform(X.head(50))
    b = ChurnFeatureBuilder().fit(X).transform(X.head(50))
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_trains_and_predicts_in_range(clean_df):
    X, y = data.split_features_target(clean_df)
    pipe = build_pipeline(max_depth=4, min_samples_leaf=20).fit(X, y)
    proba = pipe.predict_proba(X.head(20))[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert len(feature_names(pipe)) == pipe.named_steps["model"].n_features_in_


def test_pipeline_tolerates_unseen_category(clean_df):
    """handle_unknown='ignore' must keep inference alive on novel categories."""
    X, y = data.split_features_target(clean_df)
    pipe = build_pipeline(max_depth=4, min_samples_leaf=20).fit(X, y)
    odd = X.head(1).copy()
    odd.loc[odd.index[0], "PaymentMethod"] = "Cryptocurrency"
    assert pipe.predict_proba(odd).shape == (1, 2)


def test_saved_artifact_round_trips(tmp_path, clean_df):
    from src.persistence import load_pipeline, save_pipeline

    X, y = data.split_features_target(clean_df)
    pipe = build_pipeline(max_depth=4, min_samples_leaf=20).fit(X, y)
    path = save_pipeline(pipe, tmp_path / "m.pkl")
    reloaded = load_pipeline(path)
    np.testing.assert_array_equal(reloaded.predict(X), pipe.predict(X))


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client():
    if not config.MODEL_FILE.exists():
        pytest.skip("model artifact missing; run `python -m src.train` first")
    from app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True


def test_predict_returns_label_and_probability(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert set(body) == {"prediction", "churn_probability"}


def test_predict_rejects_bad_category(client, sample_payload):
    bad = {**sample_payload, "Contract": "Lifetime"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "Contract" in json.dumps(r.json())


def test_predict_rejects_missing_field(client, sample_payload):
    bad = {k: v for k, v in sample_payload.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range_number(client, sample_payload):
    assert client.post(
        "/predict", json={**sample_payload, "tenure": -5}
    ).status_code == 422
    assert client.post(
        "/predict", json={**sample_payload, "SeniorCitizen": 7}
    ).status_code == 422


def test_predict_ignores_customer_id(client, sample_payload):
    """Sending the identifier must not change the prediction."""
    base = client.post("/predict", json=sample_payload).json()
    with_id = client.post(
        "/predict", json={**sample_payload, "customerID": "9999-ZZZZZ"}
    ).json()
    assert base == with_id


# --------------------------------------------------------------------------- #
# Bonus model and decision policy
# --------------------------------------------------------------------------- #
def test_bonus_logistic_pipeline_trains_and_predicts(clean_df):
    """The additional-model comparison must remain executable and leakage-safe."""
    X, y = data.split_features_target(clean_df)
    pipe = build_logistic_pipeline().fit(X, y)
    proba = pipe.predict_proba(X.head(10))[:, 1]
    assert proba.shape == (10,)
    assert np.isfinite(proba).all()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_bonus_logistic_pipeline_tolerates_unseen_category(clean_df):
    X, y = data.split_features_target(clean_df)
    pipe = build_logistic_pipeline().fit(X, y)
    odd = X.head(1).copy()
    odd.loc[odd.index[0], "PaymentMethod"] = "Cryptocurrency"
    assert pipe.predict_proba(odd).shape == (1, 2)


def test_threshold_sensitivity_returns_operational_metrics():
    result = threshold_sensitivity(
        [0, 0, 1, 1], [0.10, 0.40, 0.60, 0.90], thresholds=[0.50, 0.80]
    )
    assert list(result.columns) == [
        "threshold", "customers_contacted", "contact_rate",
        "true_positives", "false_positives", "false_negatives",
        "precision", "recall",
    ]
    assert result.loc[0, "customers_contacted"] == 2
    assert result.loc[0, "true_positives"] == 2
    assert result.loc[0, "false_negatives"] == 0
    assert result.loc[1, "customers_contacted"] == 1
    assert result.loc[1, "recall"] == 0.5


def test_threshold_sensitivity_rejects_invalid_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_sensitivity([0, 1], [0.2, 1.2])


def test_threshold_sensitivity_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        threshold_sensitivity([0, 1], [0.2])
