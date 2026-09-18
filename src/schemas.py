"""Pydantic request/response schemas for the prediction API.

Field constraints and allowed categorical values are derived from the supplied
data dictionary via src/config.py, so validation stays in sync with the schema
the model was trained on.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from . import config

_AV = config.ALLOWED_VALUES


class CustomerFeatures(BaseModel):
    """One customer record, matching the raw dataset columns.

    ``customerID`` is intentionally absent: it is an identifier, not a feature.
    Sending it is harmless (extra keys are ignored) but it is never used.
    """

    model_config = ConfigDict(extra="ignore")

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=1000)
    TotalCharges: float = Field(..., ge=0, le=100000)

    def to_frame(self) -> pd.DataFrame:
        """Single-row DataFrame with columns ordered as the pipeline expects."""
        return pd.DataFrame([self.model_dump()])[config.MODEL_INPUT_COLUMNS]


class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
