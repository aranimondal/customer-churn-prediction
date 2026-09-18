"""Project-wide schema and path configuration.

This module is the single source of truth for the dataset schema. Both the
training pipeline and the REST API import from here, which guarantees that the
feature contract cannot drift between training and inference.

All paths are derived from this file's location, so the project is portable and
contains no machine-specific absolute paths.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

RAW_DATA_FILE = DATA_DIR / "TelcoCustomerChurn.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "TelcoCustomerChurn - Data Dictionary.csv"
MODEL_FILE = MODEL_DIR / "churn_model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Reproducibility (fixed by the assignment)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "Churn"
ID_COLUMN = "customerID"  # identifier: excluded from every feature set

# Raw numeric columns. SeniorCitizen is stored as 0/1 in the source file and is
# left numeric: a binary integer is already a valid split candidate for a tree,
# so one-hot encoding it would only add a redundant collinear column.
RAW_NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# The six optional add-on services. Used directly as model inputs.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_SERVICES,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Columns produced by ChurnFeatureBuilder (see src/features.py).
#
# `num_addon_services` and `tenure_bucket` were removed after measuring
# feature importance on the fitted trees: both scored exactly 0.0 in all three
# pruned configurations, i.e. the tree never chose to split on them. Neither
# discarded any information, because the six raw add-on columns and the raw
# `tenure` value they were derived from remain in the feature set and the tree
# splits on those directly. `monthly_charge_delta` is retained: it is the one
# engineered feature the model actually uses.
ENGINEERED_NUMERIC_FEATURES = ["monthly_charge_delta"]
ENGINEERED_CATEGORICAL_FEATURES = []

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

# The exact set of raw columns the pipeline expects as input. The API validates
# incoming payloads against this list.
MODEL_INPUT_COLUMNS = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- #
# Allowed categorical values, transcribed from the supplied data dictionary.
# Used for API request validation so that invalid categories are rejected with a
# 422 rather than being silently swallowed by handle_unknown="ignore".
# --------------------------------------------------------------------------- #
NO_INTERNET = "No internet service"

ALLOWED_VALUES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", NO_INTERNET],
    "OnlineBackup": ["Yes", "No", NO_INTERNET],
    "DeviceProtection": ["Yes", "No", NO_INTERNET],
    "TechSupport": ["Yes", "No", NO_INTERNET],
    "StreamingTV": ["Yes", "No", NO_INTERNET],
    "StreamingMovies": ["Yes", "No", NO_INTERNET],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

POSITIVE_CLASS_LABEL = "Yes"
NEGATIVE_CLASS_LABEL = "No"
