"""
Centralised configuration: file paths, column groups, and constants.
"""
from pathlib import Path

# ── Project root (two levels up from this file) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ──
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "xgb_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SCORED_CSV_PATH = ARTIFACTS_DIR / "scored_customers.csv"

# ── Target ──
TARGET = "Churn"
ID_COL = "customerID"

# ── Column groups (after dropping customerID) ──
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# ── Modelling constants ──
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING_METRIC = "roc_auc"
