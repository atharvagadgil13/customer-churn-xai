"""
Data loading, cleaning, and scikit-learn ColumnTransformer pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    DATA_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, ID_COL,
)


def load_and_clean(path=DATA_PATH) -> pd.DataFrame:
    """Load the Telco-Churn CSV and apply essential cleaning."""
    df = pd.read_csv(path)

    # TotalCharges has blank strings for tenure==0 customers → coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target: use is_numeric_dtype so it works even when pandas stores
    # strings as StringDtype instead of object.
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a *fitted* ColumnTransformer."""
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names
