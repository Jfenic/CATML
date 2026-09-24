from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    target_column: str | None = None,
    task_type: str = "binary_classification",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Extracts and cleans numeric and categorical features into a clean DataFrame

    aligned 1-to-1 with feature_names, suitable for statistical and ML selectors.
    """
    valid_names = [col for col in feature_names if col in df.columns]
    if not valid_names:
        raise ValueError("None of the specified feature names exist in the DataFrame.")

    X = df[valid_names].copy()

    # Identify numeric and categorical columns
    numeric_cols = [c for c in valid_names if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in valid_names if c not in numeric_cols]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("__missing__")
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

    y: np.ndarray | None = None
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        if task_type in ("binary_classification", "multiclass_classification"):
            # Ensure categorical/string targets are mapped to dense integers
            y = LabelEncoder().fit_transform(target_series.astype(str))
        elif task_type == "regression":
            y = pd.to_numeric(target_series, errors="coerce").fillna(0.0).to_numpy()
        else:
            y = None

    return X, y
