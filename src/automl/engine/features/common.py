from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    target_column: str | None = None,
    task_type: str = "binary_classification",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Extracts and cleans numeric and categorical features into a clean DataFrame

    aligned 1-to-1 with feature_names, suitable for statistical and ML selectors.

    Invalid target rows are dropped (never imputed as 0.0 or encoded as the
    string class ``\"nan\"``).
    """
    valid_names = [col for col in feature_names if col in df.columns]
    if not valid_names:
        raise ValueError("None of the specified feature names exist in the DataFrame.")

    X = df[valid_names].copy()
    y: np.ndarray | None = None

    if target_column and target_column in df.columns:
        target_series = df[target_column]
        if task_type in ("binary_classification", "multiclass_classification"):
            # Drop missing labels before any string cast (avoids artificial "nan" class).
            valid_mask = target_series.notna()
            if isinstance(target_series.dtype, pd.StringDtype) or target_series.dtype == object:
                as_str = target_series.astype("string")
                valid_mask = valid_mask & as_str.notna() & ~as_str.str.strip().str.lower().isin(
                    {"", "nan", "none", "null", "<na>"}
                )
            if not bool(valid_mask.any()):
                raise ValueError(
                    f"Target column '{target_column}' has no valid labels after dropping missing values."
                )
            X = X.loc[valid_mask].copy()
            y_raw = target_series.loc[valid_mask]
            y = LabelEncoder().fit_transform(y_raw.astype(str))
        elif task_type == "regression":
            y_num = pd.to_numeric(target_series, errors="coerce")
            valid_mask = y_num.notna()
            if not bool(valid_mask.any()):
                raise ValueError(
                    f"Target column '{target_column}' has no valid numeric values "
                    "(missing/invalid targets are not replaced with 0.0)."
                )
            X = X.loc[valid_mask].copy()
            y = y_num.loc[valid_mask].to_numpy(dtype=float)
        else:
            y = None

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

    return X.reset_index(drop=True), y
