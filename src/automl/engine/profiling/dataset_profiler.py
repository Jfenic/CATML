from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile


def detect_column_cardinality_and_role(
    name: str,
    series: pd.Series,
    row_count: int,
    target_column: str | None = None,
) -> tuple[float, bool, bool]:
    """
    Evaluates a dataset column to calculate its cardinality ratio and determine
    if it acts as a non-predictive identifier or high-cardinality feature.

    Returns:
        tuple[float, bool, bool]: (cardinality_ratio, is_identifier, is_high_cardinality)
    """
    if target_column and name == target_column:
        return 0.0, False, False

    unique_count = int(series.nunique(dropna=True))
    cardinality_ratio = float(unique_count / row_count) if row_count > 0 else 0.0

    clean_name = name.strip().lower()
    dtype_str = str(series.dtype).lower()
    is_object_or_string = (
        series.dtype.kind in {"O", "S", "U"}
        or "str" in dtype_str
        or "object" in dtype_str
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    # 1. High cardinality detection (primarily for string/text/categorical features)
    is_high_cardinality = False
    if is_object_or_string and row_count >= 50:
        if cardinality_ratio > 0.70 or (unique_count > 100 and cardinality_ratio > 0.50):
            is_high_cardinality = True

    # 2. Identifier detection:
    # A) Name-based ID heuristic
    name_matches_id = (
        clean_name in {"id", "guid", "uuid", "hash", "pk", "key"}
        or clean_name.endswith(("_id", ".id", "-id", "_guid", "_uuid", "_hash", "_key", "_pk"))
        or clean_name.startswith(("id_", "guid_", "uuid_", "pk_"))
        or name.endswith(("Id", "ID", "Guid", "GUID", "Uuid", "UUID"))
    )

    is_identifier = False
    if name_matches_id:
        # Check that it actually has identifier-like cardinality (avoid small enums like status_id with 2 values)
        if row_count <= 20:
            is_identifier = unique_count >= max(2, int(row_count * 0.7))
        else:
            is_identifier = unique_count >= 20 or cardinality_ratio >= 0.25

    # B) Content-based identifier heuristic (no ID name required)
    if not is_identifier and row_count >= 50:
        # High cardinality text/hash columns with near-unique values
        if is_object_or_string and cardinality_ratio > 0.70:
            is_identifier = True
        # Monotonically increasing sequential index integers (e.g. 0, 1, 2, ... N-1 or row counter)
        elif series.dtype.kind in {"i", "u"} and cardinality_ratio == 1.0:
            clean_series = series.dropna()
            if clean_series.is_monotonic_increasing:
                diffs = clean_series.diff().iloc[1:]
                if len(diffs) > 0 and (diffs == 1).all():
                    is_identifier = True

    return cardinality_ratio, is_identifier, is_high_cardinality


def profile_dataset(dataset: Dataset) -> DatasetProfile:
    path = Path(dataset.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if dataset.target_column not in df.columns:
        raise ValueError(f"Target column '{dataset.target_column}' not in dataset")

    row_count = len(df)
    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        sample = series.dropna().head(3).tolist()
        unique_count = int(series.nunique(dropna=True))
        card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
            name=name,
            series=series,
            row_count=row_count,
            target_column=dataset.target_column,
        )
        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                null_count=int(series.isna().sum()),
                unique_count=unique_count,
                sample_values=sample,
                cardinality_ratio=card_ratio,
                is_identifier=is_id,
                is_high_cardinality=is_high_card,
            )
        )

    return DatasetProfile(
        dataset_id=dataset.id,
        row_count=row_count,
        column_count=len(df.columns),
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        columns=columns,
    )


def infer_task_type(target_series: pd.Series) -> str:
    from automl.engine.planning.task_planner import infer_task_type as _infer

    return _infer(target_series).value


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
