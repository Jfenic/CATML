from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile


def profile_dataset(dataset: Dataset) -> DatasetProfile:
    path = Path(dataset.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if dataset.target_column not in df.columns:
        raise ValueError(f"Target column '{dataset.target_column}' not in dataset")

    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        sample = series.dropna().head(3).tolist()
        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=sample,
            )
        )

    return DatasetProfile(
        dataset_id=dataset.id,
        row_count=len(df),
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
