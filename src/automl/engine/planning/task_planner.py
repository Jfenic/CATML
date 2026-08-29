from __future__ import annotations

import pandas as pd

from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TaskType


def infer_task_type(target_series: pd.Series) -> TaskType:
    unique = target_series.nunique(dropna=True)
    dtype = str(target_series.dtype)
    is_discrete = dtype in {"object", "bool", "category", "int64", "int32", "int8", "uint8"}

    if is_discrete and unique <= 20:
        if unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def plan_problem(
    dataset: Dataset,
    profile: DatasetProfile,
    task_type: TaskType | None = None,
) -> ProblemDefinition:
    if task_type is not None:
        resolved = task_type
        reasoning = f"Tarea definida explícitamente: {resolved.value}."
        inferred = False
    elif dataset.task_type:
        resolved = TaskType.parse(dataset.task_type)
        reasoning = f"Tarea tomada del dataset registrado: {resolved.value}."
        inferred = False
    else:
        raise ValueError("Task type must be provided for clustering or explicit planning.")

    target = None if resolved == TaskType.CLUSTERING else dataset.target_column

    if resolved == TaskType.CLUSTERING:
        reasoning = (
            "Agrupación no supervisada: los modelos operan sobre features numéricas "
            "sin usar una columna objetivo para entrenar."
        )
        inferred = task_type is None
    elif resolved in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION}:
        if target and target not in {c.name for c in profile.columns}:
            raise ValueError(f"Target column '{target}' not found in dataset profile.")

    return ProblemDefinition.from_task(
        dataset_id=dataset.id,
        task_type=resolved,
        target_column=target,
        inferred=inferred,
        reasoning=reasoning,
    )


def plan_from_dataframe(
    dataset: Dataset,
    df: pd.DataFrame,
    task_type: TaskType | None = None,
) -> ProblemDefinition:
    if task_type == TaskType.CLUSTERING:
        profile = _minimal_profile(dataset, df)
        return plan_problem(dataset, profile, task_type=TaskType.CLUSTERING)

    if task_type is not None:
        profile = _minimal_profile(dataset, df)
        return plan_problem(dataset, profile, task_type=task_type)

    inferred = infer_task_type(df[dataset.target_column])
    profile = _minimal_profile(dataset, df)
    return ProblemDefinition.from_task(
        dataset_id=dataset.id,
        task_type=inferred,
        target_column=dataset.target_column,
        inferred=True,
        reasoning=(
            f"Inferido desde target '{dataset.target_column}': "
            f"{inferred.value} ({df[dataset.target_column].nunique()} valores únicos)."
        ),
    )


def _minimal_profile(dataset: Dataset, df: pd.DataFrame) -> DatasetProfile:
    from automl.domain.datasets.profile import ColumnProfile

    columns = [
        ColumnProfile(
            name=str(col),
            dtype=str(df[col].dtype),
            null_count=int(df[col].isna().sum()),
            unique_count=int(df[col].nunique(dropna=True)),
        )
        for col in df.columns
    ]
    return DatasetProfile(
        dataset_id=dataset.id,
        row_count=len(df),
        column_count=len(df.columns),
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        columns=columns,
    )
