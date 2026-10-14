"""Row ↔ domain mappers for SQLite experiment repository."""
from __future__ import annotations

import json
import sqlite3

from automl.domain.experiments.trial import Experiment, ExperimentStatus, TrialResult
from automl.domain.features.feature import Feature, FeatureStatus


def _row_to_feature(row: sqlite3.Row) -> Feature:
    semantic_type = "unknown"
    if "semantic_type" in row.keys() and row["semantic_type"]:
        semantic_type = row["semantic_type"]
    return Feature(
        id=row["id"],
        dataset_id=row["dataset_id"],
        name=row["name"],
        physical_dtype=row["physical_dtype"] or "unknown",
        status=FeatureStatus(row["status"]),
        user_priority=float(row["user_priority"] or 0),
        semantic_type=semantic_type,
    )


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    group_column = None
    try:
        group_column = row["group_column"]
    except (IndexError, KeyError):
        pass

    column_transforms = []
    try:
        if "column_transforms_json" in row.keys() and row["column_transforms_json"]:
            column_transforms = json.loads(row["column_transforms_json"])
    except (IndexError, KeyError, TypeError):
        pass

    derived_features = []
    try:
        if "derived_features_json" in row.keys() and row["derived_features_json"]:
            derived_features = json.loads(row["derived_features_json"])
    except (IndexError, KeyError, TypeError):
        pass

    recipe_id = None
    try:
        if "recipe_id" in row.keys():
            recipe_id = row["recipe_id"]
    except (IndexError, KeyError, TypeError):
        pass

    return Experiment(
        id=row["id"],
        run_id=row["run_id"],
        name=row["name"],
        hypothesis=row["hypothesis"] or "",
        feature_names=json.loads(row["feature_names_json"]),
        model_ids=json.loads(row["model_ids_json"]),
        metric=row["metric"],
        validation_strategy=row["validation_strategy"],
        status=ExperimentStatus(row["status"]),
        created_by=row["created_by"] or "user",
        priority=row["priority"] or "normal",
        feature_set_id=row["feature_set_id"],
        group_column=group_column,
        column_transforms=column_transforms,
        derived_features=derived_features,
        recipe_id=recipe_id,
    )


def _row_to_result(row: sqlite3.Row) -> TrialResult:
    return TrialResult(
        trial_id=row["trial_id"],
        experiment_id=row["experiment_id"],
        model_id=row["model_id"],
        primary_metric=row["primary_metric"],
        primary_score=row["primary_score"],
        secondary_metrics=json.loads(row["secondary_metrics_json"] or "{}"),
        training_time_seconds=row["training_time_seconds"] or 0.0,
        failure_reason=row["failure_reason"],
        artifacts=json.loads(row["artifacts_json"] or "{}"),
    )
