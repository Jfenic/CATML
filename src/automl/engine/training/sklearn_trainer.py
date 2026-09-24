from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.domain.experiments.trial import TrialResult, TrialStatus
from automl.domain.ports import TrialExecution, TrainerPort
from automl.engine.profiling.dataset_profiler import load_dataframe
from automl.plugins.models.sklearn_models import build_sklearn_model


class SklearnTrainer(TrainerPort):
    def __init__(self, plugin_registry: Any = None) -> None:
        self.plugin_registry = plugin_registry

    def run(self, execution: TrialExecution) -> TrialResult:
        import time

        trial = execution.trial
        trial.status = TrialStatus.RUNNING
        started = time.perf_counter()

        try:
            df = load_dataframe(execution.dataset_path)
            X = df[execution.feature_names]

            if execution.task_type == "clustering":
                return self._run_clustering(execution, X, started)

            y = df[execution.target_column]

            if self.plugin_registry and self.plugin_registry.has(execution.trial.model_id):
                model_plugin = self.plugin_registry.get_model_plugin(execution.trial.model_id)
                if model_plugin:
                    self.plugin_registry.validate_plugin_for_task(
                        execution.trial.model_id,
                        execution.task_type,
                    )
                    model = model_plugin.build_estimator(
                        parameters=execution.trial.parameters,
                        task_type=execution.task_type,
                    )
                else:
                    model = build_sklearn_model(
                        execution.trial.model_id,
                        execution.task_type,
                        parameters=execution.trial.parameters,
                    )
            else:
                model = build_sklearn_model(
                    execution.trial.model_id,
                    execution.task_type,
                    parameters=execution.trial.parameters,
                )
            pipeline = _build_pipeline(X, model)

            metric_name = execution.metric
            if execution.validation_strategy == "cross_validation":
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=execution.cv_folds,
                    scoring=_sklearn_scoring(metric_name, execution.task_type),
                    n_jobs=1,
                )
                primary_score = float(np.mean(scores))
                secondary = {"cv_std": float(np.std(scores)), "cv_scores": scores.tolist()}
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=execution.test_size,
                    random_state=execution.random_seed,
                    stratify=y if execution.task_type != "regression" else None,
                )
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                if self.plugin_registry and self.plugin_registry.has(metric_name):
                    metric_plugin = self.plugin_registry.get_metric_plugin(metric_name)
                    if metric_plugin:
                        prob = None
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                prob = pipeline.predict_proba(X_test)
                            except Exception:
                                prob = None
                        primary_score = float(metric_plugin.compute(y_test, predictions, prob))
                        secondary = {metric_name: primary_score}
                    else:
                        secondary = _compute_metrics(
                            y_test,
                            predictions,
                            pipeline,
                            X_test,
                            execution.task_type,
                        )
                        primary_score = secondary.get(metric_name, secondary.get("accuracy", 0.0))
                else:
                    secondary = _compute_metrics(
                        y_test,
                        predictions,
                        pipeline,
                        X_test,
                        execution.task_type,
                    )
                    primary_score = secondary.get(metric_name, secondary.get("accuracy", 0.0))

            elapsed = time.perf_counter() - started
            trial.status = TrialStatus.COMPLETED
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=metric_name,
                primary_score=float(primary_score),
                secondary_metrics={k: float(v) for k, v in secondary.items() if isinstance(v, (int, float))},
                training_time_seconds=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 — surface failure in TrialResult
            elapsed = time.perf_counter() - started
            trial.status = TrialStatus.FAILED
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=execution.metric,
                primary_score=0.0,
                training_time_seconds=elapsed,
                failure_reason=str(exc),
            )

    def _run_clustering(
        self,
        execution: TrialExecution,
        X: pd.DataFrame,
        started: float,
    ) -> TrialResult:
        import time

        trial = execution.trial
        numeric_X = X.select_dtypes(include=["number"])
        if numeric_X.empty:
            raise ValueError("Clustering requires at least one numeric feature.")

        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_scaled = preprocessor.fit_transform(numeric_X)
        model = build_sklearn_model(
            execution.trial.model_id,
            execution.task_type,
            parameters=execution.trial.parameters,
        )

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_scaled)

        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        secondary: dict[str, float] = {"n_clusters": float(n_clusters)}
        metric_name = execution.metric

        if n_clusters >= 2 and len(labels) > n_clusters:
            secondary["silhouette"] = float(silhouette_score(X_scaled, labels))
            secondary["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
        else:
            secondary["silhouette"] = 0.0
            secondary["calinski_harabasz"] = 0.0

        primary_score = secondary.get(metric_name, secondary.get("silhouette", 0.0))
        elapsed = time.perf_counter() - started
        trial.status = TrialStatus.COMPLETED
        return TrialResult(
            trial_id=trial.id,
            experiment_id=execution.experiment.id,
            model_id=trial.model_id,
            primary_metric=metric_name,
            primary_score=float(primary_score),
            secondary_metrics=secondary,
            training_time_seconds=elapsed,
        )


def _build_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _sklearn_scoring(metric_name: str, task_type: str) -> str:
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    if metric_name in mapping:
        return mapping[metric_name]
    return "accuracy" if task_type != "regression" else "r2"


def _compute_metrics(
    y_true: Any,
    predictions: Any,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    task_type: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if task_type == "regression":
        metrics["r2"] = float(r2_score(y_true, predictions))
        metrics["mae"] = float(mean_absolute_error(y_true, predictions))
        metrics["rmse"] = float(mean_squared_error(y_true, predictions, squared=False))
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["f1"] = float(f1_score(y_true, predictions, average="weighted"))
    if task_type == "binary_classification":
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
    return metrics
