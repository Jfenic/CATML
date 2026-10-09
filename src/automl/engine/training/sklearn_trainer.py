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
from automl.domain.runs.states import RunStatus
from automl.domain.ports import TrialExecution, TrainerPort
from automl.engine.profiling.dataset_profiler import load_dataframe
from automl.plugins.models.sklearn_models import build_sklearn_model


class SklearnTrainer(TrainerPort):
    def __init__(self, plugin_registry: Any = None) -> None:
        self.plugin_registry = plugin_registry

    def run(self, execution: TrialExecution) -> TrialResult:
        import time

        trial = execution.trial
        if execution.run and getattr(execution.run, "status", None) == RunStatus.CANCELLED:
            trial.status = TrialStatus.FAILED
            trial.error_message = "Execution cancelled."
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=execution.metric,
                primary_score=0.0,
                secondary_metrics={},
                training_time_seconds=0.0,
                failure_reason="Execution cancelled.",
            )

        trial.status = TrialStatus.RUNNING
        started = time.perf_counter()

        try:
            df = load_dataframe(execution.dataset_path)
            X = df[execution.feature_names]

            if execution.task_type == "clustering":
                return self._run_clustering(execution, X, started)

            y = df[execution.target_column]
            from automl.engine.training.target_adapter import TargetAdapter

            adapter = TargetAdapter()
            y_adapted = adapter.fit_transform(y, execution.task_type)

            model_params = {
                k: v for k, v in (execution.trial.parameters or {}).items()
                if k not in ("text_columns", "image_columns", "image_model", "time_budget")
            }

            if self.plugin_registry and self.plugin_registry.has(execution.trial.model_id):
                model_plugin = self.plugin_registry.get_model_plugin(execution.trial.model_id)
                if model_plugin:
                    self.plugin_registry.validate_plugin_for_task(
                        execution.trial.model_id,
                        execution.task_type,
                    )
                    model = model_plugin.build_estimator(
                        parameters=model_params,
                        task_type=execution.task_type,
                    )
                else:
                    model = build_sklearn_model(
                        execution.trial.model_id,
                        execution.task_type,
                        parameters=model_params,
                    )
            else:
                model = build_sklearn_model(
                    execution.trial.model_id,
                    execution.task_type,
                    parameters=model_params,
                )
            text_cols = execution.trial.parameters.get("text_columns") if execution.trial.parameters else None
            if text_cols is None and execution.run and execution.run.config and execution.run.config.extra:
                text_cols = execution.run.config.extra.get("text_columns")

            image_cols = execution.trial.parameters.get("image_columns") if execution.trial.parameters else None
            if image_cols is None and execution.run and execution.run.config and execution.run.config.extra:
                image_cols = execution.run.config.extra.get("image_columns")

            image_model = execution.trial.parameters.get("image_model") if execution.trial.parameters else None
            if image_model is None and execution.run and execution.run.config and execution.run.config.extra:
                image_model = execution.run.config.extra.get("image_model")

            pipeline = _build_pipeline(
                X,
                model,
                text_columns=text_cols,
                image_columns=image_cols,
                image_model=image_model,
            )

            metric_name = execution.metric
            strategy = (execution.validation_strategy or "holdout").lower()

            if strategy in {
                "cross_validation",
                "kfold",
                "stratified_kfold",
                "time_series",
                "time_series_split",
                "group_kfold",
                "group_cv",
                "grouped_kfold",
            }:
                from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
                n_splits = max(2, execution.cv_folds or 5)
                if len(X) >= 4:
                    if metric_name.lower() == "r2":
                        n_splits = min(n_splits, max(2, len(X) // 2))
                    else:
                        n_splits = min(n_splits, len(X))
                else:
                    n_splits = min(n_splits, max(2, len(X)))

                groups = None
                if strategy in {"group_kfold", "group_cv", "grouped_kfold"}:
                    if not execution.group_column or execution.group_column not in df.columns:
                        raise ValueError(f"group_column '{execution.group_column}' required for {strategy} validation")
                    groups = df[execution.group_column].values
                    n_unique_groups = len(set(groups))
                    n_splits = min(n_splits, n_unique_groups)
                    cv_splitter = GroupKFold(n_splits=n_splits)
                elif strategy in {"time_series", "time_series_split"}:
                    cv_splitter = TimeSeriesSplit(n_splits=n_splits)
                elif strategy == "stratified_kfold" and execution.task_type != "regression":
                    min_class_count = int(pd.Series(y_adapted).value_counts().min()) if len(y_adapted) > 0 else 0
                    if min_class_count < 2:
                        cv_splitter = KFold(n_splits=min(n_splits, max(2, len(X))), shuffle=True, random_state=execution.random_seed)
                    else:
                        cv_splitter = StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=execution.random_seed)
                else:
                    cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=execution.random_seed)

                scoring = _sklearn_scoring(metric_name, execution.task_type, self.plugin_registry)
                scores = cross_val_score(
                    pipeline,
                    X,
                    y_adapted,
                    cv=cv_splitter,
                    groups=groups,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )
                is_neg = (
                    (isinstance(scoring, str) and scoring.startswith("neg_"))
                    or _is_minimizing_metric(metric_name, self.plugin_registry)
                    or getattr(scoring, "_sign", 1) == -1
                )
                if is_neg:
                    scores = -scores

                primary_score = float(np.mean(scores))
                if np.isnan(primary_score):
                    raise ValueError(f"Cross-validation returned NaN scores: {scores}")
                secondary = {
                    "cv_std": float(np.std(scores)),
                    "cv_scores": [float(s) for s in scores],
                    "validation_strategy": strategy,
                }
                if groups is not None:
                    secondary["group_column"] = execution.group_column
                    secondary["n_groups"] = len(set(groups))
                pipeline.fit(X, y_adapted)
            else:
                if execution.group_column and execution.group_column in df.columns:
                    from sklearn.model_selection import GroupShuffleSplit
                    gss = GroupShuffleSplit(n_splits=1, test_size=execution.test_size, random_state=execution.random_seed)
                    train_idx, test_idx = next(gss.split(X, y_adapted, groups=df[execution.group_column].values))
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train = y_adapted.iloc[train_idx] if hasattr(y_adapted, "iloc") else y_adapted[train_idx]
                    y_test = y_adapted.iloc[test_idx] if hasattr(y_adapted, "iloc") else y_adapted[test_idx]
                else:
                    stratify_target = y_adapted if execution.task_type != "regression" else None
                    effective_test_size = execution.test_size
                    if stratify_target is not None:
                        n_samples = len(y_adapted)
                        n_classes = len(np.unique(y_adapted))
                        if n_classes >= 2 and n_samples >= 2 * n_classes:
                            computed_test = int(n_samples * execution.test_size) if execution.test_size < 1.0 else int(execution.test_size)
                            if computed_test < n_classes:
                                effective_test_size = n_classes

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_adapted,
                            test_size=effective_test_size,
                            random_state=execution.random_seed,
                            stratify=stratify_target,
                        )
                    except ValueError:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y_adapted,
                            test_size=execution.test_size,
                            random_state=execution.random_seed,
                            stratify=None,
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
                        if metric_name not in secondary:
                            raise ValueError(f"Metric '{metric_name}' is not supported for task '{execution.task_type}'.")
                        primary_score = secondary[metric_name]
                else:
                    secondary = _compute_metrics(
                        y_test,
                        predictions,
                        pipeline,
                        X_test,
                        execution.task_type,
                    )
                    if metric_name not in secondary:
                        raise ValueError(f"Metric '{metric_name}' is not supported for task '{execution.task_type}'.")
                    primary_score = secondary[metric_name]

                if np.isnan(primary_score):
                    raise ValueError(f"Evaluation returned NaN score for metric '{metric_name}'")

            elapsed = time.perf_counter() - started
            trial.status = TrialStatus.COMPLETED
            artifacts = {}
            if execution.group_column:
                artifacts["group_column"] = execution.group_column
            if "cv_scores" in secondary:
                artifacts["cv_scores"] = secondary["cv_scores"]
            return TrialResult(
                trial_id=trial.id,
                experiment_id=execution.experiment.id,
                model_id=trial.model_id,
                primary_metric=metric_name,
                primary_score=float(primary_score),
                secondary_metrics={k: float(v) for k, v in secondary.items() if isinstance(v, (int, float)) and not np.isnan(v)},
                training_time_seconds=elapsed,
                artifacts=artifacts,
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

    def fit_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        model_id: str,
        task_type: str = "binary_classification",
        parameters: dict[str, Any] | None = None,
    ) -> tuple[Pipeline, Any]:
        model_params = {
            k: v for k, v in (parameters or {}).items()
            if k not in ("text_columns", "image_columns", "image_model", "time_budget")
        }

        if self.plugin_registry and self.plugin_registry.has(model_id):
            model_plugin = self.plugin_registry.get_model_plugin(model_id)
            if model_plugin:
                self.plugin_registry.validate_plugin_for_task(model_id, task_type)
                model = model_plugin.build_estimator(parameters=model_params, task_type=task_type)
            else:
                model = build_sklearn_model(model_id, task_type, parameters=model_params)
        else:
            model = build_sklearn_model(model_id, task_type, parameters=model_params)

        from automl.engine.training.target_adapter import TargetAdapter

        adapter = TargetAdapter()
        y_train_adapted = adapter.fit_transform(y_train, task_type)

        text_cols = parameters.get("text_columns") if parameters else None
        image_cols = parameters.get("image_columns") if parameters else None
        image_model = parameters.get("image_model") if parameters else None
        pipeline = _build_pipeline(
            X_train,
            model,
            text_columns=text_cols,
            image_columns=image_cols,
            image_model=image_model,
        )
        pipeline.fit(X_train, y_train_adapted)
        return pipeline, adapter

    def fit_and_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame,
        model_id: str,
        task_type: str = "binary_classification",
        parameters: dict[str, Any] | None = None,
        predict_proba: bool = False,
    ) -> np.ndarray:
        pipeline, adapter = self.fit_pipeline(
            X_train=X_train,
            y_train=y_train,
            model_id=model_id,
            task_type=task_type,
            parameters=parameters,
        )

        if predict_proba and hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            if task_type == "binary_classification" and probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs

        preds = pipeline.predict(X_test)
        return adapter.inverse_transform(preds)


def _build_pipeline(
    X: pd.DataFrame,
    model: Any,
    text_columns: list[str] | None = None,
    image_columns: list[str] | None = None,
    image_model: str | None = None,
) -> Pipeline:
    from automl.engine.features.text import LightweightTextExtractor, is_text_column
    from automl.plugins.modalities.image_plugin import is_image_column, ImageModalityPlugin
    from automl.engine.vision.image_encoder import ImageEncoderNode

    all_cols = list(X.columns)

    image_cols: list[str] = []
    if image_columns:
        image_cols = [c for c in image_columns if c in all_cols]
    else:
        for c in all_cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                if is_image_column(X[c]):
                    image_cols.append(c)

    text_cols: list[str] = []
    if text_columns:
        text_cols = [c for c in text_columns if c in all_cols and c not in image_cols]
    else:
        for c in all_cols:
            if c not in image_cols and not pd.api.types.is_numeric_dtype(X[c]):
                if is_text_column(X[c]):
                    text_cols.append(c)

    non_transformed_cols = [c for c in all_cols if c not in text_cols and c not in image_cols]
    numeric_cols = [c for c in non_transformed_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in non_transformed_cols if c not in numeric_cols]

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
    if text_cols:
        for col in text_cols:
            transformers.append(
                (
                    f"txt_{col}",
                    LightweightTextExtractor(max_features=50, column_prefix=col),
                    col,
                )
            )
    if image_cols:
        vision_available = ImageModalityPlugin().available()
        chosen_image_model = image_model

        if chosen_image_model is None:
            if vision_available:
                chosen_image_model = "resnet18"
            else:
                raise RuntimeError(
                    f"Dataset contains image columns {image_cols}, but deep learning vision "
                    "dependencies (PyTorch, torchvision, Pillow) are not installed. "
                    "Install vision dependencies via `pip install 'catml[vision]'` or "
                    "specify image_model='deterministic' explicitly for testing/benchmarks."
                )

        for col in image_cols:
            transformers.append(
                (
                    f"img_{col}",
                    ImageEncoderNode(
                        node_id=f"img_{col}",
                        model_name=chosen_image_model,
                        output_dim=None if chosen_image_model not in ("deterministic", "hash") else 128,
                        handle_missing="zero",
                        allow_fallback=False,
                    ),
                    col,
                )
            )

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _is_minimizing_metric(metric_name: str, plugin_registry: Any = None) -> bool:
    if plugin_registry and hasattr(plugin_registry, "has") and plugin_registry.has(metric_name):
        plugin = plugin_registry.get_metric_plugin(metric_name)
        if plugin is not None and hasattr(plugin, "greater_is_better"):
            return not plugin.greater_is_better
    return str(metric_name).lower() in {"mae", "rmse", "mse", "loss", "log_loss"}


def _sklearn_scoring(metric_name: str, task_type: str, plugin_registry: Any = None) -> Any:
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "log_loss": "neg_log_loss",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }
    if metric_name in mapping:
        return mapping[metric_name]

    if plugin_registry and plugin_registry.has(metric_name):
        plugin = plugin_registry.get_metric_plugin(metric_name)
        if plugin:
            from sklearn.metrics import make_scorer

            req_proba = getattr(plugin, "requires_probabilities", False)
            greater = getattr(plugin, "greater_is_better", True)
            try:
                return make_scorer(
                    lambda y_true, y_pred, **kwargs: plugin.compute(y_true, y_pred),
                    greater_is_better=greater,
                    response_method="predict" if not req_proba else "predict_proba",
                )
            except TypeError:
                return make_scorer(
                    lambda y_true, y_pred, **kwargs: plugin.compute(y_true, y_pred),
                    greater_is_better=greater,
                    needs_proba=req_proba,
                )

    raise ValueError(f"Unsupported metric '{metric_name}' for task '{task_type}'.")


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
        mse_val = float(mean_squared_error(y_true, predictions))
        metrics["mse"] = mse_val
        metrics["rmse"] = float(np.sqrt(mse_val))
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["f1"] = float(f1_score(y_true, predictions, average="weighted"))
    if task_type in ("binary_classification", "multiclass_classification"):
        try:
            from sklearn.metrics import balanced_accuracy_score
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, predictions))
        except Exception:
            pass
    if task_type == "binary_classification":
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
        try:
            from sklearn.metrics import log_loss
            proba_all = pipeline.predict_proba(X_test)
            metrics["log_loss"] = float(log_loss(y_true, proba_all))
        except Exception:
            pass
    return metrics
