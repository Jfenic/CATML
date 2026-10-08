from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.artifacts.model_artifact import ModelArtifact
from automl.domain.tasks.task_type import TaskType
from automl.engine.planning.task_planner import infer_task_type


@dataclass
class AutoMLResult:
    """
    Result container returned by AutoML.fit().

    Provides direct access to the winning ModelArtifact, experiment leaderboard,
    scoring metrics, and high-level prediction methods.
    """

    best_model: ModelArtifact
    best_score: float
    best_model_id: str
    task_type: str
    metric: str
    run_id: str
    _leaderboard_data: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _workspace: Any = field(default=None, repr=False)

    def leaderboard(self) -> pd.DataFrame:
        """
        Return the experiment leaderboard as a formatted pandas DataFrame.
        """
        if not self._leaderboard_data:
            return pd.DataFrame()
        df = pd.DataFrame(self._leaderboard_data)
        if "score" in df.columns:
            is_minimize = str(self.metric).lower() in {"mae", "rmse", "mse", "loss", "log_loss"}
            df = df.sort_values(by="score", ascending=is_minimize).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
        return df

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions using the winning model artifact.
        """
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate class probabilities using the winning model artifact.
        """
        return self.best_model.predict_proba(X)

    def summary(self) -> dict[str, Any]:
        """
        Return an overview dictionary of the execution results.
        """
        return {
            "run_id": self.run_id,
            "task_type": self.task_type,
            "best_model_id": self.best_model_id,
            "best_score": self.best_score,
            "metric": self.metric,
            "total_models_evaluated": len(self._leaderboard_data),
        }

    def save_model(self, path: str | Path) -> Path:
        """
        Save the winning model artifact to disk.
        """
        return self.best_model.save(path)


class AutoML:
    """
    High-level ergonomic facade for CATML.

    Enables simple, scikit-learn compatible model fitting and automated experimentation:
    >>> automl = AutoML()
    >>> result = automl.fit(df, target="churn")
    >>> result.leaderboard()
    >>> result.best_model.save("model.pkl")
    """

    def __init__(
        self,
        task: str | None = None,
        task_type: str | None = None,
        metric: str | None = None,
        cv_folds: int = 5,
        time_budget: int | None = None,
        models: list[str] | None = None,
        random_state: int = 42,
        workspace_dir: str | Path | None = None,
        group_column: str | None = None,
        image_columns: list[str] | None = None,
        image_model: str | None = None,
    ) -> None:
        """
        Initialize the AutoML runner.

        Parameters
        ----------
        task / task_type: str | None
            Type of ML task ('binary_classification', 'multiclass_classification', 'regression', 'clustering').
        metric: str | None
            Evaluation metric (e.g. 'roc_auc', 'f1', 'r2', 'mae', 'rmse', 'mse', 'log_loss').
        cv_folds: int
            Number of cross-validation splits (default: 5).
        time_budget: int | None
            Cooperative time budget in seconds across experiments and trials. Checked before
            starting each candidate trial. Running model fits are allowed to complete gracefully
            without mid-training interruption; subsequent trials are halted once the deadline expires.
        models: list[str] | None
            Specific model identifiers to evaluate.
        random_state: int
            Seed for reproducible dataset splits and model initialization.
        workspace_dir: str | Path | None
            Custom directory path for workspace artifacts and experiment tracking.
        group_column: str | None
            Column name representing entity or cluster groups to enforce non-overlapping splits.
        image_columns: list[str] | None
            Column names containing paths to image files for multimodal representation learning.
        image_model: str | None
            Image encoder backbone ('deterministic', 'resnet18', etc.).
        """
        self.task = task or task_type
        self.metric = metric
        self.cv_folds = cv_folds
        self.time_budget = time_budget
        self.models = list(models) if models is not None else None
        self.random_state = random_state
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.group_column = group_column
        self.image_columns = list(image_columns) if image_columns is not None else None
        self.image_model = image_model
        self._result: AutoMLResult | None = None

    def fit(
        self,
        data: pd.DataFrame | np.ndarray | str | Path,
        target: str | pd.Series | np.ndarray | None = None,
        text_columns: list[str] | None = None,
        image_columns: list[str] | None = None,
        group_column: str | None = None,
        image_model: str | None = None,
    ) -> AutoMLResult:
        """
        Fit multiple candidate models on the dataset, evaluate their performance,
        and construct the winning standalone ModelArtifact.

        Args:
            data: Feature matrix as pandas DataFrame, 2D numpy array, or file path (CSV/Parquet).
            target: Target column name (str) if data is DataFrame/path, or target values
                    (Series/ndarray).
            text_columns: Optional list of column names containing freeform natural language
                          text to be transformed via n-gram TF-IDF representations. If omitted,
                          text columns are discovered automatically via heuristics.
            image_columns: Optional list of column names referencing image files or paths to
                           be transformed via ImageEncoder embeddings. If omitted, image columns
                           are discovered automatically via heuristics.
            group_column: Optional entity or group column name (e.g. 'patient_id', 'user_id')
                          used to enforce GroupKFold validation and eliminate group data leakage.
                          If omitted and group leakage is detected, it is enforced automatically.

        Returns:
            AutoMLResult containing the winning model, leaderboard, and predictions.
        """
        df, target_col = self._standardize_input(data, target)

        resolved_task_type = self._resolve_task_type(df[target_col])

        # Provision workspace
        if self.workspace_dir:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            ws_root = self.workspace_dir
        else:
            ws_root = Path(tempfile.mkdtemp(prefix="catml_fit_"))

        ws, cmd, qry = build_application(root_dir=str(ws_root))

        dataset = ws.register_dataset(
            name="fit_dataset",
            path=df,
            target=target_col,
            task_type=resolved_task_type.value,
        )

        resolved_group_col = group_column or self.group_column
        resolved_image_cols = list(image_columns) if image_columns is not None else (list(self.image_columns) if self.image_columns is not None else None)
        resolved_text_cols = list(text_columns) if text_columns is not None else None
        resolved_image_model = image_model or self.image_model

        extra_cfg: dict[str, Any] = {}
        if resolved_image_cols:
            extra_cfg["image_columns"] = resolved_image_cols
        if resolved_image_model:
            extra_cfg["image_model"] = resolved_image_model
        if resolved_text_cols:
            extra_cfg["text_columns"] = resolved_text_cols
        if self.time_budget:
            extra_cfg["time_budget"] = self.time_budget

        profile = ws.get_dataset_profile(dataset.id)
        if not resolved_group_col and profile and profile.has_group_leakage and profile.group_candidates:
            resolved_group_col = profile.group_candidates[0]

        if resolved_group_col:
            val_strategy = "group_kfold"
        elif self.cv_folds and self.cv_folds > 1:
            val_strategy = "stratified_kfold" if resolved_task_type.value != "regression" else "kfold"
        else:
            val_strategy = "holdout"

        run = ws.create_run(
            dataset,
            metric=self.metric,
            validation_strategy=val_strategy,
            group_column=resolved_group_col,
            cv_folds=self.cv_folds if (self.cv_folds and self.cv_folds > 1) else 5,
            random_seed=self.random_state,
            time_budget_seconds=self.time_budget,
            extra=extra_cfg,
        )

        # Strict anti-leakage resolution: automatically excludes leakage columns and non-predictive IDs
        exclude_cols = [resolved_group_col] if resolved_group_col else None
        if profile and hasattr(profile, "resolve_safe_feature_names"):
            feature_names = profile.resolve_safe_feature_names(
                exclude_columns=exclude_cols,
            )
        else:
            feature_names = ws.get_feature_registry(dataset.id).active_feature_names(target_col)
            if exclude_cols:
                feature_names = [f for f in feature_names if f not in exclude_cols]
            if not feature_names:
                raise ValueError(f"No safe feature candidates available for training on dataset '{dataset.name}'.")

        # Plan and run candidate models
        if self.models:
            experiment = cmd.dispatch(
                CreateExperimentCommand(
                    run_id=run.id,
                    name="custom_selection",
                    feature_names=feature_names,
                    model_ids=self.models,
                    validation_strategy=val_strategy,
                    group_column=resolved_group_col,
                )
            )
            cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=experiment.id))
        else:
            candidates = ws.plan_experiments(run_id=run.id, auto_enqueue=True)
            if candidates:
                ws.run_scheduled_experiments(run_id=run.id, max_experiments=5)
            else:
                # Fallback to registered models for task
                from automl.domain.tasks.task_type import models_for_task

                compatible = [
                    m for m in models_for_task(resolved_task_type)
                    if ws.model_registry.has(m) and m != "voting_ensemble"
                ]
                experiment = ws.create_experiment(
                    run_id=run.id,
                    name="baseline_models",
                    feature_names=feature_names,
                    model_ids=compatible[:4] if compatible else ["logistic_regression"],
                    validation_strategy=val_strategy,
                    group_column=resolved_group_col,
                )
                ws.run_experiment(run_id=run.id, experiment_id=experiment.id)

        # Extract leaderboard (excluding failed trials)
        raw_leaderboard = ws.get_leaderboard_results(run.id, include_failed=False)
        valid_leaderboard = [r for r in raw_leaderboard if r.succeeded]
        if not valid_leaderboard:
            refreshed_run = ws.get_run(run.id)
            if refreshed_run and refreshed_run.config.extra.get("time_budget_exhausted"):
                raise RuntimeError(
                    f"Time budget of {self.time_budget}s expired before any candidate model completed training."
                )
            raise RuntimeError("No models were successfully evaluated during fit().")

        leaderboard_rows = []
        for rank, res in enumerate(valid_leaderboard, start=1):
            t_obj = ws.get_trial(res.trial_id)
            leaderboard_rows.append({
                "rank": rank,
                "model_id": res.model_id,
                "score": res.primary_score,
                "metric": res.primary_metric,
                "training_time_s": round(res.training_time_seconds, 3),
                "cv_std": res.secondary_metrics.get("cv_std") if res.secondary_metrics else None,
                "parameters": t_obj.parameters if t_obj else {},
            })

        # Export self-contained winning artifact
        best_artifact = ws.export_model_artifact(run_id=run.id)

        self._result = AutoMLResult(
            best_model=best_artifact,
            best_score=best_artifact.score,
            best_model_id=best_artifact.model_id,
            task_type=resolved_task_type.value,
            metric=best_artifact.metric,
            run_id=run.id,
            _leaderboard_data=leaderboard_rows,
            _workspace=ws,
        )
        return self._result

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate class probabilities using the fitted model.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.predict_proba(X)

    def leaderboard(self) -> pd.DataFrame:
        """
        Return the leaderboard DataFrame of the last fit() call.
        """
        if self._result is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self._result.leaderboard()

    @classmethod
    def load_model(cls, path: str | Path) -> ModelArtifact:
        """
        Load a standalone saved ModelArtifact from disk.
        """
        return ModelArtifact.load(path)

    def _standardize_input(
        self,
        data: pd.DataFrame | np.ndarray | str | Path,
        target: str | pd.Series | np.ndarray | None,
    ) -> tuple[pd.DataFrame, str]:
        if isinstance(data, Path) or (
            isinstance(data, str)
            and (
                Path(data).exists()
                or any(data.lower().endswith(ext) for ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".tsv"))
            )
        ):
            from automl.engine.profiling.dataset_profiler import load_dataframe
            data = load_dataframe(data)

        if isinstance(data, pd.DataFrame):
            if isinstance(target, str):
                if target not in data.columns:
                    raise ValueError(f"Target column '{target}' not found in DataFrame.")
                return data.copy(), target
            elif target is not None:
                target_series = pd.Series(target)
                target_col = target_series.name or "__catml_target__"
                df = data.copy()
                df[target_col] = target_series.values
                return df, target_col
            else:
                raise ValueError(
                    "Target must be specified as a column name (str) or target values (Series/array)."
                )
        elif isinstance(data, np.ndarray):
            if target is None:
                raise ValueError("Target array must be provided when data is a numpy array.")
            feature_cols = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=feature_cols)
            target_series = pd.Series(target)
            target_col = "__catml_target__"
            df[target_col] = target_series.values
            return df, target_col
        else:
            raise TypeError(f"Unsupported input data type: {type(data)}")

    def _resolve_task_type(self, target_series: pd.Series) -> TaskType:
        if self.task:
            t = self.task.lower().strip()
            if t in {"classification", "clf"}:
                return (
                    TaskType.BINARY_CLASSIFICATION
                    if target_series.nunique(dropna=True) == 2
                    else TaskType.MULTICLASS_CLASSIFICATION
                )
            elif t in {"binary", "binary_classification"}:
                return TaskType.BINARY_CLASSIFICATION
            elif t in {"multiclass", "multiclass_classification"}:
                return TaskType.MULTICLASS_CLASSIFICATION
            elif t in {"regression", "reg"}:
                return TaskType.REGRESSION
            return TaskType.parse(t)
        return infer_task_type(target_series)
