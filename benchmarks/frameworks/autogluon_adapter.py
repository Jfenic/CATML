"""AutoGluon adapter — shared train/test; skips if not installed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.frameworks.eval import ResourceTracker


@dataclass
class ExternalAdapterResult:
    status: str
    y_pred: np.ndarray | None
    y_proba: np.ndarray | None
    wall_time_s: float
    peak_rss_mb: float | None = None
    cpu_user_s: float | None = None
    cpu_system_s: float | None = None
    failure_reason: str | None = None
    extra: dict[str, Any] | None = None


def _ag_eval_metric(metric: str, task: str) -> str:
    if metric == "roc_auc":
        return "roc_auc"
    if metric == "log_loss":
        return "log_loss"
    if metric in {"f1_macro", "f1"}:
        return "f1_macro"
    if metric == "accuracy":
        return "accuracy"
    if metric == "rmse":
        return "root_mean_squared_error"
    if metric == "mae":
        return "mean_absolute_error"
    return "root_mean_squared_error" if task == "regression" else "roc_auc"


def run_autogluon(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    task: str = "binary_classification",
    metric: str = "roc_auc",
    time_budget: float = 300.0,
    work_dir: str | Path | None = None,
    random_state: int = 42,
) -> ExternalAdapterResult:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        return ExternalAdapterResult(
            status="skipped",
            y_pred=None,
            y_proba=None,
            wall_time_s=0.0,
            failure_reason="autogluon_not_installed; pip install autogluon.tabular",
        )

    with ResourceTracker() as tracker:
        try:
            path = Path(work_dir) if work_dir is not None else Path("AutogluonModels") / "pilot"
            path.mkdir(parents=True, exist_ok=True)
            predictor = TabularPredictor(
                label=target,
                eval_metric=_ag_eval_metric(metric, task),
                path=str(path),
                verbosity=0,
            )
            fit_kwargs: dict[str, Any] = {
                "time_limit": int(max(1, time_budget)),
                "presets": "medium_quality",
            }
            predictor.fit(train, **fit_kwargs)
            X_test = test.drop(columns=[target])
            pred = np.asarray(predictor.predict(X_test))
            proba = None
            if task != "regression":
                try:
                    proba = np.asarray(predictor.predict_proba(X_test))
                except Exception:  # noqa: BLE001
                    proba = None
            snap = tracker.snapshot()
            return ExternalAdapterResult(
                status="ok",
                y_pred=pred,
                y_proba=proba,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
            )
        except Exception as exc:  # noqa: BLE001
            snap = tracker.snapshot()
            return ExternalAdapterResult(
                status="failed",
                y_pred=None,
                y_proba=None,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
                failure_reason=str(exc),
            )
