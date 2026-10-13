"""FLAML adapter — shared train/test; skips if not installed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


def _flaml_task(task: str) -> str:
    return "regression" if task == "regression" else "classification"


def _flaml_metric(metric: str, task: str) -> str:
    mapping = {
        "roc_auc": "roc_auc",
        "log_loss": "log_loss",
        "rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "accuracy": "accuracy",
        "f1_macro": "macro_f1",
        "f1": "macro_f1",
    }
    if metric in mapping:
        return mapping[metric]
    return "rmse" if task == "regression" else "roc_auc"


def run_flaml(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    task: str = "binary_classification",
    metric: str = "roc_auc",
    time_budget: float = 300.0,
    random_state: int = 42,
) -> ExternalAdapterResult:
    try:
        from flaml import AutoML as FlamlAutoML
    except ImportError:
        return ExternalAdapterResult(
            status="skipped",
            y_pred=None,
            y_proba=None,
            wall_time_s=0.0,
            failure_reason="flaml_not_installed; pip install flaml",
        )

    with ResourceTracker() as tracker:
        try:
            y_train = train[target]
            X_train = train.drop(columns=[target])
            X_test = test.drop(columns=[target])

            le = None
            y_fit = y_train
            if task != "regression":
                le = LabelEncoder()
                y_fit = le.fit_transform(y_train.astype(str))

            automl = FlamlAutoML()
            settings: dict[str, Any] = {
                "time_budget": float(time_budget),
                "metric": _flaml_metric(metric, task),
                "task": _flaml_task(task),
                "seed": random_state,
                "verbose": 0,
            }
            automl.fit(X_train, y_fit, **settings)
            raw_pred = automl.predict(X_test)
            if le is not None:
                pred = le.inverse_transform(np.asarray(raw_pred).astype(int))
            else:
                pred = np.asarray(raw_pred)
            proba = None
            if task != "regression" and hasattr(automl, "predict_proba"):
                try:
                    proba = np.asarray(automl.predict_proba(X_test))
                except Exception:  # noqa: BLE001
                    proba = None
            snap = tracker.snapshot()
            return ExternalAdapterResult(
                status="ok",
                y_pred=np.asarray(pred),
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
