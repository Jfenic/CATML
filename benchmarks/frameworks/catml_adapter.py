"""CATML adapter — fit on shared train only; predict shared test for external scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.frameworks.eval import ResourceTracker


@dataclass
class CatmlPilotResult:
    status: str
    y_pred: np.ndarray | None
    y_proba: np.ndarray | None
    model_id: str | None
    wall_time_s: float
    peak_rss_mb: float | None = None
    cpu_user_s: float | None = None
    cpu_system_s: float | None = None
    failure_reason: str | None = None
    extra: dict[str, Any] | None = None


def _default_models(metric: str) -> list[str]:
    m = str(metric).lower()
    if m in {"mae", "rmse", "mse", "r2"}:
        return ["ridge", "random_forest"]
    return ["logistic_regression", "random_forest"]


def run_catml_fit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    metric: str = "roc_auc",
    workspace_dir: str | Path,
    models: list[str] | None = None,
    time_budget: float | None = None,
    random_state: int = 42,
) -> CatmlPilotResult:
    from catml import AutoML

    with ResourceTracker() as tracker:
        try:
            automl = AutoML(
                metric=metric if metric not in {"f1_macro", "accuracy"} else "log_loss",
                models=models or _default_models(metric),
                time_budget=time_budget,
                workspace_dir=str(workspace_dir),
                random_state=random_state,
            )
            # Internal CV may use metric above; public score comes from external holdout.
            result = automl.fit(train, target=target)
            X_test = test.drop(columns=[target])
            pred = np.asarray(result.predict(X_test))
            proba = None
            try:
                proba = np.asarray(result.predict_proba(X_test))
            except Exception:  # noqa: BLE001
                proba = None
            board = result.leaderboard()
            model_id = result.best_model_id
            if board is not None and not getattr(board, "empty", True):
                top = board.iloc[0]
                for key in ("model_id", "model", "model_name"):
                    if key in board.columns:
                        model_id = top[key]
                        break
            snap = tracker.snapshot()
            return CatmlPilotResult(
                status="ok",
                y_pred=pred,
                y_proba=proba,
                model_id=str(model_id) if model_id is not None else None,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
                extra={
                    "leaderboard_size": int(len(board)) if board is not None else 0,
                    "internal_top_score": (
                        float(board.iloc[0]["score"])
                        if board is not None and not board.empty and "score" in board.columns
                        else None
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            snap = tracker.snapshot()
            return CatmlPilotResult(
                status="failed",
                y_pred=None,
                y_proba=None,
                model_id=None,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
                failure_reason=str(exc),
            )
