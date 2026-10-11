"""CATML adapter for Benchmark v1 pilots (public `catml` API)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CatmlPilotResult:
    status: str
    primary_score: float | None
    model_id: str | None
    wall_time_s: float
    failure_reason: str | None = None
    extra: dict[str, Any] | None = None


def run_catml_fit(
    df: pd.DataFrame,
    target: str,
    *,
    metric: str = "roc_auc",
    workspace_dir: str | Path,
    models: list[str] | None = None,
    time_budget: float | None = None,
) -> CatmlPilotResult:
    """Run a bounded CATML AutoML fit and return a status-aware score."""
    import time

    from catml import AutoML

    started = time.perf_counter()
    try:
        automl = AutoML(
            metric=metric,
            models=models or ["logistic_regression", "random_forest"],
            time_budget=time_budget,
            workspace_dir=str(workspace_dir),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        elapsed = time.perf_counter() - started
        if board is None or getattr(board, "empty", True):
            return CatmlPilotResult(
                status="failed",
                primary_score=None,
                model_id=None,
                wall_time_s=elapsed,
                failure_reason="empty_leaderboard",
            )
        top = board.iloc[0]
        score = top.get("score") if hasattr(top, "get") else top["score"]
        model_id = None
        for key in ("model_id", "model", "model_name"):
            if key in board.columns:
                model_id = top[key]
                break
        return CatmlPilotResult(
            status="ok",
            primary_score=float(score) if score is not None else None,
            model_id=str(model_id) if model_id is not None else result.best_model_id,
            wall_time_s=elapsed,
            extra={"leaderboard_size": int(len(board))},
        )
    except Exception as exc:  # noqa: BLE001 — pilot adapter surfaces failures
        return CatmlPilotResult(
            status="failed",
            primary_score=None,
            model_id=None,
            wall_time_s=time.perf_counter() - started,
            failure_reason=str(exc),
        )
