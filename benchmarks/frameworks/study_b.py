"""Study B configs: baseline / catml_base / catml_explore (train-only Explore) / agent stub."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.frameworks.catml_adapter import run_catml_fit
from benchmarks.frameworks.eval import ExternalScores, HoldoutSplit, ResourceTracker, score_external
from benchmarks.frameworks.sklearn_adapter import run_sklearn_baseline


STUDY_B_CONFIGS = (
    "sklearn_baseline",
    "catml_base",
    "catml_explore",
    "catml_agent",
)


@dataclass
class StudyBCellResult:
    config: str
    status: str
    primary_score: float | None
    wall_time_s: float
    peak_rss_mb: float | None = None
    failure_reason: str | None = None
    extra: dict[str, Any] | None = None


def run_study_b_cell(
    *,
    config: str,
    holdout: HoldoutSplit,
    task: str,
    metric: str,
    workspace_dir: Path,
    time_budget: float | None,
    random_state: int,
) -> StudyBCellResult:
    train, test, target = holdout.train, holdout.test, holdout.target
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if config == "sklearn_baseline":
        sk = run_sklearn_baseline(train, test, target, task=task)
        if sk.status != "ok":
            return StudyBCellResult(
                config=config,
                status="failed",
                primary_score=None,
                wall_time_s=sk.wall_time_s,
                peak_rss_mb=sk.peak_rss_mb,
                failure_reason=sk.failure_reason,
            )
        scores = score_external(
            task=task,
            metric=metric,
            y_true=test[target],
            y_pred=sk.y_pred,
            y_proba=sk.y_proba,
        )
        return StudyBCellResult(
            config=config,
            status="ok",
            primary_score=scores.primary,
            wall_time_s=sk.wall_time_s,
            peak_rss_mb=sk.peak_rss_mb,
            extra={"secondary": scores.secondary},
        )

    if config == "catml_base":
        return _catml_cell(
            config=config,
            train=train,
            test=test,
            target=target,
            task=task,
            metric=metric,
            workspace_dir=workspace_dir / "base",
            time_budget=time_budget,
            random_state=random_state,
            explore=False,
        )

    if config == "catml_explore":
        return _catml_cell(
            config=config,
            train=train,
            test=test,
            target=target,
            task=task,
            metric=metric,
            workspace_dir=workspace_dir / "explore",
            time_budget=time_budget,
            random_state=random_state,
            explore=True,
        )

    if config == "catml_agent":
        return StudyBCellResult(
            config=config,
            status="skipped",
            primary_score=None,
            wall_time_s=0.0,
            failure_reason=None,
            extra={
                "note": "Agent/LLM lift deferred; use FakeLLM or Ollama in a later Study B phase",
            },
        )

    return StudyBCellResult(
        config=config,
        status="failed",
        primary_score=None,
        wall_time_s=0.0,
        failure_reason=f"Unknown Study B config: {config}",
    )


def _catml_cell(
    *,
    config: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    workspace_dir: Path,
    time_budget: float | None,
    random_state: int,
    explore: bool,
) -> StudyBCellResult:
    extra: dict[str, Any] = {}
    explore_wall = 0.0
    if explore:
        with ResourceTracker() as tracker:
            try:
                study_meta = _run_explore_on_train_only(
                    train=train,
                    target=target,
                    task=task,
                    workspace_dir=workspace_dir / "explore_ws",
                )
                extra.update(study_meta)
            except Exception as exc:  # noqa: BLE001
                snap = tracker.snapshot()
                return StudyBCellResult(
                    config=config,
                    status="failed",
                    primary_score=None,
                    wall_time_s=snap.wall_time_s,
                    peak_rss_mb=snap.peak_rss_mb,
                    failure_reason=f"explore_failed: {exc}",
                )
            explore_wall = tracker.snapshot().wall_time_s

    cat = run_catml_fit(
        train,
        test,
        target,
        metric=metric,
        workspace_dir=workspace_dir / "fit_ws",
        time_budget=time_budget,
        random_state=random_state,
    )
    if cat.status != "ok":
        return StudyBCellResult(
            config=config,
            status="failed",
            primary_score=None,
            wall_time_s=cat.wall_time_s + explore_wall,
            peak_rss_mb=cat.peak_rss_mb,
            failure_reason=cat.failure_reason,
            extra=extra or None,
        )

    scores = score_external(
        task=task,
        metric=metric,
        y_true=test[target],
        y_pred=cat.y_pred,
        y_proba=cat.y_proba,
    )
    extra["model_id"] = cat.model_id
    extra["secondary"] = scores.secondary
    return StudyBCellResult(
        config=config,
        status="ok",
        primary_score=scores.primary,
        wall_time_s=cat.wall_time_s + explore_wall,
        peak_rss_mb=cat.peak_rss_mb,
        extra=extra,
    )


def _canonical_task_type(task: str) -> str:
    """Map benchmark manifest task labels to workspace task_type enums."""
    t = (task or "").strip().lower()
    aliases = {
        "binary": "binary_classification",
        "binary_classification": "binary_classification",
        "multiclass": "multiclass_classification",
        "multiclass_classification": "multiclass_classification",
        "regression": "regression",
        "clustering": "clustering",
    }
    return aliases.get(t, t)


def _run_explore_on_train_only(
    *,
    train: pd.DataFrame,
    target: str,
    task: str,
    workspace_dir: Path,
) -> dict[str, Any]:
    """Explore study fitted only on train (holdout never registered)."""
    from automl.application.analysis.commands import CreateStudyCommand, RunAnalysisCommand
    from automl.application.analysis.queries import ListFindingsQuery
    from automl.application.bootstrap import build_application

    workspace_dir.mkdir(parents=True, exist_ok=True)
    train_path = workspace_dir / "train_only.csv"
    train.to_csv(train_path, index=False)

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    ds = ws.register_dataset(
        name="study_b_train",
        path=str(train_path),
        target=target,
        task_type=_canonical_task_type(task),
    )
    study_id = cmd.dispatch(
        CreateStudyCommand(
            workspace_id=ws.id,
            dataset_id=ds.id,
            name=f"StudyB {ds.id[:8]}",
            target_column=target,
        )
    )
    run_id = cmd.dispatch(RunAnalysisCommand(study_id=study_id))
    findings = qry.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
    return {
        "study_id": study_id,
        "explore_run_id": run_id,
        "findings_count": len(findings),
        "holdout_policy": "never_seen_during_search",
    }
