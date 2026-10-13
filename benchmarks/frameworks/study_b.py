"""Study B configs: baseline / catml_base / catml_explore (verify→recipe lift) / catml_agent (FakeLLM)."""
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

# Cap verify cost in smoke / CI (each verify = paired baseline + candidate).
_MAX_HYPOTHESES_TO_VERIFY = 2


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
            mode="base",
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
            mode="explore",
        )

    if config == "catml_agent":
        return _catml_cell(
            config=config,
            train=train,
            test=test,
            target=target,
            task=task,
            metric=metric,
            workspace_dir=workspace_dir / "agent",
            time_budget=time_budget,
            random_state=random_state,
            mode="agent",
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
    mode: str,
) -> StudyBCellResult:
    """mode: base | explore | agent."""
    if mode == "base":
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
                wall_time_s=cat.wall_time_s,
                peak_rss_mb=cat.peak_rss_mb,
                failure_reason=cat.failure_reason,
            )
        scores = score_external(
            task=task,
            metric=metric,
            y_true=test[target],
            y_pred=cat.y_pred,
            y_proba=cat.y_proba,
        )
        return StudyBCellResult(
            config=config,
            status="ok",
            primary_score=scores.primary,
            wall_time_s=cat.wall_time_s,
            peak_rss_mb=cat.peak_rss_mb,
            extra={"model_id": cat.model_id, "secondary": scores.secondary, "recipes_applied": 0},
        )

    with ResourceTracker() as tracker:
        try:
            lift = _run_explore_recipe_lift(
                train=train,
                test=test,
                target=target,
                task=task,
                metric=metric,
                workspace_dir=workspace_dir,
                time_budget=time_budget,
                random_state=random_state,
                with_agent=(mode == "agent"),
            )
        except Exception as exc:  # noqa: BLE001
            snap = tracker.snapshot()
            return StudyBCellResult(
                config=config,
                status="failed",
                primary_score=None,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                failure_reason=f"lift_failed: {exc}",
            )
        snap = tracker.snapshot()

    if lift.get("fallback_fit"):
        # Score already computed in lift via run_catml_fit path
        return StudyBCellResult(
            config=config,
            status="ok",
            primary_score=lift["primary_score"],
            wall_time_s=snap.wall_time_s,
            peak_rss_mb=snap.peak_rss_mb,
            extra=lift.get("extra"),
        )

    scores = score_external(
        task=task,
        metric=metric,
        y_true=test[target],
        y_pred=lift["y_pred"],
        y_proba=lift.get("y_proba"),
    )
    extra = dict(lift.get("extra") or {})
    extra["secondary"] = scores.secondary
    return StudyBCellResult(
        config=config,
        status="ok",
        primary_score=scores.primary,
        wall_time_s=snap.wall_time_s,
        peak_rss_mb=snap.peak_rss_mb,
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


def _internal_run_metric(metric: str) -> str:
    """Map holdout scoring metric to a workspace/CV metric AutoML knows."""
    m = (metric or "").strip().lower()
    if m in {"f1_macro", "accuracy"}:
        return "log_loss"
    return m


def _run_explore_recipe_lift(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    workspace_dir: Path,
    time_budget: float | None,
    random_state: int,
    with_agent: bool,
) -> dict[str, Any]:
    """
    Train-only Explore → verify hypotheses (Propose ≠ Accept) → recipe from evidence → fit.

    Holdout ``test`` is never registered; only used at final ``predict``.
    """
    from automl.application.analysis.commands import (
        CreateStudyCommand,
        RunAnalysisCommand,
        VerifyHypothesisCommand,
    )
    from automl.application.analysis.queries import ListFindingsQuery, ListHypothesesQuery
    from automl.application.bootstrap import build_application
    from automl.application.recipes.commands import (
        CreateExperimentRecipeFromEvidenceCommand,
        SelectExperimentRecipeCommand,
        SubmitExperimentFromRecipeCommand,
    )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    train_path = workspace_dir / "train_only.csv"
    train.to_csv(train_path, index=False)
    test_feat_path = workspace_dir / "holdout_features.csv"
    test.drop(columns=[target]).to_csv(test_feat_path, index=False)

    ws, cmd, qry = build_application(root_dir=str(workspace_dir / "ws"))
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
    run_id_explore = cmd.dispatch(RunAnalysisCommand(study_id=study_id))
    findings = qry.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id_explore))
    hypotheses = qry.dispatch(ListHypothesesQuery(study_id=study_id))

    run_metric = _internal_run_metric(metric)
    run = ws.create_run(
        ds,
        metric=run_metric,
        validation_strategy="kfold",
        cv_folds=3,
        random_seed=random_state,
        time_budget_seconds=time_budget,
    )

    agent_meta: dict[str, Any] = {"agent_invoked": False}
    if with_agent:
        agent_meta = _invoke_fake_agent(ws=ws, run_id=run.id, workspace_dir=workspace_dir)

    verified: list[dict[str, Any]] = []
    for hyp in hypotheses[:_MAX_HYPOTHESES_TO_VERIFY]:
        hyp_id = hyp["id"] if isinstance(hyp, dict) else hyp.id
        try:
            link = cmd.dispatch(
                VerifyHypothesisCommand(
                    hypothesis_id=hyp_id,
                    run_id=run.id,
                    min_improvement=0.0,
                )
            )
            verified.append(link)
        except Exception as exc:  # noqa: BLE001
            verified.append({"hypothesis_id": hyp_id, "verify_error": str(exc)})

    chosen = _pick_evidence_for_recipe(verified)
    extra: dict[str, Any] = {
        "study_id": study_id,
        "explore_run_id": run_id_explore,
        "findings_count": len(findings),
        "hypotheses_count": len(hypotheses),
        "verified_count": sum(1 for v in verified if "id" in v),
        "holdout_policy": "never_seen_during_search",
        "recipes_applied": 0,
        **agent_meta,
    }

    if chosen is None:
        # No verified lift → honest fallback to base fit (still train-only).
        cat = run_catml_fit(
            train,
            test,
            target,
            metric=metric,
            workspace_dir=workspace_dir / "fit_fallback",
            time_budget=time_budget,
            random_state=random_state,
        )
        if cat.status != "ok":
            raise RuntimeError(cat.failure_reason or "fallback fit failed")
        scores = score_external(
            task=task,
            metric=metric,
            y_true=test[target],
            y_pred=cat.y_pred,
            y_proba=cat.y_proba,
        )
        extra.update(
            {
                "model_id": cat.model_id,
                "secondary": scores.secondary,
                "recipe_fallback": "no_verified_evidence",
            }
        )
        return {
            "fallback_fit": True,
            "primary_score": scores.primary,
            "extra": extra,
        }

    recipe_id = cmd.dispatch(
        CreateExperimentRecipeFromEvidenceCommand(
            evidence_link_id=chosen["id"],
            recipe_name=f"study_b_{chosen['id'][:8]}",
            metric=run_metric,
            cv_folds=3,
            random_seed=random_state,
            notes="Study B lift: recipe from verified Explore evidence (Propose ≠ Accept)",
        )
    )
    cmd.dispatch(SelectExperimentRecipeCommand(recipe_id=recipe_id))
    exp_ids = cmd.dispatch(
        SubmitExperimentFromRecipeCommand(recipe_id=recipe_id, run_id=run.id)
    )
    if not exp_ids:
        raise RuntimeError("Recipe submit produced no experiments")

    experiment_id = exp_ids[0]
    y_pred = np.asarray(
        ws.predict(
            run_id=run.id,
            test_dataset_path=test_feat_path,
            experiment_id=experiment_id,
            predict_proba=False,
        )
    )
    y_proba = None
    try:
        y_proba = np.asarray(
            ws.predict(
                run_id=run.id,
                test_dataset_path=test_feat_path,
                experiment_id=experiment_id,
                predict_proba=True,
            )
        )
    except Exception:  # noqa: BLE001
        y_proba = None

    extra.update(
        {
            "recipes_applied": 1,
            "recipe_id": recipe_id,
            "experiment_id": experiment_id,
            "evidence_link_id": chosen["id"],
            "evidence_accepted": bool(chosen.get("accepted")),
            "evidence_improved": bool(chosen.get("improved_in_experiment", chosen.get("accepted"))),
        }
    )
    return {
        "fallback_fit": False,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "extra": extra,
    }


def _pick_evidence_for_recipe(verified: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer statistically accepted evidence; else improved_in_experiment with best delta."""
    usable = [v for v in verified if isinstance(v, dict) and v.get("id") and "verify_error" not in v]
    if not usable:
        return None

    accepted = [v for v in usable if v.get("accepted") or v.get("validated")]
    pool = accepted or [v for v in usable if v.get("improved_in_experiment")]
    if not pool:
        return None

    def _delta(v: dict[str, Any]) -> float:
        try:
            return float(v.get("candidate_score", 0.0)) - float(v.get("baseline_score", 0.0))
        except (TypeError, ValueError):
            return float("-inf")

    return max(pool, key=_delta)


def _invoke_fake_agent(*, ws: Any, run_id: str, workspace_dir: Path) -> dict[str, Any]:
    """Wire FakeLLM + one deterministic agent session step (no holdout access)."""
    from automl.application.agents.orchestrator.session_manager import AgentSessionManager
    from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
    from automl.infrastructure.llm.fake_provider import FakeLLMProvider

    meta: dict[str, Any] = {
        "agent_invoked": True,
        "agent_llm": "fake",
    }
    try:
        llm = FakeLLMProvider()
        _ = llm.generate(prompt="Study B: propose one safe tabular experiment idea as JSON.")
        meta["fake_llm_ok"] = True
    except Exception as exc:  # noqa: BLE001
        meta["fake_llm_ok"] = False
        meta["fake_llm_error"] = str(exc)

    try:
        ledger_path = workspace_dir / "agent_ledger.sqlite"
        store = SqliteAgentLedger(str(ledger_path))
        manager = AgentSessionManager(workspace=ws, session_store=store)
        session = manager.create_session(
            run_id=run_id,
            goal="Study B: improve holdout metric via verified proposals only",
            max_iterations=1,
        )
        step_res = manager.step(session.session_id)
        meta["agent_session_id"] = session.session_id
        meta["agent_steps"] = 1
        meta["agent_step_status"] = getattr(step_res, "status", None) or getattr(
            step_res, "state", None
        )
    except Exception as exc:  # noqa: BLE001
        meta["agent_session_error"] = str(exc)
    return meta
