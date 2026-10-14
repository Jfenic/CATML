"""Clean Arch PR4: run/experiment transitions, Unit of Work, trial outcome atomicity."""
from __future__ import annotations

from pathlib import Path

import pytest

from automl.domain.experiments.transitions import (
    InvalidExperimentTransition,
    assert_valid_experiment_transition,
    can_transition_experiment,
)
from automl.domain.experiments.trial import (
    Experiment,
    ExperimentStatus,
    Trial,
    TrialResult,
    TrialStatus,
)
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.runs.transitions import InvalidRunTransition, assert_valid_run_transition
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.infrastructure.database.unit_of_work import SqliteUnitOfWork


def test_run_rejects_cancelled_to_completed():
    with pytest.raises(InvalidRunTransition, match="CANCELLED"):
        assert_valid_run_transition(RunStatus.CANCELLED, RunStatus.COMPLETED)


def test_run_transition_to_enforces_map():
    run = AutoMLRun(
        id="r1",
        workspace_id="w",
        dataset_id="d",
        config=RunConfig(target="y", task_type="binary_classification"),
        status=RunStatus.CANCELLED,
    )
    with pytest.raises(InvalidRunTransition):
        run.transition_to(RunStatus.COMPLETED)


def test_experiment_allows_retry_but_not_cancelled_revival():
    assert can_transition_experiment(ExperimentStatus.FAILED, ExperimentStatus.RUNNING)
    assert can_transition_experiment(ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING)
    with pytest.raises(InvalidExperimentTransition):
        assert_valid_experiment_transition(ExperimentStatus.CANCELLED, ExperimentStatus.RUNNING)
    assert can_transition_experiment(ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED)


def test_experiment_transition_to_happy_path():
    exp = Experiment(
        id="e1",
        run_id="r1",
        name="n",
        hypothesis="h",
        feature_names=["a"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
    )
    exp.transition_to(ExperimentStatus.RUNNING)
    exp.transition_to(ExperimentStatus.COMPLETED)
    exp.transition_to(ExperimentStatus.RUNNING)  # retry allowed
    exp.transition_to(ExperimentStatus.CANCELLED)
    with pytest.raises(InvalidExperimentTransition):
        exp.transition_to(ExperimentStatus.RUNNING)


def test_save_run_rejects_illegal_transition(tmp_path: Path):
    repo = SQLiteExperimentRepository(tmp_path / "a.db")
    run = AutoMLRun(
        id="r1",
        workspace_id="w",
        dataset_id="d",
        config=RunConfig(target="y", task_type="binary_classification"),
        status=RunStatus.CANCELLED,
    )
    repo.save_run(run)
    run.status = RunStatus.COMPLETED  # bypass domain method
    with pytest.raises(InvalidRunTransition):
        repo.save_run(run)


def test_persist_trial_outcome_atomic(tmp_path: Path):
    repo = SQLiteExperimentRepository(tmp_path / "b.db")
    exp = Experiment(
        id="e1",
        run_id="r1",
        name="n",
        hypothesis="h",
        feature_names=["a"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
        status=ExperimentStatus.RUNNING,
    )
    repo.save_experiment(exp)
    trial = Trial(
        id="t1",
        experiment_id="e1",
        model_id="logistic_regression",
        status=TrialStatus.COMPLETED,
    )
    result = TrialResult(
        trial_id="t1",
        experiment_id="e1",
        model_id="logistic_regression",
        primary_metric="roc_auc",
        primary_score=0.9,
    )
    exp.transition_to(ExperimentStatus.COMPLETED)
    repo.persist_trial_outcome(trial, result, experiment=exp)

    assert repo.get_trial("t1") is not None
    assert repo.list_trial_results("e1")[0].primary_score == 0.9
    assert repo.get_experiment("e1").status == ExperimentStatus.COMPLETED


def test_unit_of_work_rolls_back_on_error(tmp_path: Path):
    db = tmp_path / "c.db"
    repo = SQLiteExperimentRepository(db)
    with pytest.raises(RuntimeError, match="boom"):
        with SqliteUnitOfWork(db) as uow:
            uow.connection.execute(
                "INSERT INTO events (id, timestamp, event_type, payload_json) VALUES (?, ?, ?, ?)",
                ("evt1", "2026-01-01T00:00:00+00:00", "Test", "{}"),
            )
            raise RuntimeError("boom")
    with repo._connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM events WHERE id = 'evt1'").fetchone()["c"]
    assert n == 0
