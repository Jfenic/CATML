from pathlib import Path

import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CancelRunCommand,
    CloneRunCommand,
    CreateExperimentCommand,
    CreateFeatureSetCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    PauseRunCommand,
    PrioritizeFeatureCommand,
    ResumeRunCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    CompareExperimentsQuery,
    GetLeaderboardQuery,
    ListFeatureSetsQuery,
    ListModelsQuery,
)
from automl.domain.experiments.priority import ExperimentPriority
from automl.domain.runs.states import RunPhase, RunStatus


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "customers_churn.csv"


@pytest.fixture
def app(tmp_path: Path, sample_dataset_path: Path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
    dataset = ws.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    run = ws.create_run(dataset, metric="roc_auc")
    return ws, cmd, qry, dataset, run


def test_command_bus_exclude_and_prioritize(app) -> None:
    ws, cmd, _, dataset, run = app
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
    cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", score=1.0, run_id=run.id))

    refreshed = ws._get_run(run.id)
    assert "customer_id" in refreshed.config.features_excluded
    assert "salary" in refreshed.config.features_priority

    feature = ws.get_feature_registry(dataset.id).get("customer_id")
    assert feature is not None
    assert feature.status.value == "EXCLUDED"


def test_create_feature_set_via_command(app) -> None:
    ws, cmd, qry, dataset, _ = app
    fset = cmd.dispatch(
        CreateFeatureSetCommand(
            dataset_id=dataset.id,
            name="financial",
            feature_names=["salary", "debt", "age"],
        )
    )
    listed = qry.dispatch(ListFeatureSetsQuery(dataset.id))
    assert len(listed) == 1
    assert listed[0].id == fset.id


def test_experiment_with_feature_set(app) -> None:
    ws, cmd, _, dataset, run = app
    fset = ws.create_feature_set(dataset.id, "core", ["salary", "debt", "tenure", "age"])
    experiment = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="fset_exp",
            feature_set_id=fset.id,
            model_ids=["logistic_regression"],
            priority="high",
        )
    )
    assert experiment.feature_set_id == fset.id
    assert experiment.priority == ExperimentPriority.HIGH.value
    assert "salary" in experiment.feature_names


def test_list_models_query(app) -> None:
    _, cmd, qry, _, run = app
    cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
    models = qry.dispatch(ListModelsQuery(run_id=run.id))
    svc = next(m for m in models if m["id"] == "svc")
    lr = next(m for m in models if m["id"] == "logistic_regression")
    assert svc["excluded"] is True
    assert lr["active"] is True


def test_compare_experiments_query(app) -> None:
    ws, cmd, qry, dataset, run = app
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))

    exp_a = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="all_active",
            model_ids=["logistic_regression"],
        )
    )
    fset = ws.create_feature_set(dataset.id, "financial", ["salary", "debt", "age"])
    exp_b = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="financial_only",
            feature_set_id=fset.id,
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run.id, exp_a.id))
    cmd.dispatch(RunExperimentCommand(run.id, exp_b.id))

    comparison = qry.dispatch(CompareExperimentsQuery([exp_a.id, exp_b.id]))
    assert len(comparison) == 2
    assert all(item["trials"] for item in comparison)


def test_pause_and_resume_from_checkpoint(app) -> None:
    ws, cmd, qry, _, run = app
    experiment = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="resume_test",
            model_ids=["logistic_regression", "random_forest"],
        )
    )
    ws.repository.save_checkpoint(run.id, experiment.id, 1)
    run.transition_to(RunStatus.PAUSED, RunPhase.EXPERIMENT_EXECUTION)
    ws.repository.save_run(run)

    results = cmd.dispatch(ResumeRunCommand(run.id))
    assert len(results) == 1
    assert results[0].model_id == "random_forest"

    leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
    assert len(leaderboard) >= 1


def test_cancel_run(app) -> None:
    ws, cmd, _, _, run = app
    cancelled = cmd.dispatch(CancelRunCommand(run.id))
    assert cancelled.status == RunStatus.CANCELLED


def test_clone_run(app) -> None:
    ws, cmd, _, _, run = app
    cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
    cloned = cmd.dispatch(CloneRunCommand(run.id))
    assert cloned.id != run.id
    assert "svc" in cloned.config.models_exclude


def test_feature_persistence_after_reload(tmp_path: Path, sample_dataset_path: Path) -> None:
    root = tmp_path / "persist"
    ws, cmd, _ = build_application(root_dir=str(root))
    dataset = ws.register_dataset(name="customers", path=sample_dataset_path, target="churn")
    run = ws.create_run(dataset)
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))

    ws2, _, _ = build_application(root_dir=str(root))
    dataset2 = list(ws2._datasets.values())[-1]
    feature = ws2.get_feature_registry(dataset2.id).get("customer_id")
    assert feature is not None
    assert feature.status.value == "EXCLUDED"


def test_events_are_recorded(app) -> None:
    ws, cmd, _, dataset, run = app
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
    events = ws.repository.list_events(run_id=run.id)
    assert any(e["event_type"] == "FeatureExcluded" for e in events)


def test_rejects_experiment_from_another_run(app) -> None:
    ws, cmd, _, dataset, run = app
    other_run = ws.create_run(dataset)
    experiment = cmd.dispatch(
        CreateExperimentCommand(run_id=run.id, name="owned", model_ids=["logistic_regression"])
    )

    with pytest.raises(ValueError, match="does not belong"):
        cmd.dispatch(RunExperimentCommand(other_run.id, experiment.id))


def test_rejects_feature_set_from_another_dataset(app, sample_dataset_path: Path) -> None:
    ws, cmd, _, _, run = app
    other_dataset = ws.register_dataset(
        name="other",
        path=sample_dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    feature_set = ws.create_feature_set(other_dataset.id, "foreign", ["salary", "debt"])

    with pytest.raises(ValueError, match="belongs to dataset"):
        cmd.dispatch(
            CreateExperimentCommand(run_id=run.id, name="invalid", feature_set_id=feature_set.id)
        )


def test_rejects_unknown_model_and_feature(app) -> None:
    _, cmd, _, _, run = app

    with pytest.raises(ValueError, match="not compatible"):
        cmd.dispatch(ExcludeModelCommand(run.id, "not_registered"))
    with pytest.raises(ValueError, match="Unknown features"):
        cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name="invalid_feature",
                feature_names=["not_a_column"],
                model_ids=["logistic_regression"],
            )
        )
