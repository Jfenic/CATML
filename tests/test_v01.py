from pathlib import Path

import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.engine.profiling.dataset_profiler import profile_dataset
from automl.domain.datasets.profile import Dataset
from automl.domain.features.feature import FeatureStatus


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "customers_churn.csv"


@pytest.fixture
def workspace(tmp_path: Path) -> AutoMLWorkspace:
    return AutoMLWorkspace.create("test", root_dir=tmp_path / "ws")


def test_profile_dataset(sample_dataset_path: Path) -> None:
    dataset = Dataset(
        id="ds_test",
        workspace_id="ws_test",
        name="customers",
        path=str(sample_dataset_path),
        target_column="churn",
        task_type="binary_classification",
    )
    profile = profile_dataset(dataset)
    assert profile.row_count == 500
    assert profile.target_column == "churn"
    assert any(c.name == "salary" for c in profile.columns)


def test_register_dataset_and_profile(workspace: AutoMLWorkspace, sample_dataset_path: Path) -> None:
    dataset = workspace.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    profile = workspace.repository.get_dataset_profile(dataset.id)
    assert profile is not None
    assert profile.row_count == 500
    registry = workspace.get_feature_registry(dataset.id)
    assert len(registry.list()) == profile.column_count - 1


def test_exclude_and_prioritize_features(workspace: AutoMLWorkspace, sample_dataset_path: Path) -> None:
    dataset = workspace.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
    )
    workspace.exclude_feature(dataset.id, "customer_id")
    workspace.prioritize_feature(dataset.id, "salary")
    active = workspace.get_feature_registry(dataset.id).active_feature_names("churn")
    assert "customer_id" not in active
    assert active[0] == "salary"


def test_run_experiment_end_to_end(workspace: AutoMLWorkspace, sample_dataset_path: Path) -> None:
    dataset = workspace.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    run = workspace.create_run(dataset, metric="roc_auc")
    workspace.exclude_feature(dataset.id, "customer_id")
    experiment = workspace.create_experiment(
        run,
        name="baseline",
        model_ids=["logistic_regression", "random_forest"],
    )
    results = workspace.run_experiment(run, experiment)
    assert len(results) == 2
    assert all(r.succeeded for r in results)
    assert all(r.primary_score > 0 for r in results)

    leaderboard = workspace.leaderboard(run)
    assert len(leaderboard) == 2
    assert leaderboard[0]["score"] >= leaderboard[1]["score"]


def test_feature_registry_exclude_status() -> None:
    from automl.domain.features.feature import Feature
    from automl.domain.features.registry import FeatureRegistry

    registry = FeatureRegistry([
        Feature(id="f1", dataset_id="d1", name="a"),
        Feature(id="f2", dataset_id="d1", name="b"),
    ])
    registry.exclude("a")
    assert registry.get("a").status == FeatureStatus.EXCLUDED
