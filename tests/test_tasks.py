from pathlib import Path

import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand, RunExperimentCommand
from automl.application.queries.workspace_queries import GetTaskPlanQuery, ListModelsQuery, ListTaskTypesQuery
from automl.domain.tasks.task_type import TaskType
from automl.plugins.models.sklearn_models import build_sklearn_model


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "customers_churn.csv"


def test_list_task_types_catalog() -> None:
    ws, _, qry = build_application()
    tasks = qry.dispatch(ListTaskTypesQuery())
    task_types = {t["task_type"] for t in tasks}
    assert TaskType.BINARY_CLASSIFICATION.value in task_types
    assert TaskType.CLUSTERING.value in task_types
    binary = next(t for t in tasks if t["task_type"] == "binary_classification")
    assert "logistic_regression" in binary["models"]
    clustering = next(t for t in tasks if t["task_type"] == "clustering")
    assert "kmeans" in clustering["models"]


def test_infer_binary_classification_plan(tmp_path: Path, sample_dataset_path: Path) -> None:
    ws, _, qry = build_application(root_dir=str(tmp_path / "task"))
    dataset = ws.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
    )
    plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
    assert plan.task_type == TaskType.BINARY_CLASSIFICATION
    assert plan.default_metric == "roc_auc"
    assert "logistic_regression" in plan.recommended_models
    assert "kmeans" not in plan.recommended_models


def test_models_filtered_by_task_on_run(tmp_path: Path, sample_dataset_path: Path) -> None:
    ws, _, qry = build_application(root_dir=str(tmp_path / "models"))
    dataset = ws.register_dataset(name="customers", path=sample_dataset_path, target="churn")
    run = ws.create_run(dataset)
    models = qry.dispatch(ListModelsQuery(run_id=run.id))
    ids = {m["id"] for m in models}
    assert "logistic_regression" in ids
    assert "kmeans" not in ids


def test_rejects_incompatible_model_for_task(tmp_path: Path, sample_dataset_path: Path) -> None:
    ws, cmd, _ = build_application(root_dir=str(tmp_path / "reject"))
    dataset = ws.register_dataset(name="customers", path=sample_dataset_path, target="churn")
    run = ws.create_run(dataset)
    with pytest.raises(ValueError, match="not compatible"):
        cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name="bad",
                model_ids=["kmeans"],
            )
        )


def test_clustering_task_plan_and_run(tmp_path: Path, sample_dataset_path: Path) -> None:
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "cluster"))
    dataset = ws.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
        task_type="clustering",
    )
    plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
    assert plan.task_type == TaskType.CLUSTERING
    assert plan.default_metric == "silhouette"

    run = ws.create_run(dataset)
    features = [f.name for f in ws.get_feature_registry(dataset.id).list() if f.name != "customer_id"]
    experiment = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="clustering_exp",
            feature_names=features,
            model_ids=["kmeans"],
        )
    )
    results = cmd.dispatch(RunExperimentCommand(run.id, experiment.id))
    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].primary_metric == "silhouette"


def test_svc_classifier_exposes_calibrated_probabilities() -> None:
    model = build_sklearn_model("svc", "binary_classification")
    assert hasattr(model, "predict_proba")
