"""Register a custom model and execute it through the application buses."""

from pathlib import Path
import argparse

from sklearn.linear_model import LogisticRegression

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand, RunExperimentCommand
from automl.application.queries.workspace_queries import GetLeaderboardQuery
from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.plugins.models.sklearn_plugin import SklearnModelPlugin


class CustomLogisticPlugin(SklearnModelPlugin):
    def __init__(self) -> None:
        super().__init__(
            plugin_id="custom_logistic",
            name="Custom Logistic Regression",
            estimator_factory=LogisticRegression,
            supported_tasks=["binary_classification", "multiclass_classification"],
            default_hyperparameters={"max_iter": 1000, "random_state": 42},
        )

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        space.add(ParameterSpec.float("C", 0.01, 10.0, log=True, default=1.0))
        return space


def run_example(workspace_dir: str, dataset_path: str | Path) -> list[dict]:
    workspace, commands, queries = build_application(root_dir=workspace_dir)
    # Register at application composition time, before issuing experiments.
    workspace.register_plugin(CustomLogisticPlugin())
    dataset = workspace.register_dataset(name="churn", path=dataset_path, target="churn")
    run = workspace.create_run(dataset)
    experiment = commands.dispatch(CreateExperimentCommand(
        run_id=run.id,
        name="custom-model-demo",
        model_ids=["custom_logistic"],
        hypothesis="Evaluate a custom logistic model on the bundled churn dataset.",
    ))
    commands.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=experiment.id))
    return queries.dispatch(GetLeaderboardQuery(run_id=run.id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".automl/custom-plugin-demo")
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[1] / "data/customers_churn.csv"))
    args = parser.parse_args()
    for row in run_example(args.workspace, args.dataset):
        print(row)
