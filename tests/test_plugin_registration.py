from pathlib import Path
import runpy

import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand
from automl.application.queries.workspace_queries import ListModelsQuery, ListPluginsQuery
from automl.plugins.metrics.business_metrics import WeightedF1MetricPlugin


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/plugins/custom_model.py"


def test_custom_model_registration_and_compatibility(tmp_path):
    plugin = runpy.run_path(str(EXAMPLE))["CustomLogisticPlugin"]()
    workspace, commands, queries = build_application(root_dir=str(tmp_path))
    workspace.register_plugin(plugin)

    assert workspace.plugin_registry.get_model_plugin(plugin.plugin_id) is plugin
    assert any(row["plugin_id"] == plugin.plugin_id for row in queries.dispatch(ListPluginsQuery()))
    assert workspace.model_registry.get(plugin.plugin_id).task_types == plugin.capabilities.supported_tasks
    assert any(row["id"] == plugin.plugin_id for row in queries.dispatch(ListModelsQuery(task_type="binary_classification")))
    assert any(event["event_type"] == "PluginRegistered" for event in workspace.repository.list_events())

    dataset = workspace.register_dataset(
        name="regression", path=EXAMPLE.parents[1] / "data/customers_churn.csv", target="salary", task_type="regression",
    )
    run = workspace.create_run(dataset)
    with pytest.raises(ValueError, match="not compatible"):
        commands.dispatch(CreateExperimentCommand(run_id=run.id, name="invalid", model_ids=[plugin.plugin_id]))


def test_documented_example_trains_custom_model(tmp_path):
    example = runpy.run_path(str(EXAMPLE))
    rows = example["run_example"](str(tmp_path), EXAMPLE.parents[1] / "data/customers_churn.csv")
    assert len(rows) == 1
    assert rows[0]["model_id"] == "custom_logistic"
    assert rows[0]["failed"] is False
    assert 0.0 <= rows[0]["score"] <= 1.0


def test_metric_registration_does_not_add_a_model(tmp_path):
    workspace, _, queries = build_application(root_dir=str(tmp_path))
    workspace.register_plugin(WeightedF1MetricPlugin())
    assert any(row["plugin_id"] == "f_beta" for row in queries.dispatch(ListPluginsQuery(plugin_type="metric")))
    assert workspace.model_registry.get("f_beta") is None
