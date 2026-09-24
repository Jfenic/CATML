from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.application.plugins.registry import PluginRegistry
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    ListPluginsQuery,
)
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import MetricPluginPort, ModelPluginPort
from automl.engine.plugins.validator import CompatibilityValidator
from automl.plugins.metrics.business_metrics import CostSensitiveMetricPlugin, WeightedF1MetricPlugin
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin
from automl.plugins.models.sklearn_plugin import SklearnModelPlugin


@pytest.fixture
def sample_churn_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "churn_sample.csv"
    np.random.seed(42)
    n = 100
    salary = np.random.uniform(20000, 100000, size=n)
    debt = np.random.uniform(1000, 50000, size=n)
    churn = ((salary / 100000) * 0.7 - (debt / 50000) * 0.3 > 0.2).astype(int)

    df = pd.DataFrame({
        "salary": salary,
        "debt": debt,
        "churn": churn,
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_plugin_capability_and_validator() -> None:
    caps = PluginCapability(
        supported_tasks=["binary_classification", "multiclass_classification"],
        supported_modalities=["tabular"],
        requires_gpu=False,
        supports_proba=True,
    )
    d = caps.to_dict()
    assert "binary_classification" in d["supported_tasks"]
    assert d["supports_proba"] is True

    validator = CompatibilityValidator()
    # Compatible task and modality
    assert validator.is_compatible(caps, task_type="binary_classification", modality="tabular") is True
    validator.validate_compatibility("test_plugin", caps, task_type="binary_classification", modality="tabular")

    # Incompatible task
    assert validator.is_compatible(caps, task_type="regression") is False
    with pytest.raises(ValueError, match="does not support task 'regression'"):
        validator.validate_compatibility("test_plugin", caps, task_type="regression")

    # Incompatible modality
    assert validator.is_compatible(caps, modality="image") is False
    with pytest.raises(ValueError, match="does not support modality 'image'"):
        validator.validate_compatibility("test_plugin", caps, modality="image")


def test_plugin_registry() -> None:
    registry = PluginRegistry()

    lgb = LightGBMPlugin()
    registry.register(lgb)

    assert registry.has("lightgbm") is True
    assert registry.has("unknown") is False

    retrieved = registry.get("lightgbm")
    assert retrieved.name == "LightGBM Gradient Boosting"
    assert retrieved.plugin_type == PluginType.MODEL

    model_p = registry.get_model_plugin("lightgbm")
    assert model_p.plugin_id == "lightgbm"
    assert registry.get("unknown") is None

    with pytest.raises(KeyError, match="Plugin 'unknown' not found"):
        registry.require("unknown")

    with pytest.raises(TypeError, match="is not a ModelPluginPort"):
        cost_metric = CostSensitiveMetricPlugin()
        registry.register(cost_metric)
        registry.get_model_plugin("churn_cost")


def test_sklearn_model_plugin() -> None:
    plugin = SklearnModelPlugin(
        plugin_id="rf_custom",
        name="Custom Random Forest",
        estimator_factory=lambda **hp: __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(**hp),
        supported_tasks=["binary_classification"],
        default_hyperparameters={"n_estimators": 10, "random_state": 42},
    )
    assert plugin.plugin_type == PluginType.MODEL
    estimator = plugin.build_estimator(n_estimators=15)
    assert estimator.n_estimators == 15

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
    y = np.array([0, 0, 1, 1])
    estimator.fit(X, y)
    preds = estimator.predict(X)
    assert len(preds) == 4
    probas = estimator.predict_proba(X)
    assert probas.shape == (4, 2)


def test_gradient_boosting_plugins_fallback_and_training() -> None:
    # Test both LightGBM and XGBoost plugins (works regardless of whether lightgbm/xgboost is installed)
    lgb_plugin = LightGBMPlugin()
    xgb_plugin = XGBoostPlugin()

    X = np.random.normal(size=(50, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for p in [lgb_plugin, xgb_plugin]:
        assert p.capabilities.supported_tasks == ["binary_classification", "multiclass_classification", "regression"]
        model = p.build_estimator(n_estimators=10, learning_rate=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 2)


def test_business_metrics_plugin() -> None:
    cost_metric = CostSensitiveMetricPlugin(cost_fn=150.0, cost_fp=30.0, normalize=True)
    assert cost_metric.plugin_type == PluginType.METRIC
    assert cost_metric.greater_is_better is False

    y_true = np.array([1, 1, 0, 0])
    # Case 1: perfect predictions (0 FN, 0 FP)
    y_pred_perfect = np.array([1, 1, 0, 0])
    loss_perfect = cost_metric.compute(y_true, y_pred_perfect)
    assert loss_perfect == 0.0

    # Case 2: 1 False Negative (cost 150), 1 False Positive (cost 30) -> total 180 / 4 = 45.0
    y_pred_bad = np.array([0, 1, 1, 0])
    loss_bad = cost_metric.compute(y_true, y_pred_bad)
    assert loss_bad == 45.0

    # Test F-beta metric
    f_beta_metric = WeightedF1MetricPlugin(beta=2.0)
    assert f_beta_metric.greater_is_better is True
    f2 = f_beta_metric.compute(y_true, y_pred_perfect)
    assert f2 == 1.0


def test_workspace_plugin_query_bus(tmp_path: Path) -> None:
    ws, cmd, qry = build_application(root_dir=str(tmp_path))

    # Query all plugins
    all_plugins = qry.dispatch(ListPluginsQuery())
    plugin_ids = [p["plugin_id"] for p in all_plugins]
    assert "logistic_regression" in plugin_ids
    assert "random_forest" in plugin_ids
    assert "lightgbm" in plugin_ids
    assert "xgboost" in plugin_ids
    assert "churn_cost" in plugin_ids
    assert "f_beta" in plugin_ids

    # Query by type
    model_plugins = qry.dispatch(ListPluginsQuery(plugin_type="model"))
    assert all(p["plugin_type"] == "model" for p in model_plugins)

    # Query by task_type
    cluster_plugins = qry.dispatch(ListPluginsQuery(task_type="clustering"))
    assert any(p["plugin_id"] == "kmeans" for p in cluster_plugins)
    assert not any(p["plugin_id"] == "logistic_regression" for p in cluster_plugins)


def test_custom_plugin_registration(tmp_path: Path) -> None:
    ws, _, qry = build_application(root_dir=str(tmp_path))

    class DummyMetricPlugin(MetricPluginPort):
        plugin_id = "dummy_loss"
        name = "Dummy Loss"
        version = "0.1.0"
        plugin_type = PluginType.METRIC
        capabilities = PluginCapability(
            supported_tasks=["regression", "binary_classification"],
            supported_modalities=["tabular"],
        )
        greater_is_better = False

        def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(np.abs(y_true - y_pred)))

    ws.register_plugin(DummyMetricPlugin())

    plugins = qry.dispatch(ListPluginsQuery())
    ids = [p["plugin_id"] for p in plugins]
    assert "dummy_loss" in ids


def test_end_to_end_gradient_boosting_experiment(sample_churn_csv: str, tmp_path: Path) -> None:
    ws, cmd, qry = build_application(root_dir=str(tmp_path))
    dataset = ws.register_dataset(
        name="churn_e2e",
        path=sample_churn_csv,
        target="churn",
    )
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="lightgbm_experiment",
            feature_names=["salary", "debt"],
            model_ids=["lightgbm", "xgboost"],
            hypothesis="Evaluate gradient boosting plugins",
        )
    )

    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    lb = qry.dispatch(GetLeaderboardQuery(run.id))
    assert len(lb) == 2
    model_ids_in_lb = {entry["model_id"] for entry in lb}
    assert "lightgbm" in model_ids_in_lb
    assert "xgboost" in model_ids_in_lb
    for entry in lb:
        assert entry["score"] is not None
        assert entry["score"] > 0.0


def test_cli_plugin_and_features_commands(sample_churn_csv: str, tmp_path: Path) -> None:
    from automl.interfaces.cli.main import main as cli_main

    ws_dir = str(tmp_path / "cli_ws")

    # Test plugin list
    rc = cli_main(["plugin", "list", "--workspace", ws_dir])
    assert rc == 0

    # Test plugin list --json
    rc = cli_main(["plugin", "list", "--json", "--workspace", ws_dir])
    assert rc == 0

    # Test task list
    rc = cli_main(["task", "list", "--workspace", ws_dir])
    assert rc == 0

    # Test task plan
    rc = cli_main(["task", "plan", "--dataset", sample_churn_csv, "--workspace", ws_dir])
    assert rc == 0

    # Test features select
    rc = cli_main([
        "features", "select",
        "--dataset", sample_churn_csv,
        "--workspace", ws_dir,
        "--methods", "variance",
        "--top-k", "2",
    ])
    assert rc == 0

    # Test features select --json
    rc = cli_main([
        "features", "select",
        "--dataset", sample_churn_csv,
        "--workspace", ws_dir,
        "--methods", "variance",
        "--top-k", "2",
        "--json",
    ])
    assert rc == 0

    # Test features ablation
    rc = cli_main([
        "features", "ablation",
        "--dataset", sample_churn_csv,
        "--workspace", ws_dir,
        "--max-features", "2",
    ])
    assert rc == 0

