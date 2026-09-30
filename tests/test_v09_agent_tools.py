"""Comprehensive tests for V0.9 Agent ToolRegistry, ToolExecutor, and query tools catalog."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    PolicyDecisionType,
    ToolEffect,
    ToolErrorCode,
)
from automl.domain.datasets.profile import ColumnProfile, DatasetProfile
from automl.domain.experiments.trial import Experiment, ExperimentStatus, TrialResult
from automl.domain.features.evidence import FeatureEvidence
from automl.domain.features.selection_strategy import FeatureRank
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.application.agents.contracts import (
    ToolCallContext,
    ToolDefinition,
    ToolInvocation,
)
from automl.application.agents.executor import (
    ToolExecutor,
    create_read_only_tool_registry,
)
from automl.application.agents.policy import AgentPolicyConfig, PolicyEvaluator
from automl.application.agents.registry import ToolRegistry
from automl.application.bus.query_bus import QueryBus
from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetFeatureEvidenceQuery,
    GetFeatureRankingQuery,
    GetLeaderboardQuery,
    ListExperimentsQuery,
    ListModelsQuery,
    ListPluginsQuery,
)
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from tests.fixtures.agentic.fixtures_v09 import make_sample_context


class DummyPlugin:
    def __init__(self, plugin_id: str, name: str, plugin_type: PluginType):
        self.plugin_id = plugin_id
        self.name = name
        self.version = "1.0.0"
        self.plugin_type = plugin_type
        self.capabilities = PluginCapability(supported_tasks=["binary_classification"])


@pytest.fixture
def mock_query_bus():
    """Build a QueryBus pre-configured with mock query handlers returning typed domain entities."""
    bus = QueryBus()

    # 1. Dataset profile
    profile = DatasetProfile(
        dataset_id="ds-test-01",
        row_count=1000,
        column_count=4,
        target_column="churn",
        task_type="binary_classification",
        columns=[
            ColumnProfile(name="age", dtype="int64", null_count=0, unique_count=50),
            ColumnProfile(name="balance", dtype="float64", null_count=5, unique_count=800),
            ColumnProfile(name="geo", dtype="object", null_count=0, unique_count=3),
            ColumnProfile(name="churn", dtype="int64", null_count=0, unique_count=2),
        ],
    )
    bus.register(
        GetDatasetProfileQuery,
        lambda q: profile if q.dataset_id == "ds-test-01" else None,
    )

    # 2. Models
    bus.register(
        ListModelsQuery,
        lambda _q: ["logistic_regression", "random_forest", "xgboost", "lightgbm"],
    )

    # 3. Plugins
    bus.register(
        ListPluginsQuery,
        lambda _q: [
            DummyPlugin("xgb", "XGBoostPlugin", PluginType.MODEL),
            DummyPlugin("lgb", "LightGBMPlugin", PluginType.MODEL),
        ],
    )

    # 4. Experiments
    bus.register(
        ListExperimentsQuery,
        lambda q: [
            Experiment(
                id="exp-101",
                run_id=q.run_id,
                name="Baseline RF",
                hypothesis="Baseline random forest",
                feature_names=["age", "balance"],
                model_ids=["random_forest"],
                metric="roc_auc",
                status=ExperimentStatus.COMPLETED,
            ),
            Experiment(
                id="exp-102",
                run_id=q.run_id,
                name="XGBoost Tuning",
                hypothesis="Tuned XGBoost",
                feature_names=["age", "balance", "geo"],
                model_ids=["xgboost"],
                metric="roc_auc",
                status=ExperimentStatus.RUNNING,
            ),
        ]
        if q.run_id == "run-001"
        else [],
    )

    # 5. Leaderboard
    bus.register(
        GetLeaderboardQuery,
        lambda q: [
            TrialResult(
                trial_id="tr-01",
                experiment_id="exp-101",
                model_id="xgboost",
                primary_metric="roc_auc",
                primary_score=0.9234,
                training_time_seconds=12.5,
            ),
            TrialResult(
                trial_id="tr-02",
                experiment_id="exp-102",
                model_id="random_forest",
                primary_metric="roc_auc",
                primary_score=0.8812,
                training_time_seconds=8.0,
            ),
        ]
        if q.run_id == "run-001"
        else [],
    )

    # 6. Feature evidence
    bus.register(
        GetFeatureEvidenceQuery,
        lambda q: FeatureEvidence(
            feature_id=q.feature_id,
            mutual_information=0.15,
            shap_importance=0.28,
            confidence=0.95,
        )
        if q.run_id == "run-001" and q.feature_id == "balance"
        else None,
    )

    # 7. Feature ranking
    bus.register(
        GetFeatureRankingQuery,
        lambda q: [
            FeatureRank(feature_name="balance", score=0.85, rank=1, method="ensemble"),
            FeatureRank(feature_name="age", score=0.62, rank=2, method="ensemble"),
        ]
        if q.run_id == "run-001"
        else [],
    )

    return bus


def test_tool_registry_management():
    """Verify tool registration, lookup, duplicate checks, and unregistration."""
    registry = ToolRegistry()
    tdef = ToolDefinition(
        name="test_tool",
        version="1.0.0",
        description="Demo tool",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
    )

    registry.register(tdef, lambda args, ctx: {"ok": True})
    assert registry.has_tool("test_tool") is True
    assert registry.get_definition("test_tool") == tdef
    assert registry.get_handler("test_tool") is not None
    assert len(registry.list_tools()) == 1

    # Duplicate registration fails
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tdef, lambda args, ctx: None)

    # Unregister
    registry.unregister("test_tool")
    assert registry.has_tool("test_tool") is False
    assert registry.get_definition("test_tool") is None
    assert registry.get_handler("test_tool") is None


def test_tool_executor_error_handling_and_scope(tmp_path: Path):
    """Test ToolExecutor error mapping for missing tools, schema errors, policy denials, and scope violations."""
    registry = ToolRegistry()
    evaluator = PolicyEvaluator(
        AgentPolicyConfig(
            policy_version="1.0.0",
            default_mode=AgentPermission.PROPOSE_ONLY,
            require_approval_for_mutations=True,
            denied_models=["forbidden_model"],
        )
    )
    ledger = SqliteAgentLedger(tmp_path / "ledger.db")
    executor = ToolExecutor(registry=registry, policy_evaluator=evaluator, ledger=ledger)

    ctx = make_sample_context(run_id="run-001", permission=AgentPermission.EXECUTE_WITHIN_BUDGET)

    # 1. Unregistered tool -> NOT_FOUND
    inv_not_found = ToolInvocation(tool_name="nonexistent_tool", arguments={}, context=ctx)
    res = executor.execute(inv_not_found)
    assert res.success is False
    assert res.error.code == ToolErrorCode.NOT_FOUND

    # 2. Tool without handler -> INTERNAL_ERROR
    tool_no_handler = ToolDefinition(
        name="no_handler",
        version="1.0.0",
        description="Missing handler",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
    )
    registry._tools["no_handler"] = tool_no_handler  # directly insert without handler
    inv_no_h = ToolInvocation(tool_name="no_handler", arguments={}, context=ctx)
    res = executor.execute(inv_no_h)
    assert res.success is False
    assert res.error.code == ToolErrorCode.INTERNAL_ERROR

    # 3. Invalid schema arguments -> INVALID_ARGUMENT
    tool_with_schema = ToolDefinition(
        name="schema_tool",
        version="1.0.0",
        description="Schema tool",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
        input_schema={"type": "object", "properties": {"dataset_id": {"type": "string"}}, "required": ["dataset_id"]},
    )
    registry.register(tool_with_schema, lambda args, _ctx: {"dataset_id": args["dataset_id"]})

    inv_invalid_args = ToolInvocation(tool_name="schema_tool", arguments={}, context=ctx)
    res = executor.execute(inv_invalid_args)
    assert res.success is False
    assert res.error.code == ToolErrorCode.INVALID_ARGUMENT
    assert "Missing required parameter" in res.error.message

    # 4. Scope isolation violation (Blackboard Issue #14: cross-run access)
    tool_scoped = ToolDefinition(
        name="scoped_tool",
        version="1.0.0",
        description="Scoped tool",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
        input_schema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
    )
    registry.register(tool_scoped, lambda args, _ctx: {"run_id": args["run_id"]})

    inv_scope_violation = ToolInvocation(tool_name="scoped_tool", arguments={"run_id": "other-run-999"}, context=ctx)
    res = executor.execute(inv_scope_violation)
    assert res.success is False
    assert res.error.code == ToolErrorCode.SCOPE_VIOLATION
    assert "cannot access run_id 'other-run-999'" in res.error.message

    # Policy-level scope denial (empty run_id in context)
    empty_run_ctx = ToolCallContext(actor="a1", workspace_path="/tmp", run_id="", correlation_id="c1")
    res_scope_deny = executor.execute(ToolInvocation(tool_name="scoped_tool", arguments={"run_id": "r1"}, context=empty_run_ctx))
    assert res_scope_deny.success is False
    assert res_scope_deny.error.code == ToolErrorCode.SCOPE_VIOLATION

    # 5. Policy approval required on mutating tools
    tool_mutate = ToolDefinition(
        name="mutate_tool",
        version="1.0.0",
        description="Mutate tool",
        effect=ToolEffect.MUTATE,
        permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
        cost_estimate={"experiments": 1},
    )
    registry.register(tool_mutate, lambda args, _ctx: {"done": True})

    inv_mutate = ToolInvocation(tool_name="mutate_tool", arguments={}, context=ctx)
    res = executor.execute(inv_mutate)
    assert res.success is False
    assert res.error.code == ToolErrorCode.APPROVAL_REQUIRED
    assert res.approval_id is not None
    # Check that approval was persisted in ledger
    saved_appr = ledger.get_approval(res.approval_id)
    assert saved_appr is not None
    assert saved_appr.action == "mutate_tool"

    # 6. Handler exception mapping
    def _failing_handler(args, _ctx):
        if args.get("mode") == "key":
            raise KeyError("item_not_found")
        if args.get("mode") == "val":
            raise ValueError("bad_value")
        if args.get("mode") == "perm":
            raise PermissionError("unauthorized")
        raise RuntimeError("unexpected_crash")

    tool_fail = ToolDefinition(
        name="failing_tool",
        version="1.0.0",
        description="Fails",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
    )
    registry.register(tool_fail, _failing_handler)

    res_key = executor.execute(ToolInvocation(tool_name="failing_tool", arguments={"mode": "key"}, context=ctx))
    assert res_key.error.code == ToolErrorCode.NOT_FOUND

    res_val = executor.execute(ToolInvocation(tool_name="failing_tool", arguments={"mode": "val"}, context=ctx))
    assert res_val.error.code == ToolErrorCode.INVALID_ARGUMENT

    res_perm = executor.execute(ToolInvocation(tool_name="failing_tool", arguments={"mode": "perm"}, context=ctx))
    assert res_perm.error.code == ToolErrorCode.PERMISSION_DENIED

    res_crash = executor.execute(ToolInvocation(tool_name="failing_tool", arguments={"mode": "crash"}, context=ctx))
    assert res_crash.error.code == ToolErrorCode.INTERNAL_ERROR


def test_canonical_read_only_tool_catalog(mock_query_bus):
    """Test the full set of 7 canonical query tools registered by create_read_only_tool_registry."""
    registry = create_read_only_tool_registry(mock_query_bus)
    executor = ToolExecutor(registry=registry)
    ctx = make_sample_context(run_id="run-001", permission=AgentPermission.READ_ONLY)

    # All tools must have effect == READ and permission == READ_ONLY
    for tool in registry.list_tools():
        assert tool.effect == ToolEffect.READ
        assert tool.permission_required == AgentPermission.READ_ONLY

    # 1. get_dataset_profile
    res_prof = executor.execute(
        ToolInvocation(
            tool_name="get_dataset_profile",
            arguments={"dataset_id": "ds-test-01"},
            context=ctx,
        )
    )
    assert res_prof.success is True
    assert res_prof.data["dataset_id"] == "ds-test-01"
    assert res_prof.data["n_rows"] == 1000
    assert res_prof.data["n_columns"] == 4
    assert res_prof.data["columns"] == ["age", "balance", "geo", "churn"]

    # Not found dataset
    res_prof_nf = executor.execute(
        ToolInvocation(
            tool_name="get_dataset_profile",
            arguments={"dataset_id": "nonexistent"},
            context=ctx,
        )
    )
    assert res_prof_nf.success is False
    assert res_prof_nf.error.code == ToolErrorCode.NOT_FOUND

    # 2. list_models
    res_models = executor.execute(
        ToolInvocation(
            tool_name="list_models",
            arguments={"task_type": "binary_classification"},
            context=ctx,
        )
    )
    assert res_models.success is True
    assert "xgboost" in res_models.data["models"]
    assert "random_forest" in res_models.data["models"]

    # 3. list_plugins
    res_plugins = executor.execute(
        ToolInvocation(
            tool_name="list_plugins",
            arguments={"plugin_type": "model"},
            context=ctx,
        )
    )
    assert res_plugins.success is True
    assert len(res_plugins.data["plugins"]) == 2
    assert res_plugins.data["plugins"][0]["plugin_id"] == "xgb"

    # 4. list_experiments
    res_exp = executor.execute(
        ToolInvocation(
            tool_name="list_experiments",
            arguments={"run_id": "run-001"},
            context=ctx,
        )
    )
    assert res_exp.success is True
    assert len(res_exp.data["experiments"]) == 2
    assert res_exp.data["experiments"][0]["experiment_id"] == "exp-101"
    assert res_exp.data["experiments"][0]["status"] == "COMPLETED"

    # 5. get_leaderboard (with top_k)
    res_lb = executor.execute(
        ToolInvocation(
            tool_name="get_leaderboard",
            arguments={"run_id": "run-001", "top_k": 1},
            context=ctx,
        )
    )
    assert res_lb.success is True
    assert len(res_lb.data["leaderboard"]) == 1
    assert res_lb.data["leaderboard"][0]["trial_id"] == "tr-01"
    assert res_lb.data["leaderboard"][0]["score"] == 0.9234

    # 6. get_feature_evidence
    res_ev = executor.execute(
        ToolInvocation(
            tool_name="get_feature_evidence",
            arguments={"run_id": "run-001", "feature_id": "balance"},
            context=ctx,
        )
    )
    assert res_ev.success is True
    assert res_ev.data["evidence"]["feature_id"] == "balance"
    assert res_ev.data["evidence"]["mutual_information"] == 0.15

    # Feature evidence not found
    res_ev_nf = executor.execute(
        ToolInvocation(
            tool_name="get_feature_evidence",
            arguments={"run_id": "run-001", "feature_id": "unknown_col"},
            context=ctx,
        )
    )
    assert res_ev_nf.success is False
    assert res_ev_nf.error.code == ToolErrorCode.NOT_FOUND

    # 7. get_feature_ranking
    res_ranks = executor.execute(
        ToolInvocation(
            tool_name="get_feature_ranking",
            arguments={"run_id": "run-001"},
            context=ctx,
        )
    )
    assert res_ranks.success is True
    assert len(res_ranks.data["ranks"]) == 2
    assert res_ranks.data["ranks"][0]["feature_name"] == "balance"
    assert res_ranks.data["ranks"][0]["rank"] == 1


def test_tool_executor_budget_denial_and_cost_consumption():
    """Verify ToolExecutor handles budget exhaustion denial and cost consumption."""
    registry = ToolRegistry()
    tool_costly = ToolDefinition(
        name="costly_read",
        version="1.0.0",
        description="Costly tool",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
        cost_estimate={"experiments": 2},
    )
    registry.register(tool_costly, lambda args, _ctx: {"done": True})

    evaluator = PolicyEvaluator()
    budget = AgentBudget(max_experiments=3)
    executor = ToolExecutor(registry=registry, policy_evaluator=evaluator, default_budget=budget)
    ctx = make_sample_context(run_id="run-001", permission=AgentPermission.READ_ONLY)

    # 1. Execute successfully and verify budget consumption
    res = executor.execute(ToolInvocation(tool_name="costly_read", arguments={}, context=ctx), budget=budget)
    assert res.success is True
    assert budget.consumed_experiments == 2
    assert budget.remaining_experiments() == 1

    # 2. Execute again when budget cannot afford 2 more -> DENY with BUDGET_EXCEEDED
    res_denied = executor.execute(ToolInvocation(tool_name="costly_read", arguments={}, context=ctx), budget=budget)
    assert res_denied.success is False
    assert res_denied.error.code == ToolErrorCode.BUDGET_EXCEEDED
    assert "Budget exceeded" in res_denied.error.message


def test_query_tools_plain_dict_fallbacks():
    """Verify fallback normalization when queries return dicts or raw objects without to_dict."""
    bus = QueryBus()
    # Feature evidence as raw dict
    bus.register(
        GetFeatureEvidenceQuery,
        lambda q: {"feature_id": q.feature_id, "score": 0.42},
    )
    # Feature rank as raw object without to_dict
    class RawRank:
        feature_name = "raw_feat"
        score = 0.5
        rank = 1
        method = "custom"

    bus.register(GetFeatureRankingQuery, lambda _q: [RawRank()])

    registry = create_read_only_tool_registry(bus)
    executor = ToolExecutor(registry=registry)
    ctx = make_sample_context(run_id="run-001", permission=AgentPermission.READ_ONLY)

    res_ev = executor.execute(
        ToolInvocation(
            tool_name="get_feature_evidence",
            arguments={"run_id": "run-001", "feature_id": "raw_f"},
            context=ctx,
        )
    )
    assert res_ev.success is True
    assert res_ev.data["evidence"]["feature_id"] == "raw_f"

    res_rk = executor.execute(
        ToolInvocation(
            tool_name="get_feature_ranking",
            arguments={"run_id": "run-001"},
            context=ctx,
        )
    )
    assert res_rk.success is True
    assert res_rk.data["ranks"][0]["feature_name"] == "raw_feat"

