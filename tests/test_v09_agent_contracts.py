"""Comprehensive unit test suite for V0.9 agentic contracts, DTOs, schemas, policies, and ledger."""
import importlib
from pathlib import Path
import sys
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    Hypothesis,
    OperationStatus,
    PolicyDecisionType,
    ToolEffect,
    ToolErrorCode,
)
from automl.application.agents.contracts import (
    AgentContext,
    AgentSessionState,
    ApprovalRequest,
    OperationRecord,
    PolicyDecision,
    ToolCallContext,
    ToolDefinition,
    ToolError,
    ToolInvocation,
    ToolResult,
)
from automl.application.agents.policy import (
    AgentPolicyConfig,
    PolicyEvaluator,
    compute_arguments_hash,
)
from automl.application.agents.schemas import (
    TOOL_SCHEMAS,
    validate_arguments,
)
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from tests.fixtures.agentic.fixtures_v09 import (
    SAMPLE_INVALID_TOOL_PAYLOADS,
    SAMPLE_VALID_TOOL_PAYLOADS,
    make_sample_approval,
    make_sample_context,
    make_sample_hypothesis,
    make_sample_invocation,
    make_sample_operation,
    make_sample_session_state,
)


def test_domain_hexagonal_boundary():
    """Verify that domain/agents does not import forbidden external frameworks."""
    forbidden = ["pandas", "sklearn", "optuna", "sqlite3", "mcp", "langgraph", "pydantic"]
    for mod in sys.modules:
        if mod.startswith("automl.domain.agents"):
            for f in forbidden:
                assert f not in mod, f"Forbidden import '{f}' detected in domain module '{mod}'"


def test_agent_budget_calculations_and_serialization():
    """Test budget tracking, bounds, affordability checks, and dict round-trip."""
    budget = AgentBudget(
        max_experiments=3,
        max_trials=10,
        max_duration_seconds=120.0,
        max_llm_calls=20,
        max_tokens=50_000,
    )
    assert budget.remaining_experiments() == 3
    assert budget.remaining_trials() == 10
    assert budget.remaining_duration_seconds() == 120.0

    # Test can_afford success
    ok, err = budget.can_afford({"experiments": 1, "trials": 4, "duration_seconds": 30.0})
    assert ok is True
    assert err is None

    # Consume resources
    budget.consume({"experiments": 1, "trials": 4, "duration_seconds": 30.0, "llm_calls": 5, "tokens": 1000})
    assert budget.consumed_experiments == 1
    assert budget.consumed_trials == 4
    assert budget.consumed_duration_seconds == 30.0
    assert budget.consumed_llm_calls == 5
    assert budget.consumed_tokens == 1000
    assert budget.remaining_experiments() == 2

    # Test can_afford failures
    ok, err = budget.can_afford({"experiments": 3})
    assert ok is False
    assert "Experiment budget exceeded" in err

    ok, err = budget.can_afford({"trials": 10})
    assert ok is False
    assert "Trial budget exceeded" in err

    ok, err = budget.can_afford({"duration_seconds": 200.0})
    assert ok is False
    assert "Duration budget exceeded" in err

    ok, err = budget.can_afford({"llm_calls": 20})
    assert ok is False
    assert "LLM calls budget exceeded" in err

    ok, err = budget.can_afford({"tokens": 60_000})
    assert ok is False
    assert "Tokens budget exceeded" in err

    # Round trip
    d = budget.to_dict()
    restored = AgentBudget.from_dict(d)
    assert restored == budget


def test_hypothesis_serialization():
    """Test Hypothesis entity serialization."""
    hyp = make_sample_hypothesis("hyp-test-1", "run-123")
    d = hyp.to_dict()
    restored = Hypothesis.from_dict(d)
    assert restored.hypothesis_id == "hyp-test-1"
    assert restored.target_metric == "roc_auc"
    assert restored.candidate_config["model"] == "xgboost"
    assert restored.baseline_metric == 0.852


def test_contracts_dtos_roundtrip():
    """Test full dictionary round-trip for all application contracts DTOs."""
    # ToolCallContext
    ctx = make_sample_context()
    ctx_d = ctx.to_dict()
    assert ToolCallContext.from_dict(ctx_d) == ctx

    # ToolDefinition
    tdef = ToolDefinition(
        name="test_tool",
        version="1.0.0",
        description="A test tool",
        effect=ToolEffect.MUTATE,
        permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
        cost_estimate={"experiments": 1},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    assert ToolDefinition.from_dict(tdef.to_dict()) == tdef

    # ToolInvocation
    inv = make_sample_invocation(tool_name="test_tool", context=ctx, idempotency_key="key-1")
    assert ToolInvocation.from_dict(inv.to_dict()) == inv

    # ToolError
    terr = ToolError(
        code=ToolErrorCode.INVALID_ARGUMENT,
        message="Bad input",
        details={"field": "dataset_id"},
        retryable=False,
        correlation_id="corr-1",
    )
    assert ToolError.from_dict(terr.to_dict()) == terr

    # ToolResult
    res_success = ToolResult(
        request_id="req-1",
        tool_name="test_tool",
        success=True,
        data={"metric": 0.95},
        operation_id="op-1",
        execution_time_seconds=1.2,
    )
    assert ToolResult.from_dict(res_success.to_dict()) == res_success

    res_failure = ToolResult(
        request_id="req-2",
        tool_name="test_tool",
        success=False,
        error=terr,
        execution_time_seconds=0.1,
    )
    restored_fail = ToolResult.from_dict(res_failure.to_dict())
    assert restored_fail.success is False
    assert restored_fail.error.code == ToolErrorCode.INVALID_ARGUMENT

    # PolicyDecision
    pdec = PolicyDecision(
        decision=PolicyDecisionType.REQUIRE_APPROVAL,
        reason="Needs human approval",
        approval_required=True,
        estimated_cost={"experiments": 1},
    )
    assert PolicyDecision.from_dict(pdec.to_dict()) == pdec

    # ApprovalRequest
    appr = make_sample_approval()
    assert ApprovalRequest.from_dict(appr.to_dict()) == appr

    # OperationRecord
    op = make_sample_operation()
    assert OperationRecord.from_dict(op.to_dict()) == op

    # AgentSessionState
    sess = make_sample_session_state()
    assert AgentSessionState.from_dict(sess.to_dict()) == sess


def test_deterministic_arguments_hashing():
    """Verify arguments hashing is deterministic and independent of dict key order."""
    action = "run_experiment"
    actor = "agent-1"
    run_id = "run-1"
    args1 = {"alpha": 1, "beta": "two", "nested": {"c": 3, "d": 4}}
    args2 = {"beta": "two", "nested": {"d": 4, "c": 3}, "alpha": 1}

    hash1 = compute_arguments_hash(action, actor, run_id, args1, "1.0.0")
    hash2 = compute_arguments_hash(action, actor, run_id, args2, "1.0.0")
    assert hash1 == hash2

    # Different arguments produce different hash
    hash3 = compute_arguments_hash(action, actor, run_id, {"alpha": 2}, "1.0.0")
    assert hash1 != hash3

    # Different action produces different hash
    hash4 = compute_arguments_hash("create_experiment", actor, run_id, args1, "1.0.0")
    assert hash1 != hash4


def test_tool_schemas_validation_valid_payloads():
    """Verify that all canonical valid payloads pass schema validation."""
    for tool_name, payload in SAMPLE_VALID_TOOL_PAYLOADS.items():
        assert tool_name in TOOL_SCHEMAS, f"Missing schema for tool: {tool_name}"
        schema = TOOL_SCHEMAS[tool_name]["input"]
        errors = validate_arguments(schema, payload)
        assert errors == [], f"Validation failed for valid payload of {tool_name}: {errors}"


def test_tool_schemas_validation_invalid_payloads():
    """Verify that canonical invalid payloads are caught by schema validation."""
    for tool_name, payloads in SAMPLE_INVALID_TOOL_PAYLOADS.items():
        schema = TOOL_SCHEMAS[tool_name]["input"]
        for invalid_payload in payloads:
            errors = validate_arguments(schema, invalid_payload)
            assert len(errors) > 0, (
                f"Expected validation failure for {tool_name} with payload {invalid_payload}, but got no errors"
            )


def test_policy_evaluator_scope_and_permissions():
    """Test PolicyEvaluator decisions on scope, permissions, denied models, and budgets."""
    config = AgentPolicyConfig(
        policy_version="1.0.0",
        default_mode=AgentPermission.PROPOSE_ONLY,
        require_approval_for_mutations=True,
        denied_models=["dangerous_nn"],
        allowed_models=["xgboost", "lightgbm", "random_forest"],
    )
    evaluator = PolicyEvaluator(config)
    budget = AgentBudget(max_experiments=2)

    tool_read = ToolDefinition(
        name="get_dataset_profile",
        version="1.0.0",
        description="Read profile",
        effect=ToolEffect.READ,
        permission_required=AgentPermission.READ_ONLY,
    )
    tool_mutate = ToolDefinition(
        name="run_experiment",
        version="1.0.0",
        description="Run experiment",
        effect=ToolEffect.MUTATE,
        permission_required=AgentPermission.EXECUTE_WITHIN_BUDGET,
        cost_estimate={"experiments": 1},
    )

    # 1. Missing scope/actor
    bad_ctx = ToolCallContext(actor="", workspace_path="/tmp", run_id="run-1", correlation_id="c1")
    dec = evaluator.evaluate(tool_read, bad_ctx, budget, {})
    assert dec.decision == PolicyDecisionType.DENY
    assert "Authentication violation" in dec.reason

    bad_ctx_run = ToolCallContext(actor="a1", workspace_path="/tmp", run_id="", correlation_id="c1")
    dec = evaluator.evaluate(tool_read, bad_ctx_run, budget, {})
    assert dec.decision == PolicyDecisionType.DENY
    assert "Scope violation" in dec.reason

    # 2. Denied model
    ctx = make_sample_context(permission=AgentPermission.EXECUTE_WITHIN_BUDGET)
    dec = evaluator.evaluate(tool_mutate, ctx, budget, {"model_name": "dangerous_nn"})
    assert dec.decision == PolicyDecisionType.DENY
    assert "prohibited by policy" in dec.reason

    # 3. Model not in whitelist
    dec = evaluator.evaluate(tool_mutate, ctx, budget, {"model_name": "unknown_model"})
    assert dec.decision == PolicyDecisionType.DENY
    assert "not in allowed models whitelist" in dec.reason

    # 4. READ_ONLY actor trying to mutate
    read_only_ctx = make_sample_context(permission=AgentPermission.READ_ONLY)
    dec = evaluator.evaluate(tool_mutate, read_only_ctx, budget, {"model_name": "xgboost"})
    assert dec.decision == PolicyDecisionType.DENY
    assert "Permission denied" in dec.reason

    # 5. PROPOSE_ONLY actor trying to mutate -> REQUIRE_APPROVAL
    propose_ctx = make_sample_context(permission=AgentPermission.PROPOSE_ONLY)
    dec = evaluator.evaluate(tool_mutate, propose_ctx, budget, {"model_name": "xgboost"})
    assert dec.decision == PolicyDecisionType.REQUIRE_APPROVAL
    assert dec.approval_required is True

    # 6. EXECUTE_WITHIN_BUDGET on MUTATE with require_approval_for_mutations=True -> REQUIRE_APPROVAL
    dec = evaluator.evaluate(tool_mutate, ctx, budget, {"model_name": "xgboost"})
    assert dec.decision == PolicyDecisionType.REQUIRE_APPROVAL

    # 7. ADMIN actor on MUTATE -> ALLOW
    admin_ctx = make_sample_context(permission=AgentPermission.ADMIN)
    dec = evaluator.evaluate(tool_mutate, admin_ctx, budget, {"model_name": "xgboost"})
    assert dec.decision == PolicyDecisionType.ALLOW

    # 8. Budget exhausted -> DENY
    exhausted_budget = AgentBudget(max_experiments=0)
    dec = evaluator.evaluate(tool_mutate, admin_ctx, exhausted_budget, {"model_name": "xgboost"})
    assert dec.decision == PolicyDecisionType.DENY
    assert "Budget exceeded" in dec.reason


def test_sqlite_agent_ledger_lifecycle(tmp_path: Path):
    """Test full SQLite transactional ledger for operations, approvals, hypotheses, and sessions."""
    db_file = tmp_path / "agent_ledger.db"
    ledger = SqliteAgentLedger(db_file)

    # 1. Operation record and idempotency
    op = make_sample_operation(operation_id="op-100", idempotency_key="key-unique-1")
    ledger.record_operation(op)

    # Fetch operation
    saved_op = ledger.get_operation("op-100")
    assert saved_op is not None
    assert saved_op.operation_id == "op-100"
    assert saved_op.status == OperationStatus.PENDING

    # Idempotent re-record with identical arguments hash succeeds
    ledger.record_operation(op)

    # Re-record with same idempotency key but conflicting hash fails
    conflicting_op = make_sample_operation(
        operation_id="op-101",
        idempotency_key="key-unique-1",
        arguments={"different": 123},
    )
    with pytest.raises(ValueError, match="Idempotency conflict"):
        ledger.record_operation(conflicting_op)

    # Update operation status
    updated_op = ledger.update_operation_status(
        operation_id="op-100",
        status=OperationStatus.SUCCEEDED,
        consumed={"experiments": 1, "duration_seconds": 15.2},
        result_ref="trial-result-001",
    )
    assert updated_op.status == OperationStatus.SUCCEEDED
    assert updated_op.consumed_budget["experiments"] == 1
    assert updated_op.result_ref == "trial-result-001"

    # 2. Approval lifecycle
    appr = make_sample_approval(approval_id="appr-100")
    ledger.save_approval(appr)

    saved_appr = ledger.get_approval("appr-100")
    assert saved_appr is not None
    assert saved_appr.status == ApprovalStatus.PENDING

    updated_appr = ledger.update_approval_status("appr-100", ApprovalStatus.APPROVED, reviewer="human_operator")
    assert updated_appr.status == ApprovalStatus.APPROVED
    assert updated_appr.reviewer == "human_operator"

    # 3. Hypothesis lifecycle
    hyp1 = make_sample_hypothesis("hyp-1", "run-100")
    hyp2 = make_sample_hypothesis("hyp-2", "run-100")
    ledger.save_hypothesis(hyp1)
    ledger.save_hypothesis(hyp2)

    saved_hyp = ledger.get_hypothesis("hyp-1")
    assert saved_hyp is not None
    assert saved_hyp.target_metric == "roc_auc"

    hyps = ledger.list_hypotheses("run-100")
    assert len(hyps) == 2
    assert {h.hypothesis_id for h in hyps} == {"hyp-1", "hyp-2"}

    # 4. Session state lifecycle
    sess = make_sample_session_state("sess-100", "run-100")
    ledger.save_session_state(sess)

    saved_sess = ledger.get_session_state("sess-100")
    assert saved_sess is not None
    assert saved_sess.goal == sess.goal
    assert saved_sess.iteration_count == 1

    # 5. Ledger lookup by idempotency key and missing entity lookups
    looked_up_op = ledger.get_operation_by_idempotency_key("run-test-001", "create_experiment", "key-unique-1")
    assert looked_up_op is not None
    assert looked_up_op.operation_id == "op-100"

    assert ledger.get_operation_by_idempotency_key("run-test-001", "create_experiment", "nonexistent") is None
    assert ledger.get_operation("nonexistent-op") is None
    assert ledger.get_approval("nonexistent-appr") is None
    assert ledger.get_hypothesis("nonexistent-hyp") is None
    assert ledger.get_session_state("nonexistent-sess") is None

    with pytest.raises(KeyError, match="Operation not found"):
        ledger.update_operation_status("nonexistent-op", OperationStatus.SUCCEEDED)

    with pytest.raises(KeyError, match="Approval request not found"):
        ledger.update_approval_status("nonexistent-appr", ApprovalStatus.APPROVED)


def test_agent_context_and_policy_config_serialization():
    """Test AgentContext and AgentPolicyConfig serialization round-trips."""
    ctx = AgentContext(
        run_id="run-c1",
        dataset_id="ds-c1",
        profile_summary={"n_rows": 100},
        compatible_models=["xgboost", "lightgbm"],
        leaderboard_summary=[{"model": "xgboost", "score": 0.92}],
        feature_evidence_summary=[{"feature": "f1", "importance": 0.3}],
        active_hypotheses=[{"hypothesis_id": "h1"}],
        budget_status={"remaining_experiments": 4},
        omissions_and_limits=["raw rows omitted"],
    )
    d = ctx.to_dict()
    restored = AgentContext.from_dict(d)
    assert restored.run_id == "run-c1"
    assert restored.compatible_models == ["xgboost", "lightgbm"]

    pcfg = AgentPolicyConfig(
        policy_version="1.1.0",
        default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
        require_approval_for_mutations=False,
        allowed_models=["xgboost"],
        denied_models=["dangerous_nn"],
    )
    pcfg_d = pcfg.to_dict()
    restored_pcfg = AgentPolicyConfig.from_dict(pcfg_d)
    assert restored_pcfg.policy_version == "1.1.0"
    assert restored_pcfg.default_mode == AgentPermission.EXECUTE_WITHIN_BUDGET
    assert restored_pcfg.require_approval_for_mutations is False

    # Default evaluator constructor
    default_evaluator = PolicyEvaluator()
    assert default_evaluator.config.policy_version == "1.0.0"


def test_schema_validator_type_edge_cases():
    """Verify validator handles non-dict roots, non-object schemas, and primitive type mismatches."""
    # Non-dict arguments
    assert "Arguments must be an object/dict" in validate_arguments({}, "string_arg")[0]

    # Non-object root schema
    assert "Root schema must be of type 'object'" in validate_arguments({"type": "string"}, {})[0]

    # Type mismatches for integer, number, boolean, array, object
    schema = {
        "type": "object",
        "properties": {
            "str_field": {"type": "string"},
            "int_field": {"type": "integer", "minimum": 1, "maximum": 10},
            "num_field": {"type": "number", "minimum": 0.5, "maximum": 5.0},
            "bool_field": {"type": "boolean"},
            "arr_int": {"type": "array", "items": {"type": "integer"}},
            "arr_num": {"type": "array", "items": {"type": "number"}},
            "obj_field": {"type": "object"},
        },
    }

    # Integer type error (boolean is not integer)
    errs = validate_arguments(schema, {"int_field": True})
    assert any("must be an integer" in e for e in errs)

    # Integer bounds
    errs = validate_arguments(schema, {"int_field": 0})
    assert any("below minimum" in e for e in errs)
    errs = validate_arguments(schema, {"int_field": 15})
    assert any("exceeds maximum" in e for e in errs)

    # Number type and bounds
    errs = validate_arguments(schema, {"num_field": "not_a_num"})
    assert any("must be a number" in e for e in errs)
    errs = validate_arguments(schema, {"num_field": 0.1})
    assert any("below minimum" in e for e in errs)
    errs = validate_arguments(schema, {"num_field": 9.9})
    assert any("exceeds maximum" in e for e in errs)

    # Boolean type error
    errs = validate_arguments(schema, {"bool_field": "yes"})
    assert any("must be a boolean" in e for e in errs)

    # Array type error and array items type error
    errs = validate_arguments(schema, {"arr_int": "not_an_array"})
    assert any("must be an array" in e for e in errs)
    errs = validate_arguments(schema, {"arr_int": ["not_int"]})
    assert any("must be an integer" in e for e in errs)
    errs = validate_arguments(schema, {"arr_num": ["not_num"]})
    assert any("must be a number" in e for e in errs)

    # Object type error
    errs = validate_arguments(schema, {"obj_field": 123})
    assert any("must be an object" in e for e in errs)

