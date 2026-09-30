"""Tool executor dispatching invocations through policy checks, scope isolation, and QueryBus."""
from __future__ import annotations

import time
import uuid
from typing import Any

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    PolicyDecisionType,
    ToolEffect,
    ToolErrorCode,
)
from automl.application.agents.contracts import (
    ApprovalRequest,
    ToolCallContext,
    ToolDefinition,
    ToolError,
    ToolInvocation,
    ToolResult,
)
from automl.application.agents.policy import (
    PolicyEvaluator,
    compute_arguments_hash,
)
from automl.application.agents.ports import AgentLedgerPort
from automl.application.agents.registry import ToolRegistry
from automl.application.agents.schemas import (
    TOOL_SCHEMAS,
    validate_arguments,
)
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


class ToolExecutor:
    """Executes agent tools with schema validation, policy authorization, scope checking, and error mapping."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy_evaluator: PolicyEvaluator | None = None,
        ledger: AgentLedgerPort | None = None,
        default_budget: AgentBudget | None = None,
    ):
        self.registry = registry
        self.policy_evaluator = policy_evaluator or PolicyEvaluator()
        self.ledger = ledger
        self.default_budget = default_budget or AgentBudget()

    def execute(
        self,
        invocation: ToolInvocation,
        budget: AgentBudget | None = None,
    ) -> ToolResult:
        """Execute a tool invocation adhering to CQRS and security rules."""
        start_time = time.perf_counter()
        req_id = invocation.idempotency_key or str(uuid.uuid4())
        active_budget = budget or self.default_budget

        # 1. Lookup tool definition
        tool_def = self.registry.get_definition(invocation.tool_name)
        if not tool_def:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Tool '{invocation.tool_name}' is not registered in the catalog",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        handler = self.registry.get_handler(invocation.tool_name)
        if not handler:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.INTERNAL_ERROR,
                    message=f"No execution handler registered for tool '{invocation.tool_name}'",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # 2. Validate input schema
        if tool_def.input_schema:
            validation_errors = validate_arguments(tool_def.input_schema, invocation.arguments)
            if validation_errors:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    error=ToolError(
                        code=ToolErrorCode.INVALID_ARGUMENT,
                        message=f"Invalid arguments for tool '{invocation.tool_name}': {'; '.join(validation_errors)}",
                        details={"validation_errors": validation_errors},
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 3. Policy evaluation
        decision = self.policy_evaluator.evaluate(
            tool_def,
            invocation.context,
            active_budget,
            invocation.arguments,
        )

        if decision.decision == PolicyDecisionType.DENY:
            err_code = ToolErrorCode.PERMISSION_DENIED
            if "Budget exceeded" in decision.reason:
                err_code = ToolErrorCode.BUDGET_EXCEEDED
            elif "Scope violation" in decision.reason:
                err_code = ToolErrorCode.SCOPE_VIOLATION

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=err_code,
                    message=decision.reason,
                    details=decision.to_dict(),
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        if decision.decision == PolicyDecisionType.REQUIRE_APPROVAL:
            # Deterministic approval request generation
            arg_hash = compute_arguments_hash(
                action=invocation.tool_name,
                actor=invocation.context.actor,
                run_id=invocation.context.run_id,
                arguments=invocation.arguments,
                policy_version=decision.policy_version,
                max_cost=tool_def.cost_estimate,
            )
            approval_id = f"appr-{arg_hash[:16]}"
            if self.ledger:
                approval_req = ApprovalRequest(
                    approval_id=approval_id,
                    action=invocation.tool_name,
                    actor=invocation.context.actor,
                    run_id=invocation.context.run_id,
                    arguments_hash=arg_hash,
                    arguments=invocation.arguments,
                    policy_version=decision.policy_version,
                    max_cost=tool_def.cost_estimate,
                    expires_at="2099-12-31T23:59:59Z",
                    status=ApprovalStatus.PENDING,
                )
                self.ledger.save_approval(approval_req)

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                approval_id=approval_id,
                error=ToolError(
                    code=ToolErrorCode.APPROVAL_REQUIRED,
                    message=decision.reason,
                    details={**decision.to_dict(), "approval_id": approval_id},
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # 4. Scope Isolation (Prevent issue #14: run cross-contamination)
        target_run_id = invocation.arguments.get("run_id")
        if target_run_id and invocation.context.run_id:
            if target_run_id != invocation.context.run_id:
                return ToolResult(
                    request_id=req_id,
                    tool_name=invocation.tool_name,
                    success=False,
                    error=ToolError(
                        code=ToolErrorCode.SCOPE_VIOLATION,
                        message=f"Scope violation: cannot access run_id '{target_run_id}' from context authorized for '{invocation.context.run_id}'",
                        correlation_id=invocation.context.correlation_id,
                    ),
                    execution_time_seconds=time.perf_counter() - start_time,
                )

        # 5. Execute Handler
        try:
            raw_data = handler(invocation.arguments, invocation.context)
            if tool_def.cost_estimate:
                active_budget.consume(tool_def.cost_estimate)

            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=True,
                data=raw_data,
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except KeyError as exc:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=str(exc).strip("'"),
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except ValueError as exc:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENT,
                    message=str(exc),
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except PermissionError as exc:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message=str(exc),
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except Exception as exc:
            return ToolResult(
                request_id=req_id,
                tool_name=invocation.tool_name,
                success=False,
                error=ToolError(
                    code=ToolErrorCode.INTERNAL_ERROR,
                    message=f"Internal error executing '{invocation.tool_name}': {str(exc)}",
                    correlation_id=invocation.context.correlation_id,
                ),
                execution_time_seconds=time.perf_counter() - start_time,
            )


def create_read_only_tool_registry(query_bus: QueryBus) -> ToolRegistry:
    """Build and populate ToolRegistry with all canonical V0.9 read-only query tools."""
    registry = ToolRegistry()

    # 1. get_dataset_profile
    def _handle_get_dataset_profile(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        profile = query_bus.dispatch(GetDatasetProfileQuery(dataset_id=args["dataset_id"]))
        if profile is None:
            raise KeyError(f"Dataset profile not found: {args['dataset_id']}")
        return {
            "dataset_id": profile.dataset_id,
            "n_rows": getattr(profile, "row_count", 0),
            "n_columns": getattr(profile, "column_count", 0),
            "target_column": getattr(profile, "target_column", ""),
            "task_type": getattr(profile, "task_type", ""),
            "columns": [c.name for c in getattr(profile, "columns", [])],
        }

    registry.register(
        ToolDefinition(
            name="get_dataset_profile",
            version="1.0.0",
            description="Retrieve metadata, dimensions, and column profile of a dataset (no raw data leakage)",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_dataset_profile"]["input"],
            output_schema=TOOL_SCHEMAS["get_dataset_profile"]["output"],
        ),
        _handle_get_dataset_profile,
    )

    # 2. list_models
    def _handle_list_models(args: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        task_type = args.get("task_type")
        models = query_bus.dispatch(ListModelsQuery(run_id=ctx.run_id, task_type=task_type))
        return {"models": list(models) if models else []}

    registry.register(
        ToolDefinition(
            name="list_models",
            version="1.0.0",
            description="List compatible ML models available for the active task type and run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_models"]["input"],
            output_schema=TOOL_SCHEMAS["list_models"]["output"],
        ),
        _handle_list_models,
    )

    # 3. list_plugins
    def _handle_list_plugins(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        plugin_type = args.get("plugin_type")
        plugins = query_bus.dispatch(ListPluginsQuery(plugin_type=plugin_type))
        normalized = []
        for p in plugins or []:
            ptype = p.plugin_type.value if hasattr(p.plugin_type, "value") else str(p.plugin_type)
            normalized.append(
                {
                    "plugin_id": getattr(p, "plugin_id", ""),
                    "name": getattr(p, "name", ""),
                    "version": getattr(p, "version", "1.0.0"),
                    "plugin_type": ptype,
                }
            )
        return {"plugins": normalized}

    registry.register(
        ToolDefinition(
            name="list_plugins",
            version="1.0.0",
            description="List installed AutoML plugins (models, metrics, preprocessors, modalities)",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_plugins"]["input"],
            output_schema=TOOL_SCHEMAS["list_plugins"]["output"],
        ),
        _handle_list_plugins,
    )

    # 4. list_experiments
    def _handle_list_experiments(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        experiments = query_bus.dispatch(ListExperimentsQuery(run_id=run_id))
        normalized = []
        for exp in experiments or []:
            status_val = exp.status.value if hasattr(exp.status, "value") else str(exp.status)
            normalized.append(
                {
                    "experiment_id": exp.id,
                    "run_id": exp.run_id,
                    "name": exp.name,
                    "model_ids": getattr(exp, "model_ids", []),
                    "metric": getattr(exp, "metric", ""),
                    "status": status_val,
                }
            )
        return {"experiments": normalized}

    registry.register(
        ToolDefinition(
            name="list_experiments",
            version="1.0.0",
            description="List all experiments configured or executed within a specific run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["list_experiments"]["input"],
            output_schema=TOOL_SCHEMAS["list_experiments"]["output"],
        ),
        _handle_list_experiments,
    )

    # 5. get_leaderboard
    def _handle_get_leaderboard(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        top_k = args.get("top_k")
        results = query_bus.dispatch(GetLeaderboardQuery(run_id=run_id)) or []
        if top_k:
            results = results[:top_k]
        normalized = []
        for r in results:
            normalized.append(
                {
                    "trial_id": getattr(r, "trial_id", ""),
                    "experiment_id": getattr(r, "experiment_id", ""),
                    "model_id": getattr(r, "model_id", ""),
                    "metric": getattr(r, "primary_metric", ""),
                    "score": getattr(r, "primary_score", 0.0),
                    "training_time_seconds": getattr(r, "training_time_seconds", 0.0),
                }
            )
        return {"leaderboard": normalized}

    registry.register(
        ToolDefinition(
            name="get_leaderboard",
            version="1.0.0",
            description="Get ranked leaderboard of evaluated trials and metric scores for a run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_leaderboard"]["input"],
            output_schema=TOOL_SCHEMAS["get_leaderboard"]["output"],
        ),
        _handle_get_leaderboard,
    )

    # 6. get_feature_evidence
    def _handle_get_feature_evidence(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        feature_id = args["feature_id"]
        evidence = query_bus.dispatch(GetFeatureEvidenceQuery(run_id=run_id, feature_id=feature_id))
        if evidence is None:
            raise KeyError(f"Feature evidence not found for '{feature_id}' in run '{run_id}'")
        if hasattr(evidence, "to_dict"):
            evidence_data = evidence.to_dict()
        else:
            evidence_data = {"feature_id": feature_id}
        return {"evidence": evidence_data}

    registry.register(
        ToolDefinition(
            name="get_feature_evidence",
            version="1.0.0",
            description="Get statistical and mathematical evidence for a specific feature in a run",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_feature_evidence"]["input"],
            output_schema=TOOL_SCHEMAS["get_feature_evidence"]["output"],
        ),
        _handle_get_feature_evidence,
    )

    # 7. get_feature_ranking
    def _handle_get_feature_ranking(args: dict[str, Any], _ctx: ToolCallContext) -> dict[str, Any]:
        run_id = args["run_id"]
        ranks = query_bus.dispatch(GetFeatureRankingQuery(run_id=run_id)) or []
        normalized = []
        for r in ranks:
            if hasattr(r, "to_dict"):
                normalized.append(r.to_dict())
            else:
                normalized.append(
                    {
                        "feature_name": getattr(r, "feature_name", ""),
                        "score": getattr(r, "score", 0.0),
                        "rank": getattr(r, "rank", 0),
                        "method": getattr(r, "method", "ensemble"),
                    }
                )
        return {"ranks": normalized}

    registry.register(
        ToolDefinition(
            name="get_feature_ranking",
            version="1.0.0",
            description="Get ranked list of features ordered by importance scores across methods",
            effect=ToolEffect.READ,
            permission_required=AgentPermission.READ_ONLY,
            cost_estimate={},
            input_schema=TOOL_SCHEMAS["get_feature_ranking"]["input"],
            output_schema=TOOL_SCHEMAS["get_feature_ranking"]["output"],
        ),
        _handle_get_feature_ranking,
    )

    return registry
