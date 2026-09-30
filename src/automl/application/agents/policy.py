"""Policy evaluation engine, permission enforcement, budget checks, and cryptographic hashing."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    PolicyDecisionType,
    ToolEffect,
)
from automl.application.agents.contracts import (
    PolicyDecision,
    ToolCallContext,
    ToolDefinition,
)


def compute_arguments_hash(
    action: str,
    actor: str,
    run_id: str,
    arguments: dict[str, Any],
    policy_version: str,
    max_cost: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash of action, actor, run, arguments, and cost limits."""
    payload = {
        "action": action,
        "actor": actor,
        "run_id": run_id,
        "arguments": arguments,
        "policy_version": policy_version,
        "max_cost": max_cost or {},
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass
class AgentPolicyConfig:
    """Finite, non-null configuration defaults for agent operation rules."""
    policy_version: str = "1.0.0"
    default_mode: AgentPermission = AgentPermission.PROPOSE_ONLY
    require_approval_for_mutations: bool = True
    allowed_models: list[str] | None = None
    denied_models: list[str] = field(default_factory=list)
    max_budget: AgentBudget = field(default_factory=AgentBudget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "default_mode": self.default_mode.value,
            "require_approval_for_mutations": self.require_approval_for_mutations,
            "allowed_models": self.allowed_models,
            "denied_models": self.denied_models,
            "max_budget": self.max_budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPolicyConfig:
        mode = AgentPermission(data.get("default_mode", AgentPermission.PROPOSE_ONLY.value))
        budget_data = data.get("max_budget", {})
        budget = AgentBudget.from_dict(budget_data) if budget_data else AgentBudget()
        return cls(
            policy_version=data.get("policy_version", "1.0.0"),
            default_mode=mode,
            require_approval_for_mutations=data.get("require_approval_for_mutations", True),
            allowed_models=data.get("allowed_models"),
            denied_models=data.get("denied_models", []),
            max_budget=budget,
        )


class PolicyEvaluator:
    """Single pure rule evaluator deciding ALLOW, DENY, or REQUIRE_APPROVAL for agent tool calls."""

    def __init__(self, config: AgentPolicyConfig | None = None):
        self.config = config or AgentPolicyConfig()

    def evaluate(
        self,
        tool: ToolDefinition,
        context: ToolCallContext,
        budget: AgentBudget,
        arguments: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate an intended tool call against security, permissions, scope, and budgets."""
        # 1. Scope and actor validation
        if not context.run_id or not context.run_id.strip():
            return PolicyDecision(
                decision=PolicyDecisionType.DENY,
                reason="Scope violation: run_id is missing or empty",
                policy_version=self.config.policy_version,
            )

        if not context.actor or not context.actor.strip():
            return PolicyDecision(
                decision=PolicyDecisionType.DENY,
                reason="Authentication violation: actor identifier is missing or empty",
                policy_version=self.config.policy_version,
            )

        # 2. Model restriction checks
        target_model = arguments.get("model_name") or arguments.get("model_id")
        if target_model and isinstance(target_model, str):
            if target_model in self.config.denied_models:
                return PolicyDecision(
                    decision=PolicyDecisionType.DENY,
                    reason=f"Model '{target_model}' is explicitly prohibited by policy",
                    policy_version=self.config.policy_version,
                )
            if self.config.allowed_models is not None and target_model not in self.config.allowed_models:
                return PolicyDecision(
                    decision=PolicyDecisionType.DENY,
                    reason=f"Model '{target_model}' is not in allowed models whitelist",
                    policy_version=self.config.policy_version,
                )

        # 3. Permission level checks
        actor_perm = context.permission
        if actor_perm == AgentPermission.READ_ONLY and tool.effect in (ToolEffect.PROPOSE, ToolEffect.MUTATE):
            return PolicyDecision(
                decision=PolicyDecisionType.DENY,
                reason=f"Permission denied: actor with '{actor_perm.value}' cannot execute tool with effect '{tool.effect.value}'",
                policy_version=self.config.policy_version,
            )

        if actor_perm == AgentPermission.PROPOSE_ONLY and tool.effect == ToolEffect.MUTATE:
            return PolicyDecision(
                decision=PolicyDecisionType.REQUIRE_APPROVAL,
                reason="Human approval required: actor is in propose_only mode and tool mutates state",
                approval_required=True,
                estimated_cost=tool.cost_estimate,
                policy_version=self.config.policy_version,
            )

        # 4. Budget limits check
        cost = tool.cost_estimate
        if cost:
            can_afford, budget_error = budget.can_afford(cost)
            if not can_afford:
                return PolicyDecision(
                    decision=PolicyDecisionType.DENY,
                    reason=f"Budget exceeded: {budget_error}",
                    effective_limits=budget.to_dict(),
                    policy_version=self.config.policy_version,
                )

        # 5. Mutation approval requirement
        if tool.effect == ToolEffect.MUTATE and self.config.require_approval_for_mutations:
            if actor_perm != AgentPermission.ADMIN:
                return PolicyDecision(
                    decision=PolicyDecisionType.REQUIRE_APPROVAL,
                    reason="Human approval required for mutating state operation",
                    approval_required=True,
                    estimated_cost=tool.cost_estimate,
                    policy_version=self.config.policy_version,
                )

        # 6. Approved / Allowed
        return PolicyDecision(
            decision=PolicyDecisionType.ALLOW,
            reason="Tool invocation authorized by policy",
            effective_limits=budget.to_dict(),
            policy_version=self.config.policy_version,
            estimated_cost=tool.cost_estimate,
        )
