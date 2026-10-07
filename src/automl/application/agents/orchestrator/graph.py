"""LangGraph StateGraph orchestrator for durable, crash-resilient autonomous agent cycles."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable, Optional
from uuid import uuid4

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    Hypothesis,
    OperationStatus,
    PolicyDecisionType,
)
from automl.application.agents.contracts import (
    AgentSessionState,
    ApprovalRequest,
    CandidateProposal,
    EvaluationFeedback,
    SessionStepResult,
    ToolCallContext,
    ToolInvocation,
)
from automl.application.agents.orchestrator.state_machine import (
    AgentStateMachine,
    CycleState,
    SessionStatus,
    StopReason,
)
from automl.application.agents.ports import (
    AgentApprovalStorePort,
    AgentOperationStorePort,
    AgentSessionStorePort,
)
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner
from automl.application.agents.state import (
    GraphAgentState,
    graph_state_to_session,
    session_to_graph_state,
)

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command, interrupt
    from langgraph.checkpoint.base import BaseCheckpointSaver
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment, misc]
    START = None  # type: ignore[assignment, misc]
    END = None  # type: ignore[assignment, misc]
    Command = None  # type: ignore[assignment, misc]
    interrupt = None  # type: ignore[assignment, misc]
    BaseCheckpointSaver = Any  # type: ignore[assignment, misc]
    _LANGGRAPH_AVAILABLE = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_langgraph_available() -> bool:
    """Check if LangGraph and checkpointing dependencies are installed."""
    return _LANGGRAPH_AVAILABLE


class LangGraphAgentOrchestrator:
    """Orchestrates autonomous agent cycles via LangGraph StateGraph with durable SQLite checkpointing."""

    def __init__(
        self,
        workspace: Any,
        session_store: AgentSessionStorePort,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        approval_store: Optional[AgentApprovalStorePort] = None,
        operation_store: Optional[AgentOperationStorePort] = None,
        context_builder: Optional[ContextBuilder] = None,
        planner: Optional[Planner] = None,
        critic: Optional[Critic] = None,
        feature_advisor: Optional[Any] = None,
        executor: Optional[Any] = None,
        policy_evaluator: Optional[Any] = None,
        patience: int = 3,
        permission: AgentPermission = AgentPermission.EXECUTE_WITHIN_BUDGET,
    ) -> None:
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph dependencies not found. "
                "Install them via `pip install 'catml[agents]'`."
            )

        self.workspace = workspace
        self.session_store = session_store
        self.checkpointer = checkpointer
        self.approval_store = approval_store or (
            session_store if isinstance(session_store, AgentApprovalStorePort) else None
        )
        self.operation_store = operation_store or (
            session_store if isinstance(session_store, AgentOperationStorePort) else None
        )
        self.context_builder = context_builder or ContextBuilder()
        self.planner = planner or Planner()
        self.critic = critic or Critic()
        self.feature_advisor = feature_advisor
        self.executor = executor
        self.policy_evaluator = policy_evaluator
        self.state_machine = AgentStateMachine(patience=patience)
        self.permission = permission

        self._compiled_graph = self._build_state_graph()

    @property
    def compiled_graph(self) -> Any:
        """Return the compiled LangGraph StateGraph."""
        return self._compiled_graph

    def _build_state_graph(self) -> Any:
        """Construct and compile the state graph with nodes, conditional edges, and checkpointer."""
        builder = StateGraph(GraphAgentState)

        builder.add_node("observe", self._node_observe)
        builder.add_node("propose", self._node_propose)
        builder.add_node("gate", self._node_gate)
        builder.add_node("execute", self._node_execute)
        builder.add_node("critique", self._node_critique)
        builder.add_node("check_stop", self._node_check_stop)

        builder.add_edge(START, "observe")
        builder.add_edge("observe", "propose")
        builder.add_edge("propose", "gate")

        builder.add_conditional_edges(
            "gate",
            self._route_after_gate,
            {
                "execute": "execute",
                "observe": "observe",
                END: END,
            },
        )

        builder.add_edge("execute", "critique")
        builder.add_edge("critique", "check_stop")

        builder.add_conditional_edges(
            "check_stop",
            self._route_after_check_stop,
            {
                "observe": "observe",
                END: END,
            },
        )

        return builder.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------------
    # Graph Nodes
    # -------------------------------------------------------------------------

    def _node_observe(self, state: GraphAgentState) -> dict[str, Any]:
        """OBSERVE: Build structured context without raw dataset leaking."""
        run_id = state.get("run_id", "")
        budget_dict = dict(state.get("budget", {}))
        history = list(budget_dict.get("history", []))

        context = self.context_builder.build(
            workspace=self.workspace,
            run_id=run_id,
            history=history,
            budget_status=budget_dict,
        )

        return {
            "context": context.to_dict(),
            "status": SessionStatus.ACTIVE.value,
            "updated_at": _utc_now(),
        }

    def _node_propose(self, state: GraphAgentState) -> dict[str, Any]:
        """PROPOSE: Formulate candidate hypothesis and check for duplication."""
        run_id = state.get("run_id", "")
        budget_dict = dict(state.get("budget", {}))
        past_signatures = set(budget_dict.get("past_signatures", []))

        # Check existing hypotheses from session store
        existing_hyps = self.session_store.list_hypotheses(run_id)
        for h in existing_hyps:
            sig = self.state_machine.compute_hypothesis_signature(
                h.candidate_config.get("action_type", "create_experiment"),
                h.candidate_config.get("action_payload", {}),
            )
            past_signatures.add(sig)

        # Context reconstruction
        ctx_dict = state.get("context", {})
        context = self.context_builder.build(
            workspace=self.workspace,
            run_id=run_id,
            history=list(budget_dict.get("history", [])),
            budget_status=budget_dict,
        )

        # Formulate proposal
        proposal: CandidateProposal = self.planner.analyze(context)
        prop_sig = self.state_machine.compute_hypothesis_signature(
            proposal.action_type, proposal.action_payload
        )

        # Repetition check with fallback specialist
        if self.state_machine.is_hypothesis_duplicate(prop_sig, past_signatures):
            if self.feature_advisor is not None:
                adv_proposal = self.feature_advisor.analyze(context)
                adv_sig = self.state_machine.compute_hypothesis_signature(
                    adv_proposal.action_type, adv_proposal.action_payload
                )
                if not self.state_machine.is_hypothesis_duplicate(adv_sig, past_signatures):
                    proposal = adv_proposal
                    prop_sig = adv_sig

        if self.state_machine.is_hypothesis_duplicate(prop_sig, past_signatures):
            return {
                "status": SessionStatus.STOPPED.value,
                "stop_reason": StopReason.REPEATED_HYPOTHESIS.value,
                "proposal": proposal.to_dict(),
                "updated_at": _utc_now(),
            }

        # Persist hypothesis
        now = _utc_now()
        hyp = Hypothesis(
            hypothesis_id=f"hyp-{uuid4().hex[:8]}",
            run_id=run_id,
            reasoning=proposal.hypothesis,
            candidate_config={
                "action_type": proposal.action_type,
                "action_payload": proposal.action_payload,
                "specialist": proposal.specialist_name,
            },
            target_metric=context.target_metric,
            metric_direction=context.metric_direction,
            baseline_metric=proposal.verification_plan.get("baseline_score"),
            verification_criteria=proposal.verification_plan,
            status="proposed",
            created_at=now,
        )
        self.session_store.save_hypothesis(hyp)

        past_signatures.add(prop_sig)
        budget_dict["past_signatures"] = list(past_signatures)

        return {
            "proposal": proposal.to_dict(),
            "current_hypothesis_id": hyp.hypothesis_id,
            "budget": budget_dict,
            "updated_at": now,
        }

    def _node_gate(self, state: GraphAgentState) -> dict[str, Any]:
        """GATE: Verify budget constraints, permissions, and human approvals."""
        if state.get("status") in (SessionStatus.STOPPED.value, SessionStatus.COMPLETED.value):
            return {}

        budget_dict = dict(state.get("budget", {}))
        budget_obj = AgentBudget.from_dict(budget_dict)
        proposal_dict = state.get("proposal") or {}
        estimated_cost = proposal_dict.get("estimated_cost")

        # 1. Budget check
        if self.state_machine.check_budget_exhausted(budget_obj, estimated_cost):
            return {
                "status": SessionStatus.STOPPED.value,
                "stop_reason": StopReason.BUDGET_EXHAUSTED.value,
                "updated_at": _utc_now(),
            }

        # 2. Check approval requirement
        action_type = proposal_dict.get("action_type", "")
        action_payload = proposal_dict.get("action_payload", {})
        approval_required = False

        if self.permission == AgentPermission.PROPOSE_ONLY:
            approval_required = True
        elif self.policy_evaluator is not None:
            decision = self.policy_evaluator.evaluate_action(
                action=action_type,
                actor="agent",
                arguments=action_payload,
                permission=self.permission,
                budget=budget_obj,
                estimated_cost=estimated_cost,
            )
            if decision.decision == PolicyDecisionType.REQUIRE_APPROVAL:
                approval_required = True

        if approval_required:
            if not self.approval_store:
                raise RuntimeError("Approval required by policy but no approval store is configured.")

            approval_id = state.get("pending_approval_id")
            if not approval_id:
                pending_list = self.approval_store.list_approvals(
                    run_id=state.get("run_id", ""),
                    status=ApprovalStatus.PENDING,
                )
                if pending_list:
                    approval_id = pending_list[-1].approval_id
                else:
                    approval_id = f"appr-{uuid4().hex[:8]}"
                    approval_req = ApprovalRequest(
                        approval_id=approval_id,
                        action=action_type,
                        actor="agent",
                        run_id=state.get("run_id", ""),
                        arguments_hash=self.state_machine.compute_hypothesis_signature(
                            action_type, action_payload
                        ),
                        arguments=action_payload,
                        policy_version="1.0.0",
                        max_cost=estimated_cost,
                        expires_at="",
                        status=ApprovalStatus.PENDING,
                    )
                    self.approval_store.save_approval(approval_req)

            # Interrupt execution for human approval via LangGraph interrupt
            human_decision = interrupt({
                "approval_id": approval_id,
                "action": action_type,
                "arguments": action_payload,
                "prompt": f"Action '{action_type}' requires human review and authorization.",
            })

            is_approved = False
            if isinstance(human_decision, dict):
                is_approved = bool(human_decision.get("approved", False))
            elif isinstance(human_decision, bool):
                is_approved = human_decision

            new_status = ApprovalStatus.APPROVED if is_approved else ApprovalStatus.REJECTED
            self.approval_store.update_approval_status(
                approval_id=approval_id,
                status=new_status,
                reviewer="human_reviewer",
            )

            if not is_approved:
                return {
                    "approval_status": "rejected",
                    "pending_approval_id": None,
                    "proposal": None,
                    "updated_at": _utc_now(),
                }
            else:
                return {
                    "approval_status": "approved",
                    "pending_approval_id": None,
                    "updated_at": _utc_now(),
                }

        return {"approval_status": "approved", "updated_at": _utc_now()}

    def _node_execute(self, state: GraphAgentState) -> dict[str, Any]:
        """EXECUTE: Execute proposal tools with idempotency and crash recovery."""
        if state.get("status") in (SessionStatus.STOPPED.value, SessionStatus.COMPLETED.value):
            return {}

        proposal_dict = state.get("proposal")
        if not proposal_dict:
            return {}

        run_id = state.get("run_id", "")
        session_id = state.get("session_id", "")
        action_type = proposal_dict.get("action_type", "")
        action_payload = proposal_dict.get("action_payload", {})
        budget_dict = dict(state.get("budget", {}))
        budget_obj = AgentBudget.from_dict(budget_dict)

        latest_result: dict[str, Any] = {}
        operation_id: Optional[str] = None

        if self.executor is not None:
            tool_context = ToolCallContext(
                actor="agent_orchestrator",
                workspace_path=str(getattr(self.workspace, "root_dir", "")),
                run_id=run_id,
                correlation_id=session_id,
                permission=self.permission,
            )

            # Idempotency check: verify if operation already succeeded
            idempotency_sig = self.state_machine.compute_hypothesis_signature(action_type, action_payload)
            if self.operation_store is not None:
                existing_op = self.operation_store.get_operation_by_idempotency_key(
                    run_id=run_id,
                    action=action_type,
                    idempotency_key=idempotency_sig,
                )
                if existing_op and existing_op.status == OperationStatus.SUCCEEDED:
                    latest_result = {
                        "operation_id": existing_op.operation_id,
                        "status": "reused",
                        "result_ref": existing_op.result_ref,
                    }
                    operation_id = existing_op.operation_id

            if not latest_result:
                if action_type == "create_experiment":
                    model_name = action_payload.get("model_name") or action_payload.get("model_id", "logistic_regression")
                    params = action_payload.get("parameters") or action_payload.get("hyperparameters", {})
                    args: dict[str, Any] = {
                        "run_id": run_id,
                        "model_name": model_name,
                    }
                    if params:
                        args["parameters"] = params
                    if "feature_names" in action_payload:
                        args["feature_names"] = action_payload["feature_names"]

                    inv = ToolInvocation(
                        tool_name="create_experiment",
                        arguments=args,
                        context=tool_context,
                    )
                    tool_res = self.executor.execute(inv, budget=budget_obj)
                    operation_id = tool_res.operation_id
                    if tool_res.success and isinstance(tool_res.data, dict):
                        exp_id = tool_res.data.get("experiment_id")
                        if exp_id:
                            inv_run = ToolInvocation(
                                tool_name="run_experiment",
                                arguments={"run_id": run_id, "experiment_id": exp_id},
                                context=tool_context,
                            )
                            run_res = self.executor.execute(inv_run, budget=budget_obj)
                            if run_res.success and isinstance(run_res.data, dict):
                                latest_result = dict(run_res.data)
                                latest_result["experiment_id"] = exp_id
                                if "score" not in latest_result and "metric_value" in latest_result:
                                    latest_result["score"] = latest_result["metric_value"]
                elif action_type in ("tune_hyperparameters", "optimize_experiment"):
                    inv_opt = ToolInvocation(
                        tool_name="optimize_experiment",
                        arguments={
                            "run_id": run_id,
                            "experiment_id": action_payload.get("experiment_id", ""),
                            "n_trials": action_payload.get("n_trials", 10),
                        },
                        context=tool_context,
                    )
                    tool_res = self.executor.execute(inv_opt, budget=budget_obj)
                    operation_id = tool_res.operation_id
                    if tool_res.success and isinstance(tool_res.data, dict):
                        latest_result = {
                            "score": tool_res.data.get("best_score", 0.0),
                            "trial_id": tool_res.data.get("best_trial_id", ""),
                        }
                elif action_type in ("propose_feature_set", "prioritize_feature"):
                    inv_feat = ToolInvocation(
                        tool_name="prioritize_feature",
                        arguments={
                            "run_id": run_id,
                            "feature_set_name": action_payload.get("feature_set_name", "candidate_set"),
                        },
                        context=tool_context,
                    )
                    tool_res = self.executor.execute(inv_feat, budget=budget_obj)
                    operation_id = tool_res.operation_id
        else:
            # Simulated execution fallback for unit tests and headless environments
            iteration_count = int(state.get("iteration_count", 0))
            model_id = action_payload.get("model_id", "baseline")
            ctx_dict = state.get("context", {})
            lb = ctx_dict.get("leaderboard", [])
            base_score = float(lb[0].get("score")) if lb else 0.80
            sim_score = base_score + (iteration_count + 1) * 0.02
            latest_result = {
                "model_id": model_id,
                "score": sim_score,
                "experiment_id": f"exp-graph-{iteration_count + 1}",
            }
            if hasattr(self.workspace, "get_leaderboard"):
                try:
                    wlb = self.workspace.get_leaderboard(run_id)
                    if isinstance(wlb, list):
                        wlb.insert(0, {
                            "model_id": model_id,
                            "metric": ctx_dict.get("target_metric", "roc_auc"),
                            "score": sim_score,
                            "experiment_id": f"exp-graph-{iteration_count + 1}",
                        })
                except Exception:
                    pass

        # Update budget consumption
        cost = proposal_dict.get("estimated_cost") or {"experiments": 1, "trials": 1}
        budget_obj.consume(cost)
        budget_dict.update(budget_obj.to_dict())

        return {
            "latest_result": latest_result,
            "operation_id": operation_id,
            "budget": budget_dict,
            "updated_at": _utc_now(),
        }

    def _node_critique(self, state: GraphAgentState) -> dict[str, Any]:
        """CRITIQUE: Evaluate empirical outcomes, diagnose stagnation, check target metric."""
        if state.get("status") in (SessionStatus.STOPPED.value, SessionStatus.COMPLETED.value):
            return {}

        run_id = state.get("run_id", "")
        budget_dict = dict(state.get("budget", {}))
        history = list(budget_dict.get("history", []))

        context = self.context_builder.build(
            workspace=self.workspace,
            run_id=run_id,
            history=history,
            budget_status=budget_dict,
        )

        latest_result = state.get("latest_result") or {}
        feedback: EvaluationFeedback = self.critic.analyze(context, latest_result=latest_result)

        target_score = budget_dict.get("target_score")
        status = state.get("status", SessionStatus.ACTIVE.value)
        stop_reason = state.get("stop_reason")

        # Check target reached
        if self.state_machine.check_target_reached(
            current_score=feedback.current_score,
            target_score=target_score,
            metric_direction=context.metric_direction,
        ):
            status = SessionStatus.COMPLETED.value
            stop_reason = StopReason.TARGET_REACHED.value

        # Check patience / stagnation
        consecutive_no_imp = int(budget_dict.get("consecutive_no_improvements", 0))
        if not feedback.is_improvement:
            consecutive_no_imp += 1
            if self.state_machine.check_patience_exhausted(consecutive_no_imp):
                status = SessionStatus.STOPPED.value
                stop_reason = StopReason.NO_IMPROVEMENT.value
        else:
            consecutive_no_imp = 0

        budget_dict["consecutive_no_improvements"] = consecutive_no_imp

        return {
            "feedback": feedback.to_dict(),
            "status": status,
            "stop_reason": stop_reason,
            "budget": budget_dict,
            "updated_at": _utc_now(),
        }

    def _node_check_stop(self, state: GraphAgentState) -> dict[str, Any]:
        """CHECK_STOP: Advance iteration count, synchronize session state to durable store."""
        session_id = state.get("session_id", "")
        iteration_count = int(state.get("iteration_count", 0)) + 1
        max_iterations = int(state.get("max_iterations", 10))
        status = state.get("status", SessionStatus.ACTIVE.value)
        stop_reason = state.get("stop_reason")

        if status == SessionStatus.ACTIVE.value and iteration_count >= max_iterations:
            status = SessionStatus.COMPLETED.value
            stop_reason = StopReason.MAX_ITERATIONS_REACHED.value

        checkpoint_id = f"chk-{session_id}-{iteration_count}"
        now = _utc_now()

        # Update execution history
        budget_dict = dict(state.get("budget", {}))
        history = list(budget_dict.get("history", []))
        proposal_dict = state.get("proposal") or {}
        feedback_dict = state.get("feedback") or {}

        history.append({
            "step": iteration_count,
            "hypothesis": proposal_dict.get("hypothesis", ""),
            "action_type": proposal_dict.get("action_type", ""),
            "score": feedback_dict.get("current_score"),
            "is_improvement": feedback_dict.get("is_improvement", False),
        })
        budget_dict["history"] = history

        # Update session store
        updated_state = dict(state)
        updated_state.update({
            "iteration_count": iteration_count,
            "status": status,
            "stop_reason": stop_reason,
            "session_checkpoint_id": checkpoint_id,
            "budget": budget_dict,
            "updated_at": now,
        })
        session = graph_state_to_session(updated_state)  # type: ignore[arg-type]
        self.session_store.save_session_state(session)

        return {
            "iteration_count": iteration_count,
            "status": status,
            "stop_reason": stop_reason,
            "session_checkpoint_id": checkpoint_id,
            "budget": budget_dict,
            "updated_at": now,
        }

    # -------------------------------------------------------------------------
    # Routing Logic
    # -------------------------------------------------------------------------

    def _route_after_gate(self, state: GraphAgentState) -> str:
        """Route to execute, observe (if approval rejected), or stop."""
        status = state.get("status")
        if status in (SessionStatus.STOPPED.value, SessionStatus.COMPLETED.value):
            return END
        if state.get("approval_status") == "rejected":
            return "observe"
        return "execute"

    def _route_after_check_stop(self, state: GraphAgentState) -> str:
        """Route to next cycle observe or terminate at END."""
        status = state.get("status")
        if status in (SessionStatus.COMPLETED.value, SessionStatus.STOPPED.value, SessionStatus.FAILED.value):
            return END
        return "observe"

    # -------------------------------------------------------------------------
    # Public Execution Methods
    # -------------------------------------------------------------------------

    def run(
        self,
        session_id_or_state: str | AgentSessionState,
        thread_id: Optional[str] = None,
        recursion_limit: int = 50,
    ) -> GraphAgentState:
        """Execute autonomous cycle until pause, interruption, or termination."""
        if isinstance(session_id_or_state, str):
            session = self.session_store.get_session_state(session_id_or_state)
            if not session:
                raise KeyError(f"Session '{session_id_or_state}' not found.")
        else:
            session = session_id_or_state

        t_id = thread_id or session.session_id
        config = {
            "configurable": {
                "thread_id": t_id,
                "checkpoint_ns": "",
            },
            "recursion_limit": recursion_limit,
        }

        # Check existing state in checkpointer
        existing_graph_state: Optional[dict[str, Any]] = None
        if self.checkpointer:
            tup = self.checkpointer.get_tuple(config)
            if tup and tup.checkpoint:
                existing_graph_state = dict(tup.checkpoint.get("channel_values", {}))

        # Reconcile in-flight operations if recovering from a crash
        if self.operation_store:
            self.operation_store.reconcile_operations(session.run_id)

        input_payload = existing_graph_state or session_to_graph_state(session)
        result = self._compiled_graph.invoke(input_payload, config=config)
        return GraphAgentState(**result)

    def resume(
        self,
        session_id: str,
        human_decision: bool | dict[str, Any] = True,
        thread_id: Optional[str] = None,
        recursion_limit: int = 50,
    ) -> GraphAgentState:
        """Resume execution following an approval interruption or manual pause."""
        t_id = thread_id or session_id
        config = {
            "configurable": {
                "thread_id": t_id,
                "checkpoint_ns": "",
            },
            "recursion_limit": recursion_limit,
        }

        # Reconcile in-flight operations upon resumption
        if self.operation_store:
            session = self.session_store.get_session_state(session_id)
            if session:
                self.operation_store.reconcile_operations(session.run_id)

        result = self._compiled_graph.invoke(Command(resume=human_decision), config=config)
        return GraphAgentState(**result)

    def get_latest_graph_state(self, session_id: str) -> Optional[GraphAgentState]:
        """Fetch latest persisted state from checkpointer for given session."""
        if not self.checkpointer:
            return None
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",
            }
        }
        tup = self.checkpointer.get_tuple(config)
        if not tup or not tup.checkpoint:
            return None
        return GraphAgentState(**tup.checkpoint.get("channel_values", {}))
