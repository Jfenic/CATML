"""Session manager for creating, checkpointing, and deterministically stepping autonomous agent sessions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import uuid4

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    Hypothesis,
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
    AgentSessionStorePort,
)
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentSessionManager:
    """Orchestrates autonomous session state, checkpoint persistence, and deterministic step transitions."""

    def __init__(
        self,
        workspace: Any,
        session_store: AgentSessionStorePort,
        approval_store: AgentApprovalStorePort | None = None,
        context_builder: ContextBuilder | None = None,
        planner: Planner | None = None,
        critic: Critic | None = None,
        feature_advisor: Any | None = None,
        executor: Any | None = None,
        policy_evaluator: Any | None = None,
        patience: int = 3,
        permission: AgentPermission = AgentPermission.EXECUTE_WITHIN_BUDGET,
    ) -> None:
        self.workspace = workspace
        self.session_store = session_store
        self.approval_store = approval_store or (
            session_store if isinstance(session_store, AgentApprovalStorePort) else None
        )
        self.context_builder = context_builder or ContextBuilder()
        self.planner = planner or Planner()
        self.critic = critic or Critic()
        self.feature_advisor = feature_advisor
        self.executor = executor
        self.policy_evaluator = policy_evaluator
        self.state_machine = AgentStateMachine(patience=patience)
        self.permission = permission

    def create_session(
        self,
        run_id: str,
        goal: str = "",
        max_iterations: int = 10,
        budget: AgentBudget | dict[str, Any] | None = None,
        target_score: float | None = None,
        patience: int = 3,
    ) -> AgentSessionState:
        """Create and persist a new autonomous agent session."""
        session_id = f"sess-{uuid4().hex[:8]}"
        budget_dict: dict[str, Any]
        if isinstance(budget, AgentBudget):
            budget_dict = budget.to_dict()
        elif isinstance(budget, dict):
            budget_dict = budget
        else:
            budget_dict = AgentBudget().to_dict()

        budget_dict["target_score"] = target_score
        budget_dict["patience"] = patience
        budget_dict["consecutive_no_improvements"] = 0
        budget_dict["past_signatures"] = []
        budget_dict["history"] = []

        now = _utc_now()
        session = AgentSessionState(
            session_id=session_id,
            run_id=run_id,
            goal=goal or f"Optimize model performance for run {run_id}",
            version="1.0.0",
            status=SessionStatus.ACTIVE.value,
            iteration_count=0,
            max_iterations=max_iterations,
            budget=budget_dict,
            stop_reason=None,
            checkpoint_id=f"chk-{session_id}-0",
            created_at=now,
            updated_at=now,
        )

        self.session_store.save_session_state(session)
        return session

    def get_session(self, session_id: str) -> AgentSessionState | None:
        """Retrieve the persisted state of a session."""
        return self.session_store.get_session_state(session_id)

    def list_sessions(self, run_id: str | None = None) -> list[AgentSessionState]:
        """List active and historical sessions, optionally filtered by run_id."""
        if hasattr(self.session_store, "list_sessions"):
            return getattr(self.session_store, "list_sessions")(run_id)
        # Fallback if list_sessions not directly on port
        session = self.get_session(run_id or "")
        return [session] if session else []

    def pause_session(self, session_id: str) -> AgentSessionState:
        """Pause an active session."""
        session = self._require_session(session_id)
        if session.status in (SessionStatus.COMPLETED.value, SessionStatus.STOPPED.value, SessionStatus.FAILED.value):
            return session
        session.status = SessionStatus.PAUSED.value
        session.updated_at = _utc_now()
        self.session_store.save_session_state(session)
        return session

    def stop_session(self, session_id: str, reason: str = StopReason.MANUALLY_STOPPED.value) -> AgentSessionState:
        """Force-stop a session with a designated reason."""
        session = self._require_session(session_id)
        session.status = SessionStatus.STOPPED.value
        session.stop_reason = reason
        session.updated_at = _utc_now()
        self.session_store.save_session_state(session)
        return session

    def step(self, session_id: str) -> SessionStepResult:
        """Execute exactly one deterministic cycle step of the autonomous agent."""
        session = self._require_session(session_id)

        # 1. Terminal or paused check
        if session.status in (SessionStatus.COMPLETED.value, SessionStatus.STOPPED.value, SessionStatus.FAILED.value):
            return SessionStepResult(
                session_id=session_id,
                step_number=session.iteration_count,
                state=session.status,
                stop_reason=session.stop_reason,
                checkpoint_id=session.checkpoint_id,
                created_at=_utc_now(),
            )

        if session.status == SessionStatus.PAUSED.value:
            return SessionStepResult(
                session_id=session_id,
                step_number=session.iteration_count,
                state=SessionStatus.PAUSED.value,
                checkpoint_id=session.checkpoint_id,
                created_at=_utc_now(),
            )

        # Check if budget is already exhausted prior to this step
        budget_obj = AgentBudget.from_dict(session.budget)
        if self.state_machine.check_budget_exhausted(budget_obj):
            session.status = SessionStatus.STOPPED.value
            session.stop_reason = StopReason.BUDGET_EXHAUSTED.value
            session.updated_at = _utc_now()
            self.session_store.save_session_state(session)
            return SessionStepResult(
                session_id=session_id,
                step_number=session.iteration_count,
                state=SessionStatus.STOPPED.value,
                stop_reason=StopReason.BUDGET_EXHAUSTED.value,
                checkpoint_id=session.checkpoint_id,
                created_at=_utc_now(),
            )

        # 2. Check pending approval if waiting
        proposal_to_execute: CandidateProposal | None = None
        if session.status == SessionStatus.WAITING_APPROVAL.value:
            if not session.pending_approval_id or not self.approval_store:
                session.status = SessionStatus.ACTIVE.value
            else:
                approval = self.approval_store.get_approval(session.pending_approval_id)
                if approval is None or approval.status == ApprovalStatus.PENDING:
                    return SessionStepResult(
                        session_id=session_id,
                        step_number=session.iteration_count,
                        state=SessionStatus.WAITING_APPROVAL.value,
                        checkpoint_id=session.checkpoint_id,
                        created_at=_utc_now(),
                    )
                elif approval.status == ApprovalStatus.REJECTED:
                    # User rejected the proposal
                    session.status = SessionStatus.ACTIVE.value
                    session.pending_approval_id = None
                    session.updated_at = _utc_now()
                    self.session_store.save_session_state(session)
                    # Proceed to generate alternative proposal below
                elif approval.status == ApprovalStatus.APPROVED:
                    # Approved: resume execution
                    session.status = SessionStatus.ACTIVE.value
                    session.pending_approval_id = None
                    # Reconstruct proposal from approved arguments
                    proposal_to_execute = CandidateProposal(
                        proposal_id=f"prop-resumed-{approval.approval_id}",
                        specialist_name="planner",
                        run_id=session.run_id,
                        hypothesis=f"Approved action: {approval.action}",
                        action_type=approval.action,
                        action_payload=approval.arguments,
                    )

        # 3. OBSERVE (Context Building)
        history = session.budget.get("history", [])
        context = self.context_builder.build(
            workspace=self.workspace,
            run_id=session.run_id,
            history=history,
            budget_status=session.budget,
        )

        # 4. PROPOSE (if not resuming approved proposal)
        proposal = proposal_to_execute
        if proposal is None:
            # Query specialist
            proposal = self.planner.analyze(context)

            # Check hypothesis repetition
            past_sigs = set(session.budget.get("past_signatures", []))
            # Also read existing hypotheses from store
            existing_hyps = self.session_store.list_hypotheses(session.run_id)
            for h in existing_hyps:
                sig = self.state_machine.compute_hypothesis_signature(
                    h.candidate_config.get("action_type", "create_experiment"),
                    h.candidate_config.get("action_payload", {}),
                )
                past_sigs.add(sig)

            prop_sig = self.state_machine.compute_hypothesis_signature(
                proposal.action_type, proposal.action_payload
            )

            if self.state_machine.is_hypothesis_duplicate(prop_sig, past_sigs):
                # Attempt alternative specialist (FeatureAdvisor) if available
                if self.feature_advisor is not None:
                    adv_proposal = self.feature_advisor.analyze(context)
                    adv_sig = self.state_machine.compute_hypothesis_signature(
                        adv_proposal.action_type, adv_proposal.action_payload
                    )
                    if not self.state_machine.is_hypothesis_duplicate(adv_sig, past_sigs):
                        proposal = adv_proposal
                        prop_sig = adv_sig

            if self.state_machine.is_hypothesis_duplicate(prop_sig, past_sigs):
                # Repetition detected: stop loop
                session.status = SessionStatus.STOPPED.value
                session.stop_reason = StopReason.REPEATED_HYPOTHESIS.value
                session.updated_at = _utc_now()
                self.session_store.save_session_state(session)
                return SessionStepResult(
                    session_id=session_id,
                    step_number=session.iteration_count + 1,
                    state=SessionStatus.STOPPED.value,
                    proposal=proposal,
                    stop_reason=StopReason.REPEATED_HYPOTHESIS.value,
                    checkpoint_id=session.checkpoint_id,
                    created_at=_utc_now(),
                )

            # Persist proposed hypothesis
            hyp = Hypothesis(
                hypothesis_id=f"hyp-{uuid4().hex[:8]}",
                run_id=session.run_id,
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
                created_at=_utc_now(),
            )
            self.session_store.save_hypothesis(hyp)
            session.current_hypothesis_id = hyp.hypothesis_id
            past_sigs.add(prop_sig)
            session.budget["past_signatures"] = list(past_sigs)

        # 5. GATE (Policy & Budget Check)
        budget_obj = AgentBudget.from_dict(session.budget)
        if self.state_machine.check_budget_exhausted(budget_obj, proposal.estimated_cost):
            session.status = SessionStatus.STOPPED.value
            session.stop_reason = StopReason.BUDGET_EXHAUSTED.value
            session.updated_at = _utc_now()
            self.session_store.save_session_state(session)
            return SessionStepResult(
                session_id=session_id,
                step_number=session.iteration_count + 1,
                state=SessionStatus.STOPPED.value,
                proposal=proposal,
                stop_reason=StopReason.BUDGET_EXHAUSTED.value,
                checkpoint_id=session.checkpoint_id,
                created_at=_utc_now(),
            )

        # Evaluate policy & approval requirement (only when not resuming approved proposal)
        approval_required = False
        if proposal_to_execute is None:
            if self.permission == AgentPermission.PROPOSE_ONLY:
                approval_required = True
            elif self.policy_evaluator is not None:
                decision = self.policy_evaluator.evaluate_action(
                    action=proposal.action_type,
                    actor="agent",
                    arguments=proposal.action_payload,
                    permission=self.permission,
                    budget=budget_obj,
                    estimated_cost=proposal.estimated_cost,
                )
                if decision.decision == PolicyDecisionType.REQUIRE_APPROVAL:
                    approval_required = True

        if approval_required:
            if not self.approval_store:
                raise RuntimeError("Approval required by policy but no approval store is configured.")
            approval_id = f"appr-{uuid4().hex[:8]}"
            approval_req = ApprovalRequest(
                approval_id=approval_id,
                action=proposal.action_type,
                actor="agent",
                run_id=session.run_id,
                arguments_hash=self.state_machine.compute_hypothesis_signature(
                    proposal.action_type, proposal.action_payload
                ),
                arguments=proposal.action_payload,
                policy_version="1.0.0",
                max_cost=proposal.estimated_cost,
                expires_at="",
                status=ApprovalStatus.PENDING,
            )
            self.approval_store.save_approval(approval_req)
            session.status = SessionStatus.WAITING_APPROVAL.value
            session.pending_approval_id = approval_id
            session.updated_at = _utc_now()
            self.session_store.save_session_state(session)
            return SessionStepResult(
                session_id=session_id,
                step_number=session.iteration_count + 1,
                state=SessionStatus.WAITING_APPROVAL.value,
                proposal=proposal,
                checkpoint_id=session.checkpoint_id,
                created_at=_utc_now(),
            )

        # 6. EXECUTE
        latest_result: dict[str, Any] = {}
        operation_id: str | None = None
        if self.executor is not None:
            tool_context = ToolCallContext(
                actor="agent_orchestrator",
                workspace_path=str(getattr(self.workspace, "root_dir", "")),
                run_id=session.run_id,
                correlation_id=session_id,
                permission=self.permission,
            )

            if proposal.action_type == "create_experiment":
                inv = ToolInvocation(
                    tool_name="create_experiment",
                    arguments={
                        "run_id": session.run_id,
                        "model_id": proposal.action_payload.get("model_id"),
                        "parameters": proposal.action_payload.get("hyperparameters", {}),
                    },
                    context=tool_context,
                )
                tool_res = self.executor.execute(inv, budget=budget_obj)
                operation_id = tool_res.operation_id
                if tool_res.success and isinstance(tool_res.data, dict):
                    exp_id = tool_res.data.get("experiment_id")
                    if exp_id:
                        # Follow up with run_experiment
                        inv_run = ToolInvocation(
                            tool_name="run_experiment",
                            arguments={"run_id": session.run_id, "experiment_id": exp_id},
                            context=tool_context,
                        )
                        run_res = self.executor.execute(inv_run, budget=budget_obj)
                        if run_res.success and isinstance(run_res.data, dict):
                            latest_result = run_res.data
                            latest_result["experiment_id"] = exp_id
            elif proposal.action_type in ("tune_hyperparameters", "optimize_experiment"):
                inv_opt = ToolInvocation(
                    tool_name="optimize_experiment",
                    arguments={
                        "run_id": session.run_id,
                        "experiment_id": proposal.action_payload.get("experiment_id", ""),
                        "n_trials": proposal.action_payload.get("n_trials", 10),
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
            elif proposal.action_type in ("propose_feature_set", "prioritize_feature"):
                inv_feat = ToolInvocation(
                    tool_name="prioritize_feature",
                    arguments={
                        "run_id": session.run_id,
                        "feature_set_name": proposal.action_payload.get("feature_set_name", "candidate_set"),
                    },
                    context=tool_context,
                )
                tool_res = self.executor.execute(inv_feat, budget=budget_obj)
                operation_id = tool_res.operation_id
        else:
            # Simulated execution fallback for unit tests and dry runs
            model_id = proposal.action_payload.get("model_id", "baseline")
            existing_score = context.leaderboard[0].get("score") if context.leaderboard else None
            if existing_score is not None:
                sim_score = float(existing_score)
            else:
                sim_score = 0.80 + (session.iteration_count + 1) * 0.02
            latest_result = {
                "model_id": model_id,
                "score": sim_score,
                "experiment_id": f"exp-sim-{session.iteration_count + 1}",
            }
            if hasattr(self.workspace, "get_leaderboard"):
                try:
                    lb = self.workspace.get_leaderboard(session.run_id)
                    if isinstance(lb, list):
                        lb.insert(0, {
                            "model_id": model_id,
                            "metric": context.target_metric,
                            "score": sim_score,
                            "experiment_id": f"exp-sim-{session.iteration_count + 1}",
                        })
                except Exception:
                    pass

        # Update budget consumption
        cost = proposal.estimated_cost or {"experiments": 1, "trials": 1}
        budget_obj.consume(cost)
        session.budget.update(budget_obj.to_dict())

        # 7. CRITIQUE
        feedback = self.critic.analyze(context, latest_result=latest_result)

        # Check target metric threshold
        target_score = session.budget.get("target_score")
        if self.state_machine.check_target_reached(
            current_score=feedback.current_score,
            target_score=target_score,
            metric_direction=context.metric_direction,
        ):
            session.status = SessionStatus.COMPLETED.value
            session.stop_reason = StopReason.TARGET_REACHED.value

        # Check patience / no improvement
        consecutive_no_imp = session.budget.get("consecutive_no_improvements", 0)
        if not feedback.is_improvement:
            consecutive_no_imp += 1
            if self.state_machine.check_patience_exhausted(consecutive_no_imp):
                session.status = SessionStatus.STOPPED.value
                session.stop_reason = StopReason.NO_IMPROVEMENT.value
        else:
            consecutive_no_imp = 0
        session.budget["consecutive_no_improvements"] = consecutive_no_imp

        # 8. CHECK_STOP & CHECKPOINT
        session.iteration_count += 1
        if (
            session.status == SessionStatus.ACTIVE.value
            and session.iteration_count >= session.max_iterations
        ):
            session.status = SessionStatus.COMPLETED.value
            session.stop_reason = StopReason.MAX_ITERATIONS_REACHED.value

        session.checkpoint_id = f"chk-{session_id}-{session.iteration_count}"
        session.updated_at = _utc_now()

        # Update history
        history.append({
            "step": session.iteration_count,
            "hypothesis": proposal.hypothesis,
            "action_type": proposal.action_type,
            "score": feedback.current_score,
            "is_improvement": feedback.is_improvement,
        })
        session.budget["history"] = history

        self.session_store.save_session_state(session)

        return SessionStepResult(
            session_id=session_id,
            step_number=session.iteration_count,
            state=session.status,
            action_taken=proposal.action_type,
            proposal=proposal,
            feedback=feedback,
            operation_id=operation_id,
            stop_reason=session.stop_reason,
            checkpoint_id=session.checkpoint_id,
            created_at=_utc_now(),
        )

    def resume(self, session_id: str, max_steps: int | None = None) -> list[SessionStepResult]:
        """Step session repeatedly until completion, pause, waiting for approval, or max_steps reached."""
        results: list[SessionStepResult] = []
        steps_taken = 0

        while True:
            if max_steps is not None and steps_taken >= max_steps:
                break
            result = self.step(session_id)
            results.append(result)
            steps_taken += 1

            if result.state in (
                SessionStatus.COMPLETED.value,
                SessionStatus.STOPPED.value,
                SessionStatus.FAILED.value,
                SessionStatus.WAITING_APPROVAL.value,
                SessionStatus.PAUSED.value,
            ):
                break

        return results

    def _require_session(self, session_id: str) -> AgentSessionState:
        session = self.session_store.get_session_state(session_id)
        if session is None:
            raise KeyError(f"Agent session '{session_id}' not found.")
        return session
