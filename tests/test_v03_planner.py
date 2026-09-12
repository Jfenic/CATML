from __future__ import annotations

import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    ExecuteNextExperimentCommand,
    PlanExperimentsCommand,
    PrioritizeCandidateCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import (
    GetExperimentQueueQuery,
    GetLeaderboardQuery,
    ListCandidatesQuery,
    ListExperimentsQuery,
)
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.policies.budget import BudgetPolicy
from automl.engine.priority.scheduler import ExperimentQueue, Scheduler
from automl.engine.priority.scorer import RuleBasedPriorityScorer


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "sample_churn.csv"
    df = pd.DataFrame(
        {
            "customer_id": [f"c_{i}" for i in range(1, 101)],
            "salary": [30000 + i * 500 for i in range(100)],
            "debt": [5000 + (i % 7) * 1000 for i in range(100)],
            "tenure": [i % 10 for i in range(100)],
            "churn": [1 if i % 4 == 0 else 0 for i in range(100)],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_rule_based_experiment_planner_generates_candidates(tmp_path, sample_csv):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_planner"))
    dataset = ws.register_dataset("churn_data", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    # Mark salary as user priority
    ws.prioritize_feature(dataset.id, "salary", score=1.5, run_id=run.id)

    candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
    assert len(candidates) >= 2

    names = [c.name for c in candidates]
    assert "baseline_fast" in names

    baseline = next(c for c in candidates if c.name == "baseline_fast")
    assert "logistic_regression" in baseline.model_ids
    assert "salary" in baseline.feature_names
    assert baseline.priority is not None
    assert baseline.priority.system_score > 0.0

    # User prioritized feature subset candidate should be proposed
    subset_cand = next((c for c in candidates if "prioritized" in c.tags), None)
    assert subset_cand is not None
    assert "salary" in subset_cand.feature_names


def test_priority_scorer_and_explanations(tmp_path, sample_csv):
    scorer = RuleBasedPriorityScorer()
    ws, _, _ = build_application(root_dir=str(tmp_path / "ws_scorer"))
    dataset = ws.register_dataset("churn_data", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")
    profile = ws.repository.get_dataset_profile(dataset.id)

    cand_baseline = ExperimentCandidate.create(
        run_id=run.id,
        name="cand_base",
        hypothesis="Test baseline",
        feature_names=["salary", "debt"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
        tags=["baseline"],
    )

    cand_complex = ExperimentCandidate.create(
        run_id=run.id,
        name="cand_svc",
        hypothesis="Test complex",
        feature_names=["salary", "debt"],
        model_ids=["svc"],
        metric="roc_auc",
        tags=["model_exploration"],
    )

    score_base = scorer.score(cand_baseline, run, profile, user_priority="normal")
    score_svc = scorer.score(cand_complex, run, profile, user_priority="normal")
    score_pinned = scorer.score(cand_complex, run, profile, user_priority="pinned")
    score_high = scorer.score(cand_complex, run, profile, user_priority="high")
    score_low = scorer.score(cand_complex, run, profile, user_priority="low")


    # Baseline should have lower cost penalty than SVC
    assert score_base.breakdown.cost_penalty < score_svc.breakdown.cost_penalty
    assert score_base.effective_score > score_svc.effective_score

    # Pinned should have effective score > 100
    assert score_pinned.is_pinned is True
    assert score_pinned.effective_score > 100.0

    # High > Normal > Low
    assert score_high.effective_score > score_svc.effective_score > score_low.effective_score

    # Explanations should be descriptive
    assert len(score_base.breakdown.explanation) > 10
    assert "PINNED" in score_pinned.breakdown.explanation


def test_experiment_queue_ordering_and_pinned():
    queue = ExperimentQueue()

    cand_low = ExperimentCandidate.create(
        run_id="run_1",
        name="cand_low",
        hypothesis="",
        feature_names=["a"],
        model_ids=["m1"],
        metric="roc_auc",
        priority=Priority(system_score=0.5, effective_score=0.3, level=ExperimentPriority.LOW),
    )
    cand_high = ExperimentCandidate.create(
        run_id="run_1",
        name="cand_high",
        hypothesis="",
        feature_names=["a"],
        model_ids=["m1"],
        metric="roc_auc",
        priority=Priority(system_score=0.7, effective_score=0.85, level=ExperimentPriority.HIGH),
    )
    cand_pinned = ExperimentCandidate.create(
        run_id="run_1",
        name="cand_pinned",
        hypothesis="",
        feature_names=["a"],
        model_ids=["m1"],
        metric="roc_auc",
        priority=Priority(system_score=0.4, effective_score=100.4, is_pinned=True, level=ExperimentPriority.PINNED),
    )

    queue.enqueue(cand_low)
    queue.enqueue(cand_high)
    queue.enqueue(cand_pinned)

    assert len(queue) == 3
    # Pinned first, regardless of low initial system score
    first = queue.pop_next()
    assert first.name == "cand_pinned"

    # Then highest effective score
    second = queue.pop_next()
    assert second.name == "cand_high"

    # Then lowest
    third = queue.pop_next()
    assert third.name == "cand_low"
    assert len(queue) == 0


def test_budget_policy_filtering():
    budget = BudgetPolicy(max_experiments=2, excluded_models=["svc"])
    cand1 = ExperimentCandidate.create(
        run_id="run_1",
        name="c1",
        hypothesis="",
        feature_names=["x"],
        model_ids=["logistic_regression"],
        metric="accuracy",
    )
    cand_excluded = ExperimentCandidate.create(
        run_id="run_1",
        name="c_svc",
        hypothesis="",
        feature_names=["x"],
        model_ids=["svc"],
        metric="accuracy",
    )
    cand3 = ExperimentCandidate.create(
        run_id="run_1",
        name="c3",
        hypothesis="",
        feature_names=["x"],
        model_ids=["random_forest"],
        metric="accuracy",
    )

    assert budget.is_candidate_eligible(cand1, executed_experiments=0) is True
    assert budget.is_candidate_eligible(cand_excluded, executed_experiments=0) is False

    # Max experiments limit reached
    assert budget.is_candidate_eligible(cand3, executed_experiments=2) is False

    filtered = budget.filter_candidates([cand1, cand_excluded, cand3])
    assert len(filtered) == 2
    assert [c.name for c in filtered] == ["c1", "c3"]


def test_cqrs_planning_and_scheduled_execution(tmp_path, sample_csv):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_cqrs"))
    dataset = ws.register_dataset("customers", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    # 1. Plan experiments
    candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
    assert len(candidates) >= 2

    # 2. Query queue
    queued = qry.dispatch(ListCandidatesQuery(run_id=run.id))
    assert len(queued) == len(candidates)

    # 3. Prioritize candidate to PINNED
    cand_to_pin = queued[-1]["id"]
    new_prio = cmd.dispatch(
        PrioritizeCandidateCommand(
            run_id=run.id,
            candidate_id=cand_to_pin,
            priority="pinned",
        )
    )
    assert new_prio.is_pinned is True

    # 4. Execute next experiment
    exp, results = cmd.dispatch(ExecuteNextExperimentCommand(run_id=run.id))
    assert exp is not None
    assert exp.name == queued[-1]["name"]  # Pinned candidate was popped first!
    assert len(results) >= 1
    assert results[0].succeeded is True

    # 5. Run rest of scheduled experiments with budget limit
    executed_batch = cmd.dispatch(
        RunScheduledExperimentsCommand(
            run_id=run.id,
            max_experiments=1,  # execute only 1 more
        )
    )
    assert len(executed_batch) == 1

    # 6. Check leaderboard and repository list_experiments
    leaderboard = qry.dispatch(GetLeaderboardQuery(run_id=run.id))
    assert len(leaderboard) >= 2

    stored_experiments = qry.dispatch(ListExperimentsQuery(run_id=run.id))
    assert len(stored_experiments) == 2
