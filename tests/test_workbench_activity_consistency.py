"""
Tests for Workbench activity consistency and metric direction resolution (PR D).

Verifies:
1. Empty workspace returns empty activity feed (no fake/mock entries).
2. Anti-leakage exclusions are transparently recorded in activity feed.
3. Loss metrics (minimizing) correctly select lowest score across runs and leaderboard.
4. Failed trials are recorded as REJECT events in activity feed.
5. /api/agent/hypotheses dynamically queries real hypotheses from SQLite ledger.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import pytest

from automl.application.bootstrap import build_application
from automl.domain.experiments.trial import TrialResult
from automl.domain.agents.entities import Hypothesis
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.interfaces.web.server import AutoMLWebHandler


def _create_handler(workspace_dir: Path, path: str) -> tuple[AutoMLWebHandler, io.BytesIO]:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.headers = {"Host": "127.0.0.1"}
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.path = path
    return handler, handler.wfile


def _parse_response_json(wfile: io.BytesIO) -> tuple[int, dict]:
    raw = wfile.getvalue().decode("utf-8", errors="ignore")
    status_line = raw.split("\r\n")[0]
    status_code = int(status_line.split()[1])
    body = raw.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in raw else "{}"
    return status_code, json.loads(body)


def test_empty_workspace_activity_feed_is_empty(tmp_path: Path):
    ws_dir = tmp_path / "empty_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    assert data["total_runs"] == 0
    assert data["total_trials"] == 0
    assert data["best_score"] is None
    assert data["best_model"] is None
    assert data["activity_feed"] == []


def test_activity_feed_records_leakage_exclusion(tmp_path: Path):
    ws_dir = tmp_path / "leak_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    # Create CSV with customer_id (pseudo-identifier) and target
    csv_file = ws_dir / "data.csv"
    lines = ["customer_id,age,balance,churn"]
    for i in range(25):
        lines.append(f"CUST_{i:04d},{20 + i},{1000 + i * 50},{i % 2}")
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    ds = ws.register_dataset(name="churn_data", path=str(csv_file), target="churn")

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    feed = data["activity_feed"]
    # Check that Anti-Leakage / Pseudo-Identifier guardian recorded rejection
    reject_events = [e for e in feed if e["type"] == "REJECT"]
    assert len(reject_events) >= 1
    assert any("customer_id" in e["description"] for e in reject_events)


def test_activity_feed_and_overview_with_minimizing_loss_metric(tmp_path: Path):
    ws_dir = tmp_path / "rmse_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    csv_file = ws_dir / "regression.csv"
    csv_file.write_text("x1,x2,target\n1,2,10.5\n2,3,15.2\n3,4,19.8\n", encoding="utf-8")

    ds = ws.register_dataset(name="reg_data", path=str(csv_file), target="target")
    run = ws.create_run(dataset=ds, metric="rmse")
    exp = ws.create_experiment(run=run, name="exp_rmse", model_ids=["ridge", "svr"])

    # Save two completed trials: model A with higher RMSE, model B with lower RMSE (better)
    t1 = TrialResult(
        trial_id="t1",
        experiment_id=exp.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=1.85,
    )
    t2 = TrialResult(
        trial_id="t2",
        experiment_id=exp.id,
        model_id="svr",
        primary_metric="rmse",
        primary_score=0.42,
    )
    ws.repository.save_trial_result(t1)
    ws.repository.save_trial_result(t2)

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    # For RMSE, lower is better: best_score must be 0.42, model svr!
    assert data["best_score"] == 0.42
    assert data["best_model"] == "svr"

    feed = data["activity_feed"]
    accept_events = [e for e in feed if e["type"] == "ACCEPT" and "Best CV" in e["title"]]
    assert len(accept_events) == 1
    assert "svr" in accept_events[0]["description"]
    assert "0.42" in accept_events[0]["description"]


def test_activity_feed_records_trial_failure(tmp_path: Path):
    ws_dir = tmp_path / "fail_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    csv_file = ws_dir / "data.csv"
    csv_file.write_text("a,b,y\n1,2,0\n3,4,1\n", encoding="utf-8")

    ds = ws.register_dataset(name="toy", path=str(csv_file), target="y")
    run = ws.create_run(dataset=ds, metric="accuracy")
    exp = ws.create_experiment(run=run, name="failing_exp", model_ids=["logistic_regression"])

    failed_trial = TrialResult(
        trial_id="tfail",
        experiment_id=exp.id,
        model_id="logistic_regression",
        primary_metric="accuracy",
        primary_score=0.0,
        failure_reason="ConvergenceWarning: Max iterations reached",
    )
    ws.repository.save_trial_result(failed_trial)

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    feed = data["activity_feed"]
    reject_events = [e for e in feed if e["type"] == "REJECT" and "Trial Rejected" in e["title"]]
    assert len(reject_events) == 1
    assert "logistic_regression" in reject_events[0]["title"]
    assert "ConvergenceWarning" in reject_events[0]["description"]


def test_agent_hypotheses_dynamic_resolution_from_ledger(tmp_path: Path):
    ws_dir = tmp_path / "agent_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    # A. Empty ledger initially
    handler, wfile = _create_handler(ws_dir, "/api/agent/hypotheses")
    handler.do_GET()
    status, data = _parse_response_json(wfile)
    assert status == 200
    assert data["principle"] == "Propose ≠ Accept"
    assert data["hypotheses"] == []  # No fake mock hypotheses!

    # B. Ledger with actual hypothesis
    ledger = SqliteAgentLedger(ws_dir / "agent_ledger.db")
    hyp = Hypothesis(
        hypothesis_id="hyp_custom_99",
        run_id="run_123",
        reasoning="Log transform high skewness target",
        candidate_config={"action": "Apply np.log1p transform"},
        target_metric="rmse",
        metric_direction="minimize",
        baseline_metric=1.20,
        verification_criteria={
            "cost": "1 CV run",
            "after_score": 0.95,
            "critique": "Empirical reduction in RMSE confirmed across folds.",
        },
        status="accepted",
    )
    ledger.save_hypothesis(hyp)

    handler, wfile = _create_handler(ws_dir, "/api/agent/hypotheses?run_id=run_123")
    handler.do_GET()
    status, data = _parse_response_json(wfile)
    assert status == 200
    assert len(data["hypotheses"]) == 1
    ret = data["hypotheses"][0]
    assert ret["id"] == "hyp_custom_99"
    assert ret["statement"] == "Log transform high skewness target"
    assert ret["critic_decision"] == "PROMOTE"
    assert ret["delta"] == "-0.25000"


def test_workbench_selects_active_experimenting_run_over_completed_run(tmp_path: Path):
    from automl.domain.runs.states import RunPhase, RunStatus

    ws_dir = tmp_path / "active_run_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    csv_file = ws_dir / "data.csv"
    csv_file.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")

    ds = ws.register_dataset(name="benchmark_ds", path=str(csv_file), target="y")

    # Run 1: Completed earlier
    run1 = ws.create_run(dataset=ds, metric="accuracy")
    exp1 = ws.create_experiment(run=run1, name="exp_old", model_ids=["logistic_regression"])
    t1 = TrialResult(
        trial_id="t_done",
        experiment_id=exp1.id,
        model_id="logistic_regression",
        primary_metric="accuracy",
        primary_score=0.91,
    )
    ws.repository.save_trial_result(t1)
    run1.transition_to(RunStatus.COMPLETED)
    ws.save_run(run1)

    # Run 2: Currently active (e.g. EXPERIMENTING)
    run2 = ws.create_run(dataset=ds, metric="roc_auc")
    exp2 = ws.create_experiment(run=run2, name="exp_active", model_ids=["lightgbm"])
    t2 = TrialResult(
        trial_id="t_active",
        experiment_id=exp2.id,
        model_id="lightgbm",
        primary_metric="roc_auc",
        primary_score=0.88,
    )
    ws.repository.save_trial_result(t2)
    run2.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
    ws.save_run(run2)

    # Test /api/overview
    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    assert data["is_active"] is True
    # The active run (Run 2: lightgbm / roc_auc / 0.88) takes precedence over the completed one
    assert data["best_model"] == "lightgbm"
    assert data["best_score"] == 0.88
    assert data["best_metric"] == "roc_auc"

    # Test /api/runs
    handler_runs, wfile_runs = _create_handler(ws_dir, "/api/runs")
    handler_runs.do_GET()
    status_runs, runs_data = _parse_response_json(wfile_runs)

    assert status_runs == 200
    assert len(runs_data) == 2
    run1_dict = next(r for r in runs_data if r["id"] == run1.id)
    run2_dict = next(r for r in runs_data if r["id"] == run2.id)
    assert run1_dict["is_active"] is False
    assert run1_dict["status"] == "COMPLETED"
    assert run2_dict["is_active"] is True
    assert run2_dict["status"] == "EXPERIMENTING"


def test_activity_feed_multi_run_segregated_without_comparing_incompatible_metrics(tmp_path: Path):
    ws_dir = tmp_path / "multi_feed_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    csv_file = ws_dir / "data.csv"
    csv_file.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")

    ds1 = ws.register_dataset(name="ds_loss", path=str(csv_file), target="y", task_type="regression")
    run1 = ws.create_run(dataset=ds1, metric="rmse")
    exp1 = ws.create_experiment(run=run1, name="exp_rmse", model_ids=["ridge"])
    t1 = TrialResult(
        trial_id="t_rmse",
        experiment_id=exp1.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=0.42,
    )
    ws.repository.save_trial_result(t1)

    ds2 = ws.register_dataset(name="ds_auc", path=str(csv_file), target="y", task_type="binary_classification")
    run2 = ws.create_run(dataset=ds2, metric="roc_auc")
    exp2 = ws.create_experiment(run=run2, name="exp_auc", model_ids=["xgboost"])
    t2 = TrialResult(
        trial_id="t_auc",
        experiment_id=exp2.id,
        model_id="xgboost",
        primary_metric="roc_auc",
        primary_score=0.88,
    )
    ws.repository.save_trial_result(t2)

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    feed = data["activity_feed"]
    lb_events = [e for e in feed if e["type"] == "ACCEPT" and "Best CV Leaderboard" in e["title"]]
    # Must have 2 distinct leaderboard events (one per run), not 1 mixed event
    assert len(lb_events) == 2

    ridge_event = next(e for e in lb_events if "ridge" in e["title"])
    assert "0.42" in ridge_event["description"]
    assert "rmse" in ridge_event["description"]
    assert "ds_loss" in ridge_event["description"]

    xgb_event = next(e for e in lb_events if "xgboost" in e["title"])
    assert "0.88" in xgb_event["description"]
    assert "roc_auc" in xgb_event["description"]
    assert "ds_auc" in xgb_event["description"]


def test_domain_is_active_run_status_and_run_property():
    from automl.domain.runs.states import RunStatus, is_active_run_status
    from automl.domain.runs.run import AutoMLRun, RunConfig

    assert is_active_run_status(RunStatus.CREATED) is False
    assert is_active_run_status(RunStatus.PROFILING) is False
    assert is_active_run_status(RunStatus.PLANNING) is False
    assert is_active_run_status(RunStatus.EXPERIMENTING) is True
    assert is_active_run_status(RunStatus.OPTIMIZING) is True
    assert is_active_run_status(RunStatus.FINALIZING) is True
    assert is_active_run_status("RUNNING") is True

    assert is_active_run_status(RunStatus.COMPLETED) is False
    assert is_active_run_status(RunStatus.PAUSED) is False
    assert is_active_run_status(RunStatus.FAILED) is False
    assert is_active_run_status(RunStatus.CANCELLED) is False
    assert is_active_run_status(None) is False

    cfg = RunConfig(target="y", task_type="binary_classification")
    run_active = AutoMLRun(id="run_act", workspace_id="ws_1", dataset_id="ds_1", config=cfg, status=RunStatus.OPTIMIZING)
    assert run_active.is_active is True

    run_done = AutoMLRun(id="run_fin", workspace_id="ws_1", dataset_id="ds_1", config=cfg, status=RunStatus.COMPLETED)
    assert run_done.is_active is False


def test_workbench_frontend_active_status_and_agent_error_alert():
    utils_path = Path("src/automl/interfaces/web/static/js/utils.js")
    assert utils_path.exists()
    content = utils_path.read_text(encoding="utf-8")
    assert "export const ACTIVE_RUN_STATUSES" in content
    assert "export function isRunActive" in content

    agent_path = Path("src/automl/interfaces/web/static/js/views/agent.js")
    agent_content = agent_path.read_text(encoding="utf-8")
    assert 'alert("Agent action failed: "' in agent_content
    assert 'alert("Agent action scheduled: "' not in agent_content

    overview_path = Path("src/automl/interfaces/web/static/js/views/overview.js")
    overview_content = overview_path.read_text(encoding="utf-8")
    assert "isRunActive" in overview_content
    assert "overview.best_score != null" in overview_content


def test_agent_hypotheses_preserves_exact_zero_metrics(tmp_path: Path):
    ws_dir = tmp_path / "zero_metric_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    # Create dataset and run
    data_file = ws_dir / "data.csv"
    data_file.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")
    ds = ws.register_dataset(name="ds_zero", path=str(data_file), target="y", task_type="binary_classification")
    run = ws.create_run(ds, metric="log_loss")

    # Save hypothesis with 0.0 metric score in ledger
    from automl.domain.agents.entities import Hypothesis
    from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

    ledger_path = ws_dir / "agent_ledger.db"
    ledger = SqliteAgentLedger(ledger_path)
    hyp = Hypothesis(
        hypothesis_id="hyp_zero_001",
        run_id=run.id,
        reasoning="Test zero score preservation",
        candidate_config={"action": "evaluate_perfect_predictions"},
        target_metric="log_loss",
        metric_direction="minimize",
        baseline_metric=0.0,
        verification_criteria={"verified_metric": 0.0, "cost": "1 CV run"},
        status="accepted",
    )
    ledger.save_hypothesis(hyp)

    handler, wfile = _create_handler(ws_dir, f"/api/agent/hypotheses?run_id={run.id}")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    assert len(data["hypotheses"]) == 1
    h_data = data["hypotheses"][0]
    assert h_data["id"] == "hyp_zero_001"
    assert h_data["before_score"] == 0.0
    assert h_data["after_score"] == 0.0
    assert h_data["delta"] == "+0.00000"


def test_overview_segregates_metrics_across_different_datasets(tmp_path: Path):
    ws_dir = tmp_path / "multi_ds_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    data_file1 = ws_dir / "data1.csv"
    data_file1.write_text("x,y\n1,10\n2,20\n", encoding="utf-8")
    ds1 = ws.register_dataset(name="dataset_1", path=str(data_file1), target="y", task_type="regression")
    run1 = ws.create_run(ds1, metric="rmse")
    exp1 = ws.create_experiment(run1, name="exp1", model_ids=["ridge"])

    from automl.domain.experiments.trial import TrialResult
    tr1 = TrialResult(
        trial_id="tr_ds1",
        experiment_id=exp1.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=100.0,
    )
    ws.repository.save_trial_result(tr1)

    data_file2 = ws_dir / "data2.csv"
    data_file2.write_text("x,y\n1,1\n2,2\n", encoding="utf-8")
    ds2 = ws.register_dataset(name="dataset_2", path=str(data_file2), target="y", task_type="regression")
    run2 = ws.create_run(ds2, metric="rmse")
    exp2 = ws.create_experiment(run2, name="exp2", model_ids=["ridge"])

    tr2 = TrialResult(
        trial_id="tr_ds2",
        experiment_id=exp2.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=5.0,
    )
    ws.repository.save_trial_result(tr2)

    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    # best_by_dataset separates ds1 and ds2
    assert "best_by_dataset" in data
    assert ds1.id in data["best_by_dataset"]
    assert ds2.id in data["best_by_dataset"]
    assert data["best_by_dataset"][ds1.id]["rmse"]["score"] == 100.0
    assert data["best_by_dataset"][ds2.id]["rmse"]["score"] == 5.0

    # best_by_metric includes dataset_id
    assert "dataset_id" in data["best_by_metric"]["rmse"]


