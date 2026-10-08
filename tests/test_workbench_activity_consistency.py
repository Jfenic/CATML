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
    assert data["best_score"] == 0.0
    assert data["best_model"] == "-"
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
