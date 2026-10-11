from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from automl.application.analysis.commands import CreateStudyCommand
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand
from automl.application.recipes.commands import (
    CreateExperimentRecipeFromEvidenceCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
)
from automl.domain.experiments.trial import Experiment, ExperimentStatus
from automl.domain.tasks.questionnaire import (
    DatasetQuestionnaire,
    ErrorCostPriority,
    ExplainabilityLevel,
    LatencyConstraint,
    TemporalStructure,
)
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.interfaces.web.server import AutoMLWebHandler


# ---------------------------------------------------------------------------
# 1. Experiment Domain & Persistence Traceability Tests
# ---------------------------------------------------------------------------

def test_experiment_recipe_id_persistence(tmp_path: Path):
    db_path = tmp_path / "test_exp.db"
    repo = SQLiteExperimentRepository(str(db_path))

    exp = Experiment(
        id="exp_test_rec_1",
        run_id="run_100",
        name="Test Experiment with Recipe",
        hypothesis="Testing recipe traceability",
        feature_names=["f1", "f2"],
        model_ids=["lightgbm"],
        metric="roc_auc",
        validation_strategy="stratified_kfold",
        status=ExperimentStatus.CREATED,
        recipe_id="rcp_golden_01",
        column_transforms=[{"column": "f1", "transformer": "yeo_johnson"}],
        derived_features=[{"name": "f1_ratio", "expression": "f1/f2"}],
    )

    repo.save_experiment(exp)
    loaded = repo.get_experiment("exp_test_rec_1")

    assert loaded is not None
    assert loaded.id == "exp_test_rec_1"
    assert loaded.recipe_id == "rcp_golden_01"
    assert len(loaded.column_transforms) == 1
    assert loaded.column_transforms[0]["transformer"] == "yeo_johnson"
    assert len(loaded.derived_features) == 1
    assert loaded.derived_features[0]["name"] == "f1_ratio"


def test_create_experiment_command_with_recipe_id(tmp_path: Path):
    ws_dir = tmp_path / "app_ws"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    ds = ws.register_dataset("demo_ds", df, target="target")
    run = ws.create_run(ds)

    exp = cmd_bus.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_from_recipe",
            feature_names=["num1", "num2"],
            model_ids=["lightgbm"],
            recipe_id="rcp_custom_42",
        )
    )

    assert exp.recipe_id == "rcp_custom_42"
    retrieved = ws.get_experiment(exp.id)
    assert retrieved is not None
    assert retrieved.recipe_id == "rcp_custom_42"


def test_recipe_submission_links_recipe_id(tmp_path: Path):
    ws_dir = tmp_path / "app_recipes"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 0, 0, 1, 1, 1],
    })
    ds = ws.register_dataset("ds_recipe_test", df, target="target")
    run = ws.create_run(ds)

    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Study Test", target_column="target"))
    study_repo = ws.get_study_repository()
    analysis_run = AnalysisRun.create(study_id=study_id)
    study_repo.save_analysis_run(analysis_run)

    finding = StatisticalFinding.create(
        study_id=study_id,
        analysis_run_id=analysis_run.id,
        finding_type="skewed_distribution",
        summary="x1 is skewed",
        column_name="x1",
    )
    study_repo.save_finding(finding)

    hypothesis = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id=finding.id,
        description="Apply power transform",
        proposed_action="power_transform",
        experiment_delta={"column": "x1", "method": "yeo_johnson", "transform_kind": "yeo_johnson"},
    )
    study_repo.save_hypothesis(hypothesis)

    cand_exp = ws.create_experiment(
        run=run,
        name="Candidate",
        model_ids=["random_forest"],
        feature_names=["x1", "x2"],
        validation_strategy="holdout",
        column_transforms=[{"column": "x1", "kind": "yeo_johnson", "params": {}}],
    )
    ws.run_experiment(run, cand_exp)

    evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=cand_exp.id,
        baseline_score=0.8,
        candidate_score=0.9,
        metric="accuracy",
        accepted=True,
    )
    study_repo.save_evidence_link(evidence)

    recipe_id = cmd_bus.dispatch(
        CreateExperimentRecipeFromEvidenceCommand(
            evidence_link_id=evidence.id,
            recipe_name="Golden Pipeline Recipe",
        )
    )

    exp_ids = cmd_bus.dispatch(
        SubmitExperimentFromRecipeCommand(recipe_id=recipe_id, run_id=run.id)
    )
    assert len(exp_ids) == 1
    submitted_exp = ws.get_experiment(exp_ids[0])
    assert submitted_exp is not None
    assert submitted_exp.recipe_id == recipe_id
    assert len(submitted_exp.column_transforms) == 1


# ---------------------------------------------------------------------------
# 2. REST Endpoints Traceability & Compute Budget Enforcement Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def running_server(tmp_path: Path):
    ws_dir = tmp_path / "server_test"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    ds = ws.register_dataset("synth_ds", df, target="target")
    run = ws.create_run(ds)

    AutoMLWebHandler.workspace_dir = str(ws_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "url": f"http://127.0.0.1:{port}",
        "workspace_dir": str(ws_dir),
        "dataset_id": ds.id,
        "run_id": run.id,
        "cmd_bus": cmd_bus,
        "qry_bus": qry_bus,
    }

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_server_experiments_serialization_traceability(running_server):
    base_url = running_server["url"]
    run_id = running_server["run_id"]
    cmd_bus = running_server["cmd_bus"]

    exp = cmd_bus.dispatch(
        CreateExperimentCommand(
            run_id=run_id,
            name="exp_trace_check",
            feature_names=["feature_a", "feature_b"],
            model_ids=["lightgbm"],
            recipe_id="rcp_trace_99",
            column_transforms=[{"column": "feature_a", "transformer": "yeo_johnson"}],
            derived_features=[{"name": "ratio_ab", "expression": "feature_a/feature_b"}],
        )
    )

    req = Request(f"{base_url}/api/experiments?run_id={run_id}")
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

    matching = [e for e in data if e["id"] == exp.id]
    assert len(matching) == 1
    exp_json = matching[0]
    assert exp_json["recipe_id"] == "rcp_trace_99"
    assert len(exp_json["column_transforms"]) == 1
    assert exp_json["column_transforms"][0]["transformer"] == "yeo_johnson"
    assert len(exp_json["derived_features"]) == 1
    assert exp_json["derived_features"][0]["name"] == "ratio_ab"


def test_server_create_and_run_budget_enforcement(running_server):
    base_url = running_server["url"]
    run_id = running_server["run_id"]
    ws_dir = running_server["workspace_dir"]

    payload = {
        "run_id": run_id,
        "mode": "auto",
        "budget": "quick",
        "time_budget_seconds": 10.0,
        "models": ["random_forest"],
    }
    req = Request(
        f"{base_url}/api/experiment/create_and_run",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "success"

    # Re-read workspace from workspace_dir to verify persisted run budget
    ws, _, _ = build_application(root_dir=ws_dir)
    run = ws.get_run(run_id)
    assert run.config.time_budget_seconds == 10.0


def test_server_questionnaire_lifecycle(running_server):
    base_url = running_server["url"]
    ds_id = running_server["dataset_id"]

    # 1. Generate questionnaire suggestions
    req_gen = Request(
        f"{base_url}/api/dataset/questionnaire/generate",
        data=json.dumps({"dataset_id": ds_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_gen) as resp:
        assert resp.status == 200
        gen_data = json.loads(resp.read().decode("utf-8"))
        assert gen_data["status"] == "success"
        q_gen = gen_data["questionnaire"]
        assert q_gen["engine_source"] == "heuristic"
        assert q_gen["user_confirmed"] is False

    # 2. Save user-confirmed questionnaire
    q_gen["error_cost"] = "avoid_false_negatives"
    q_gen["latency_constraint"] = "ultra_low_realtime"
    q_gen["user_confirmed"] = True

    req_save = Request(
        f"{base_url}/api/dataset/questionnaire",
        data=json.dumps({"dataset_id": ds_id, "questionnaire": q_gen}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_save) as resp:
        assert resp.status == 200
        save_data = json.loads(resp.read().decode("utf-8"))
        assert save_data["status"] == "success"
        saved_q = save_data["questionnaire"]
        assert saved_q["error_cost"] == "avoid_false_negatives"
        assert saved_q["latency_constraint"] == "ultra_low_realtime"
        assert saved_q["user_confirmed"] is True

    # 3. GET questionnaire
    req_get = Request(f"{base_url}/api/dataset/questionnaire?dataset_id={ds_id}")
    with urlopen(req_get) as resp:
        assert resp.status == 200
        get_data = json.loads(resp.read().decode("utf-8"))
        assert get_data["dataset_id"] == ds_id
        assert get_data["error_cost"] == "avoid_false_negatives"
        assert get_data["latency_constraint"] == "ultra_low_realtime"
        assert get_data["user_confirmed"] is True
