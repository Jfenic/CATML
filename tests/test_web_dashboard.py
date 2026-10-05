from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def running_web_server(tmp_path: Path):
    # Prepare dummy dataset
    csv_path = tmp_path / "test_data.csv"
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "id": range(n),
        "feat_a": np.random.randn(n),
        "feat_b": np.random.uniform(10, 100, size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    df.to_csv(csv_path, index=False)

    workspace_dir = str(tmp_path / "test_workspace")

    # Configure handler
    AutoMLWebHandler.workspace_dir = workspace_dir

    # Bind to ephemeral port
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {"base_url": base_url, "csv_path": str(csv_path), "tmp_path": tmp_path}

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_web_dashboard_get_index(running_web_server):
    url = running_web_server["base_url"] + "/"
    with urlopen(url) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type")
        assert "text/html" in content_type
        html = resp.read().decode("utf-8")
        assert "CATML" in html
        assert "AutoML" in html

    # Verify static CSS file serving
    with urlopen(running_web_server["base_url"] + "/static/css/workbench.css") as resp:
        assert resp.status == 200
        assert "text/css" in resp.headers.get("Content-Type")

    # Verify modular JS file serving
    with urlopen(running_web_server["base_url"] + "/static/js/app.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type")


def test_web_dashboard_overview_and_plugins(running_web_server):
    base = running_web_server["base_url"]

    # Test Overview
    with urlopen(f"{base}/api/overview") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "platform" in data
        assert data["version"] == "0.7.0"
        assert "total_runs" in data

    # Test Plugins
    with urlopen(f"{base}/api/plugins") as resp:
        assert resp.status == 200
        plugins = json.loads(resp.read().decode("utf-8"))
        assert isinstance(plugins, list)
        plugin_ids = [p["plugin_id"] for p in plugins]
        assert "lightgbm" in plugin_ids or "logistic_regression" in plugin_ids


def test_web_dashboard_dataset_and_experiment_flow(running_web_server):
    base = running_web_server["base_url"]
    csv_path = running_web_server["csv_path"]

    # 1. Register Dataset via POST
    reg_payload = json.dumps({
        "name": "web_test_dataset",
        "path": csv_path,
        "target": "target",
        "task_type": "binary_classification",
    }).encode("utf-8")

    req = Request(f"{base}/api/dataset/register", data=reg_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        reg_data = json.loads(resp.read().decode("utf-8"))
        assert reg_data["status"] == "success"
        dataset_id = reg_data["dataset_id"]
        run_id = reg_data["run_id"]
        assert dataset_id is not None
        assert run_id is not None

    # 1b. List Datasets via GET
    with urlopen(f"{base}/api/datasets") as resp:
        assert resp.status == 200
        datasets = json.loads(resp.read().decode("utf-8"))
        assert len(datasets) >= 1
        assert any(d["id"] == dataset_id for d in datasets)

    # 2. Get Profile via GET
    with urlopen(f"{base}/api/dataset/profile?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        profile = json.loads(resp.read().decode("utf-8"))
        assert profile["row_count"] == 60
        assert "columns" in profile

    # 3. List Runs via GET
    with urlopen(f"{base}/api/runs") as resp:
        assert resp.status == 200
        runs = json.loads(resp.read().decode("utf-8"))
        assert len(runs) >= 1
        assert runs[0]["id"] == run_id

    # 4. Run Experiment via POST
    exp_payload = json.dumps({
        "run_id": run_id,
        "model_id": "logistic_regression",
        "name": "web_exp_lr",
    }).encode("utf-8")

    req = Request(f"{base}/api/experiment/run", data=exp_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        exp_res = json.loads(resp.read().decode("utf-8"))
        assert exp_res["status"] == "success"
        assert exp_res["succeeded"] is True
        assert exp_res["model_id"] == "logistic_regression"

    # 5. Get Leaderboard via GET
    with urlopen(f"{base}/api/leaderboard?run_id={run_id}") as resp:
        assert resp.status == 200
        lb = json.loads(resp.read().decode("utf-8"))
        assert len(lb) >= 1
        assert lb[0]["model_id"] == "logistic_regression"

    # 6. Generate Predictions / Submission via POST
    test_csv_path = running_web_server["tmp_path"] / "test_split.csv"
    pd.DataFrame({
        "id": range(10),
        "feat_a": np.random.randn(10),
        "feat_b": np.random.uniform(10, 100, size=10),
    }).to_csv(test_csv_path, index=False)
    out_csv_path = str(running_web_server["tmp_path"] / "submission_web.csv")

    pred_payload = json.dumps({
        "run_id": run_id,
        "test_dataset_path": str(test_csv_path),
        "output_path": out_csv_path,
        "predict_proba": True,
    }).encode("utf-8")

    req = Request(f"{base}/api/predict", data=pred_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        pred_res = json.loads(resp.read().decode("utf-8"))
        assert pred_res["status"] == "success"
        assert pred_res["row_count"] == 10
        assert Path(out_csv_path).exists()

    # 7. Test Plan & Explicability via GET
    with urlopen(f"{base}/api/plan?run_id={run_id}") as resp:
        assert resp.status == 200
        plan = json.loads(resp.read().decode("utf-8"))
        assert "steps" in plan
        assert len(plan["steps"]) >= 3

    # 8. Test Knowledge (V0.8 preview) via GET
    with urlopen(f"{base}/api/knowledge?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        k = json.loads(resp.read().decode("utf-8"))
        assert "fingerprint" in k
        assert "similar_datasets" in k
        assert "warm_start" in k

    # 9. Test Agent Hypotheses ("Proponer != Aceptar") via GET & POST
    with urlopen(f"{base}/api/agent/hypotheses?run_id={run_id}") as resp:
        assert resp.status == 200
        agent_data = json.loads(resp.read().decode("utf-8"))
        assert "hypotheses" in agent_data

    agent_act_payload = json.dumps({"hypothesis_id": "hyp_12", "action": "approve"}).encode("utf-8")
    req = Request(f"{base}/api/agent/action", data=agent_act_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        act_res = json.loads(resp.read().decode("utf-8"))
        assert act_res["status"] == "success"

    # 10. Test Kaggle Status via GET
    with urlopen(f"{base}/api/kaggle/status?run_id={run_id}") as resp:
        assert resp.status == 200
        k_status = json.loads(resp.read().decode("utf-8"))
        assert "competition" in k_status
        assert "checklist" in k_status

    # 11. Test Pause & Resume Run via POST
    pause_payload = json.dumps({"run_id": run_id}).encode("utf-8")
    req = Request(f"{base}/api/run/pause", data=pause_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        pause_res = json.loads(resp.read().decode("utf-8"))
        assert pause_res["state"] == "PAUSED"

    resume_payload = json.dumps({"run_id": run_id}).encode("utf-8")
    req = Request(f"{base}/api/run/resume", data=resume_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        resume_res = json.loads(resp.read().decode("utf-8"))
        assert resume_res["state"] == "RUNNING"


@pytest.mark.parametrize("with_trial", [True, False])
def test_experiments_endpoint_reads_parameters_from_persisted_trial(running_web_server, with_trial):
    from automl.application.bootstrap import build_application
    from automl.application.commands.workspace_commands import CreateExperimentCommand
    from automl.application.queries.workspace_queries import GetExperimentTrialsQuery
    from automl.domain.experiments.trial import Trial, TrialResult

    workspace, commands, queries = build_application(AutoMLWebHandler.workspace_dir)
    dataset = workspace.register_dataset("persisted", running_web_server["csv_path"], "target")
    run = workspace.create_run(dataset, metric="roc_auc")
    experiment = commands.dispatch(CreateExperimentCommand(run.id, "persisted", ["feat_a"],
                                  model_ids=["logistic_regression"]))
    if with_trial:
        workspace.repository.save_trial(Trial(id="trial_persisted", experiment_id=experiment.id,
                                             model_id="logistic_regression", parameters={"C": 0.5}))
    workspace.repository.save_trial_result(TrialResult(trial_id="trial_persisted", experiment_id=experiment.id,
                                           model_id="logistic_regression", primary_metric="roc_auc", primary_score=0.81))
    expected = {"C": 0.5} if with_trial else {}
    events = workspace.repository.list_events()
    dto = queries.dispatch(GetExperimentTrialsQuery(experiment.id))
    assert dto[0]["parameters"] == expected
    if with_trial:
        dto[0]["parameters"]["C"] = 99
        assert workspace.repository.get_trial("trial_persisted").parameters == expected
    assert workspace.repository.list_events() == events
    with urlopen(f'{running_web_server["base_url"]}/api/experiments?run_id={run.id}') as response:
        assert response.status == 200
        result = json.load(response)
    assert result[0]["trials"][0]["params"] == expected
    assert result[0]["best_score"] == 0.81


def test_web_dashboard_dataset_profile_enrichment_and_custom_features(running_web_server):
    base = running_web_server["base_url"]
    tmp_path = running_web_server["tmp_path"]

    # Create dataset with numeric, categorical, id, and target
    csv_path = tmp_path / "rich_dataset.csv"
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "row_id": range(n),
        "num_x": np.linspace(1.0, 10.0, n),
        "num_y": np.random.normal(50.0, 5.0, n),
        "cat_group": ["group_a" if i % 2 == 0 else "group_b" for i in range(n)],
        "target": [0 if i < 25 else 1 for i in range(n)],
    })
    df.to_csv(csv_path, index=False)

    # 1. Register Dataset
    reg_payload = json.dumps({
        "name": "rich_test_dataset",
        "path": str(csv_path),
        "target": "target",
        "task_type": "binary_classification",
    }).encode("utf-8")

    req = Request(f"{base}/api/dataset/register", data=reg_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        reg_data = json.loads(resp.read().decode("utf-8"))
        dataset_id = reg_data["dataset_id"]
        run_id = reg_data["run_id"]

    # 2. Query Profile endpoint
    with urlopen(f"{base}/api/dataset/profile?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        p = json.loads(resp.read().decode("utf-8"))

        # Verify preview_rows
        assert "preview_rows" in p
        assert isinstance(p["preview_rows"], list)
        assert len(p["preview_rows"]) == min(8, n)
        assert "num_x" in p["preview_rows"][0]
        assert "cat_group" in p["preview_rows"][0]

        # Verify recommendations
        assert "recommendations" in p
        assert isinstance(p["recommendations"], list)
        rec_cols = [r["column"] for r in p["recommendations"]]
        assert "row_id" in rec_cols

        # Verify column statistics
        cols_by_name = {c["name"]: c for c in p["columns"]}
        num_x_col = cols_by_name["num_x"]
        assert num_x_col["mean"] is not None
        assert num_x_col["std"] is not None
        assert num_x_col["min"] is not None
        assert num_x_col["max"] is not None
        assert num_x_col["median"] is not None
        assert num_x_col["q25"] is not None
        assert num_x_col["q75"] is not None
        assert num_x_col["skew"] is not None
        assert num_x_col["target_correlation"] is not None
        assert abs(num_x_col["target_correlation"]) > 0.5
        assert num_x_col["catml_action"] == "Keep"

        # Verify box_plot and histogram
        assert "box_plot" in num_x_col
        assert num_x_col["box_plot"]["min"] is not None
        assert len(num_x_col["box_plot"]["by_target"]) == 2
        assert "histogram" in num_x_col
        assert len(num_x_col["histogram"]["bins"]) == 10

        # Verify dataset-level correlation_matrix
        assert "correlation_matrix" in p
        assert "columns" in p["correlation_matrix"]
        assert "num_x" in p["correlation_matrix"]["columns"]

        # Verify categorical top_categories and target_rate
        cat_col = cols_by_name["cat_group"]
        assert "top_categories" in cat_col
        assert isinstance(cat_col["top_categories"], list)
        assert len(cat_col["top_categories"]) == 2
        assert cat_col["catml_action"] == "Encode"
        assert "target_rate" in cat_col["top_categories"][0]

    # 3. Create and run experiment with custom feature_names selection
    exp_payload = json.dumps({
        "run_id": run_id,
        "mode": "quick",
        "models": ["logistic_regression"],
        "feature_names": ["num_x"],
        "name": "custom_num_x_only",
    }).encode("utf-8")

    req = Request(f"{base}/api/experiment/create_and_run", data=exp_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        exp_res = json.loads(resp.read().decode("utf-8"))
        assert exp_res["status"] == "success"
        exp_id = exp_res["experiment_id"]

    # Verify that the created experiment contains exactly the selected features
    with urlopen(f"{base}/api/experiments?run_id={run_id}") as resp:
        assert resp.status == 200
        exps = json.loads(resp.read().decode("utf-8"))
        created_exp = next((e for e in exps if e["id"] == exp_id), None)
        assert created_exp is not None
        assert created_exp["feature_names"] == ["num_x"]
        assert created_exp["features_count"] == 1


def test_web_dashboard_export_model_artifact(running_web_server):
    import io
    import joblib
    from automl.artifacts.model_artifact import ModelArtifact

    base = running_web_server["base_url"]
    csv_path = running_web_server["csv_path"]

    # 1. Register dataset
    reg_payload = json.dumps({
        "path": csv_path,
        "name": "export_test_ds",
        "target": "target",
        "task_type": "binary_classification",
    }).encode("utf-8")
    req = Request(f"{base}/api/dataset/register", data=reg_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        reg_data = json.loads(resp.read().decode("utf-8"))
        run_id = reg_data["run_id"]

    # 2. Run quick experiment
    exp_payload = json.dumps({
        "run_id": run_id,
        "mode": "quick",
        "models": ["logistic_regression"],
        "name": "export_quick_exp",
    }).encode("utf-8")
    req = Request(f"{base}/api/experiment/create_and_run", data=exp_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200

    # 3. Test /api/models/export-info
    with urlopen(f"{base}/api/models/export-info?run_id={run_id}") as resp:
        assert resp.status == 200
        info = json.loads(resp.read().decode("utf-8"))
        assert info["model_id"] == "logistic_regression"
        assert "score" in info
        assert "feature_names" in info
        assert "/api/models/export" in info["download_url"]

    # 4. Test /api/models/export downloading .pkl
    with urlopen(f"{base}/api/models/export?run_id={run_id}") as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "application/octet-stream"
        disposition = resp.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert ".pkl" in disposition

        raw_bytes = resp.read()
        assert len(raw_bytes) > 0

        # Load artifact and verify standalone inference
        artifact = joblib.load(io.BytesIO(raw_bytes))
        assert isinstance(artifact, ModelArtifact)
        assert artifact.model_id == "logistic_regression"
        assert artifact.task_type in ("binary_classification", "multiclass_classification", "regression")

        # Test predict with sample data
        sample_df = pd.DataFrame({
            "feat_a": [0.1, -0.5],
            "feat_b": [25.0, 80.0],
        })
        preds = artifact.predict(sample_df)
        assert len(preds) == 2


def test_web_dashboard_temporal_endpoints(running_web_server):
    base = running_web_server["base_url"]
    csv_path = running_web_server["csv_path"]

    # 1. Register dataset
    reg_payload = json.dumps({
        "name": "temporal_test",
        "path": csv_path,
        "target": "target",
    }).encode("utf-8")
    req = Request(f"{base}/api/dataset/register", data=reg_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        dataset_id = data["dataset_id"]
        run_id = data["run_id"]

    # 2. GET /api/dataset/temporal
    with urlopen(f"{base}/api/dataset/temporal?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert "temporal_structure" in data
        assert "is_sequential" in data["temporal_structure"]

    # 3. POST /api/features/temporal
    gen_payload = json.dumps({
        "run_id": run_id,
        "dataset_id": dataset_id,
        "max_lags": 1,
        "include_lags": True,
        "include_deltas": True,
        "include_cyclical": True,
    }).encode("utf-8")
    req_gen = Request(f"{base}/api/features/temporal", data=gen_payload, headers={"Content-Type": "application/json"})
    with urlopen(req_gen) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert len(data["candidate_sets"]) >= 1


