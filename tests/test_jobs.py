"""Durability, cooperative control, and CLI/HTTP parity for background jobs."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.job_commands import SubmitJobCommand, ControlJobCommand
from automl.application.commands.workspace_commands import CreateExperimentCommand
from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery
from automl.domain.jobs.job import Job, JobStatus, ACTIVE_STATUSES
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository
from automl.infrastructure.jobs.worker import JobWorker
from automl.interfaces.cli.main import main
from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def app(tmp_path):
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"id": range(40), "x": rng.normal(size=40), "target": [0, 1] * 20})
    path = tmp_path / "train.csv"
    data.to_csv(path, index=False)
    test_path = tmp_path / "test.csv"
    data.drop(columns="target").iloc[:5].to_csv(test_path, index=False)
    workspace, commands, queries = build_application(str(tmp_path / "workspace"))
    dataset = workspace.register_dataset("data", path, "target")
    run = workspace.create_run(dataset, metric="roc_auc")
    return workspace, commands, queries, run, test_path


def submit(app, key="request", **payload):
    _, commands, _, run, _ = app
    return commands.dispatch(SubmitJobCommand("experiment", run.id,
                             payload or {"model_ids": ["logistic_regression", "random_forest"]}, key))


def repository(app):
    return SQLiteJobRepository(app[0].repository.db_path)


def test_submit_is_idempotent_and_queries_are_snapshots(app):
    workspace, commands, queries, run, _ = app
    job_id = submit(app)
    assert submit(app) == job_id
    assert not workspace.repository.list_experiments(run.id)
    before = repository(app).get(job_id).to_dict()
    dto = queries.dispatch(GetJobQuery(job_id))
    dto["payload"]["model_ids"].clear()
    assert queries.dispatch(GetJobQuery(job_id)) == before
    assert len(queries.dispatch(ListJobsQuery(run.id))) == 1
    with pytest.raises(ValueError, match="different arguments"):
        submit(app, model_ids=["random_forest"])
    _, _, reloaded = build_application(str(workspace.root_dir))
    assert reloaded.dispatch(GetJobQuery(job_id)) == before


def test_atomic_claim_serializes_workers_and_idempotency(app):
    job_id = submit(app)
    second = submit(app, "second")
    repo = repository(app)
    with ThreadPoolExecutor(max_workers=6) as executor:
        claimed = list(executor.map(lambda _: repo.claim(), range(6)))
    assert [job.id for job in claimed if job] == [job_id]
    assert repo.get(job_id).attempt == 1
    repo.change(job_id, ACTIVE_STATUSES, status=JobStatus.COMPLETED)
    assert repo.claim().id == second
    with ThreadPoolExecutor(max_workers=5) as executor:
        ids = list(executor.map(lambda _: submit(app, "third"), range(5)))
    assert len(set(ids)) == 1


@pytest.mark.parametrize("action,target", [("pause", "paused"), ("cancel", "cancelled")])
def test_control_before_execution(app, action, target):
    _, commands, queries, _, _ = app
    job_id = submit(app)
    assert commands.dispatch(ControlJobCommand(job_id, action)) == job_id
    assert queries.dispatch(GetJobQuery(job_id))["status"] == target
    worker = JobWorker(str(app[0].root_dir)).start(background=False)
    try:
        assert not worker.run_once()
    finally:
        worker.close()


def test_invalid_state_transitions_and_unknown_job(app):
    _, commands, queries, _, _ = app
    job_id = submit(app)
    for action in ("resume", "retry", "unknown"):
        with pytest.raises(ValueError):
            commands.dispatch(ControlJobCommand(job_id, action))
    with pytest.raises(KeyError):
        queries.dispatch(GetJobQuery("missing"))
    with pytest.raises(KeyError):
        repository(app).change("missing", ACTIVE_STATUSES, message="test")


@pytest.mark.parametrize("operation,payload,key", [
    ("unknown", {}, "k"), ("experiment", {}, "k"),
    ("experiment", {"model_ids": ["missing"]}, "k"),
    ("experiment", {"model_ids": ["random_forest"], "extra": 1}, "k"),
    ("experiment", {"experiment_id": "missing"}, "k"),
    ("experiment", {"model_ids": ["random_forest"]}, ""),
    ("oof", {"run_id": "overridden"}, "k"), ("submission", [], "k"),
])
def test_invalid_submissions_leave_no_jobs(app, operation, payload, key):
    _, commands, queries, run, _ = app
    with pytest.raises((ValueError, TypeError)):
        commands.dispatch(SubmitJobCommand(operation, run.id, payload, key))
    assert queries.dispatch(ListJobsQuery()) == []


def test_foreign_experiment_is_rejected_for_all_operations(app):
    workspace, commands, _, run, test_path = app
    foreign_run = workspace.create_run(workspace._get_dataset(run.dataset_id))
    foreign = commands.dispatch(CreateExperimentCommand(foreign_run.id, "foreign", ["x"], model_ids=["random_forest"]))
    for operation in ("experiment", "oof", "submission"):
        payload = {"experiment_id": foreign.id}
        if operation != "experiment":
            payload.update(test_dataset_path=str(test_path), output_path=str(test_path.parent / "out.csv"))
        with pytest.raises(ValueError, match="belong"):
            commands.dispatch(SubmitJobCommand(operation, run.id, payload, operation))


def test_worker_completes_real_experiment_and_persists_progress(app):
    workspace, _, queries, run, _ = app
    job_id = submit(app)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        assert worker.run_once()
        assert not worker.run_once()
    finally:
        worker.close()
    job = queries.dispatch(GetJobQuery(job_id))
    assert job["status"] == "completed", job
    assert job["completed"] == job["total"] == 2
    results = workspace.repository.list_trial_results(job["result"]["experiment_id"])
    assert len(results) == 2 and all(r.succeeded for r in results)
    assert job["result"]["primary_metric"] == "roc_auc"
    assert workspace.repository.get_checkpoint(run.id) is None


@pytest.mark.parametrize("action,target", [("pause", "paused"), ("cancel", "cancelled")])
def test_control_during_fit_is_observed_at_model_boundary(app, monkeypatch, action, target):
    workspace, commands, queries, _, _ = app
    job_id = submit(app)
    original = SklearnTrainer.run
    calls = []
    def controlled(self, execution):
        calls.append(execution.trial.model_id)
        result = original(self, execution)
        commands.dispatch(ControlJobCommand(job_id, action))
        return result
    monkeypatch.setattr(SklearnTrainer, "run", controlled)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
        job = queries.dispatch(GetJobQuery(job_id))
        assert job["status"] == target
        assert job["completed"] == 1
        assert calls == ["logistic_regression"]
        if action == "pause":
            commands.dispatch(ControlJobCommand(job_id, "resume"))
            monkeypatch.setattr(SklearnTrainer, "run", original)
            worker.run_once()
            done = queries.dispatch(GetJobQuery(job_id))
            assert done["status"] == "completed", done
            assert done["attempt"] == 2
            assert done["completed"] == 2
            assert len(workspace.repository.list_trial_results(done["result"]["experiment_id"])) == 2
    finally:
        worker.close()


def test_restart_marks_inflight_interrupted_and_requires_retry(app):
    workspace, commands, queries, _, _ = app
    job_id = submit(app)
    repository(app).claim()  # Simulate a process disappearing with a running record.
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "interrupted"
        assert not worker.run_once()
        commands.dispatch(ControlJobCommand(job_id, "retry"))
        worker.run_once()
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "completed"
    finally:
        worker.close()


def test_restart_honors_pending_cancellation(app):
    job_id = submit(app)
    repo = repository(app)
    repo.claim()
    app[1].dispatch(ControlJobCommand(job_id, "cancel"))
    worker = JobWorker(str(app[0].root_dir)).start(background=False)
    try:
        assert repo.get(job_id).status == JobStatus.CANCELLED
    finally:
        worker.close()


def test_workspace_lease_prevents_second_worker_and_read_recovery(app):
    workspace, _, queries, _, _ = app
    job_id = submit(app)
    first = JobWorker(str(workspace.root_dir)).start(background=False)
    second = JobWorker(str(workspace.root_dir))
    try:
        first.repository.claim()
        with pytest.raises(RuntimeError, match="Another worker"):
            second.start(background=False)
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "running"
        with pytest.raises(RuntimeError, match="already started"):
            first.start()
        with pytest.raises(RuntimeError, match="Start the worker"):
            second.run_once()
    finally:
        first.close()
        second.close()


def test_shutdown_during_fit_retains_finished_model_for_retry(app, monkeypatch):
    workspace, commands, queries, _, _ = app
    job_id = submit(app)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    original = SklearnTrainer.run
    def stopping(self, execution):
        result = original(self, execution)
        worker.close()  # Foreground executor; callback sees cooperative shutdown.
        return result
    monkeypatch.setattr(SklearnTrainer, "run", stopping)
    worker.run_once()
    job = queries.dispatch(GetJobQuery(job_id))
    assert job["status"] == "interrupted"
    assert job["completed"] == 1
    commands.dispatch(ControlJobCommand(job_id, "retry"))
    monkeypatch.setattr(SklearnTrainer, "run", original)
    worker.start(background=False)
    try:
        worker.run_once()
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "completed"
    finally:
        worker.close()


def test_failure_is_durable_and_retry_is_explicit(app, monkeypatch):
    workspace, commands, queries, _, _ = app
    job_id = submit(app)
    original = SklearnTrainer.run
    def fail(self, execution):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(SklearnTrainer, "run", fail)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
        job = queries.dispatch(GetJobQuery(job_id))
        assert job["status"] == "failed"
        assert job["error"] == "backend unavailable"
        monkeypatch.setattr(SklearnTrainer, "run", original)
        commands.dispatch(ControlJobCommand(job_id, "retry"))
        worker.run_once()
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "completed"
    finally:
        worker.close()


def test_oof_job_reports_actual_folds_and_submission(app, tmp_path):
    workspace, commands, queries, run, test_path = app
    source = commands.dispatch(CreateExperimentCommand(run.id, "source", ["x"], model_ids=["logistic_regression"]))
    output = tmp_path / "submission.csv"
    payload = dict(experiment_id=source.id, test_dataset_path=str(test_path), output_path=str(output), folds=2)
    job_id = commands.dispatch(SubmitJobCommand("oof", run.id, payload, "oof"))
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
    finally:
        worker.close()
    job = queries.dispatch(GetJobQuery(job_id))
    assert job["status"] == "completed", job
    assert job["completed"] == job["total"] == 2
    assert job["result"]["oof"]["folds"] == 2
    assert len(pd.read_csv(output)) == 5
    assert not job["result"]["oof"]["promoted"]


def test_cli_uses_same_job_commands_and_queries(app, capsys):
    workspace, _, queries, run, _ = app
    common = ["--workspace", str(workspace.root_dir)]
    assert main(["job", "submit", *common, "--operation", "experiment", "--run-id", run.id,
                 "--payload", '{"model_ids":["random_forest"]}', "--key", "cli"]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    assert main(["job", "pause", *common, "--job-id", job_id]) == 0
    capsys.readouterr()
    assert queries.dispatch(GetJobQuery(job_id))["status"] == "paused"
    assert main(["job", "show", *common, "--job-id", job_id]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == job_id
    assert main(["job", "list", *common, "--run-id", run.id]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_http_returns_202_without_training_and_exposes_controls(app, monkeypatch):
    workspace, _, queries, run, _ = app
    monkeypatch.setattr(AutoMLWebHandler, "workspace_dir", str(workspace.root_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(base + path, data=body, headers={"Content-Type": "application/json"}), timeout=5) as response:
            return response.status, json.loads(response.read())
    try:
        status, response = request("/api/jobs", dict(operation="experiment", run_id=run.id,
                         payload={"model_ids": ["random_forest"]}, idempotency_key="http"))
        assert status == 202
        job_id = response["job_id"]
        assert request(response["status_url"])[1]["status"] == "queued"
        assert workspace.repository.list_experiments(run.id) == []
        assert request(f"/api/jobs/{job_id}/pause", {})[0] == 200
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "paused"
        assert len(request("/api/jobs?run_id=" + run.id)[1]) == 1
        with pytest.raises(HTTPError) as error:
            request("/api/jobs/missing")
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            request(f"/api/jobs/{job_id}/retry", {})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("action,status", [("pause", JobStatus.PAUSED), ("cancel", JobStatus.CANCELLED)])
def test_committed_control_wins_completion_race(app, action, status):
    job_id = submit(app)
    repo = repository(app)
    repo.claim()
    app[1].dispatch(ControlJobCommand(job_id, action))
    result = repo.finish(job_id, JobStatus.COMPLETED, result={"completed_fit": True})
    assert result.status == status
    assert result.result == {"completed_fit": True}


def test_running_worker_retains_lease_until_fit_stops(app, monkeypatch):
    workspace, _, queries, _, _ = app
    job_id = submit(app)
    entered, release = threading.Event(), threading.Event()
    original = SklearnTrainer.run
    def blocked(self, execution):
        entered.set()
        assert release.wait(5)
        return original(self, execution)
    monkeypatch.setattr(SklearnTrainer, "run", blocked)
    first = JobWorker(str(workspace.root_dir)).start()
    second = JobWorker(str(workspace.root_dir))
    try:
        assert entered.wait(5)
        assert not first.close(timeout=0.01)
        with pytest.raises(RuntimeError, match="Another worker"):
            second.start()
        release.set()
        assert first.close(timeout=5)
        assert queries.dispatch(GetJobQuery(job_id))["status"] == "interrupted"
    finally:
        release.set()
        first.close(timeout=5)
        second.close()


def test_oof_pause_restarts_partial_folds_without_export(app, tmp_path, monkeypatch):
    import automl.application.services.oof_submission as oof
    workspace, commands, queries, run, test_path = app
    source = commands.dispatch(CreateExperimentCommand(run.id, "source", ["x"], model_ids=["logistic_regression"]))
    output = tmp_path / "submission.csv"
    payload = dict(experiment_id=source.id, test_dataset_path=str(test_path), output_path=str(output), folds=2)
    job_id = commands.dispatch(SubmitJobCommand("oof", run.id, payload, "oof"))
    original = oof.evaluate_oof
    def paused(*args, **kwargs):
        progress = kwargs["progress"]
        def observe(completed, total, message):
            progress(completed, total, message)
            if completed == 1:
                commands.dispatch(ControlJobCommand(job_id, "pause"))
        kwargs["progress"] = observe
        return original(*args, **kwargs)
    monkeypatch.setattr(oof, "evaluate_oof", paused)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
        job = queries.dispatch(GetJobQuery(job_id))
        assert job["status"] == "paused", job
        assert job["completed"] == 1
        assert not output.exists()
        commands.dispatch(ControlJobCommand(job_id, "resume"))
        monkeypatch.setattr(oof, "evaluate_oof", original)
        worker.run_once()
        job = queries.dispatch(GetJobQuery(job_id))
        assert job["status"] == "completed", job
        assert job["attempt"] == 2 and job["completed"] == 2
        assert len(pd.read_csv(output)) == 5
    finally:
        worker.close()


def test_submission_job_uses_application_export(app, tmp_path):
    workspace, commands, queries, run, test_path = app
    experiment_job = submit(app)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
        experiment_id = queries.dispatch(GetJobQuery(experiment_job))["result"]["experiment_id"]
        output = tmp_path / "submission.csv"
        job_id = commands.dispatch(SubmitJobCommand("submission", run.id, {
            "experiment_id": experiment_id, "test_dataset_path": str(test_path),
            "output_path": str(output), "predict_proba": True,
        }, "export"))
        worker.run_once()
        job = queries.dispatch(GetJobQuery(job_id))
        assert job["status"] == "completed", job
        assert job["completed"] == job["total"] == 1
        assert job["result"]["row_count"] == 5
    finally:
        worker.close()


def test_running_claim_after_stale_control_cannot_be_marked_paused_directly(app, monkeypatch):
    _, commands, queries, _, _ = app
    job_id = submit(app)
    from automl.application.services.jobs import JobService
    service = JobService(app[0], repository(app))
    original = service.repository.get
    def raced(job_id):
        queued = original(job_id)
        service.repository.claim()
        return queued
    monkeypatch.setattr(service.repository, "get", raced)
    with pytest.raises(ValueError, match="not allowed"):
        service.control(ControlJobCommand(job_id, "pause"))
    assert queries.dispatch(GetJobQuery(job_id))["status"] == "running"


@pytest.mark.parametrize("partial_success", [False, True])
def test_retry_failed_trial_results_refits_failures_and_reuses_successes(app, monkeypatch, partial_success):
    from automl.domain.experiments.trial import TrialResult
    workspace, commands, queries, _, _ = app
    job_id = submit(app)
    original = SklearnTrainer.run
    def failed_result(self, execution):
        if partial_success:
            if execution.trial.model_id == "logistic_regression":
                return original(self, execution)
            raise RuntimeError("temporary failure")
        return TrialResult(trial_id=execution.trial.id, experiment_id=execution.experiment.id,
                           model_id=execution.trial.model_id, primary_metric="roc_auc", primary_score=0,
                           failure_reason="temporary failure")
    monkeypatch.setattr(SklearnTrainer, "run", failed_result)
    worker = JobWorker(str(workspace.root_dir)).start(background=False)
    try:
        worker.run_once()
        failed = queries.dispatch(GetJobQuery(job_id))
        assert failed["status"] == "failed" and failed["completed"] == (1 if partial_success else 2)
        commands.dispatch(ControlJobCommand(job_id, "retry"))
        calls = []
        def recording(self, execution):
            calls.append(execution.trial.model_id)
            return original(self, execution)
        monkeypatch.setattr(SklearnTrainer, "run", recording)
        worker.run_once()
        done = queries.dispatch(GetJobQuery(job_id))
        assert done["status"] == "completed", done
        assert calls == (["random_forest"] if partial_success else ["logistic_regression", "random_forest"])
        assert len(workspace.repository.list_trial_results(done["result"]["experiment_id"])) == (2 if partial_success else 4)
    finally:
        worker.close()
