"""Sprint 1 — Workbench state coherence: statuses, trial counters, identifier hyps."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.domain.experiments.trial import Experiment, ExperimentStatus, TrialResult
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer
from automl.interfaces.web.server import (
    AutoMLWebHandler,
    _effective_experiment_status,
)


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


def _parse_response_json(wfile: io.BytesIO) -> tuple[int, object]:
    raw = wfile.getvalue().decode("utf-8", errors="ignore")
    status_line = raw.split("\r\n")[0]
    status_code = int(status_line.split()[1])
    body = raw.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in raw else "{}"
    return status_code, json.loads(body)


def test_effective_experiment_status_derives_completed_from_trials():
    exp = Experiment(
        id="exp_1",
        run_id="run_1",
        name="Lagging",
        hypothesis="test",
        feature_names=["a"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
        status=ExperimentStatus.CREATED,
    )
    trials = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp_1",
            model_id="logistic_regression",
            primary_metric="roc_auc",
            primary_score=0.9,
        )
    ]
    assert _effective_experiment_status(exp, trials) == "COMPLETED"
    assert _effective_experiment_status(exp, []) == "CREATED"


def test_api_experiments_status_coerces_created_with_trials(tmp_path: Path):
    ws_dir = tmp_path / "ws"
    ws, cmd, qry = build_application(root_dir=str(ws_dir))

    csv_file = ws_dir / "data.csv"
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })
    df.to_csv(csv_file, index=False)

    ds = ws.register_dataset(name="demo", path=str(csv_file), target="y")
    run = ws.create_run(dataset=ds, metric="accuracy")
    exp = ws.create_experiment(
        run=run,
        name="Baseline",
        hypothesis="smoke",
        feature_names=["f1", "f2"],
        model_ids=["logistic_regression"],
    )
    ws.run_experiment(run, exp)

    # Simulate stale persisted status after trials were written (bypass transition map).
    with ws.repository._connect() as conn:
        conn.execute(
            "UPDATE experiments SET status = ? WHERE id = ?",
            (ExperimentStatus.CREATED.value, exp.id),
        )
    exp.status = ExperimentStatus.CREATED

    handler, wfile = _create_handler(ws_dir, f"/api/experiments?run_id={run.id}")
    handler.do_GET()
    status, data = _parse_response_json(wfile)
    assert status == 200
    assert isinstance(data, list) and len(data) >= 1
    assert data[0]["status"] == "COMPLETED"
    assert data[0]["trials_count"] >= 1

    handler2, wfile2 = _create_handler(ws_dir, f"/api/plan?run_id={run.id}")
    handler2.do_GET()
    status2, plan = _parse_response_json(wfile2)
    assert status2 == 200
    steps = plan.get("steps") or plan.get("plan_steps") or []
    if not steps and isinstance(plan, dict):
        steps = plan.get("plan", {}).get("steps") if isinstance(plan.get("plan"), dict) else []
    assert any(s.get("status") == "COMPLETED" for s in steps) or any(
        s.get("name") == "Baseline" and s.get("status") == "COMPLETED" for s in (plan.get("steps") or [])
    ) or '"status": "COMPLETED"' in json.dumps(plan)


def test_frontend_utils_resolve_trials_and_status_labels():
    utils = Path("src/automl/interfaces/web/static/js/utils.js").read_text(encoding="utf-8")
    assert "function resolveTrialsCount" in utils
    assert "function formatExperimentStatus" in utils
    assert "EXPERIMENT_STATUS_LABELS" in utils

    app = Path("src/automl/interfaces/web/static/js/app.js").read_text(encoding="utf-8")
    assert "estado de run, no de experimento" in app or "LISTO" in app
    assert "runs[0]" not in app or "activeRun" in app

    studio = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")
    assert "resolveTrialsCount" in studio

    compare = Path("src/automl/interfaces/web/static/js/views/compare.js").read_text(encoding="utf-8")
    assert "Estado del experimento" in compare
    assert "formatExperimentStatus" in compare


def test_analyzer_skips_transform_hyps_on_identifier_column(tmp_path: Path):
    """Numeric high-cardinality `id` must get exclude_identifier, not discretize."""
    n = 120
    rng = np.random.default_rng(7)
    # Nearly-unique numeric id + multimodal feature that would otherwise suggest discretize
    ids = np.arange(n, dtype=float)
    multimodal = np.concatenate([
        rng.normal(-3, 0.3, n // 2),
        rng.normal(3, 0.3, n - n // 2),
    ])
    y = (multimodal > 0).astype(int)
    df = pd.DataFrame({"id": ids, "x_multi": multimodal, "y": y})
    path = tmp_path / "id_demo.csv"
    df.to_csv(path, index=False)

    analyzer = StatisticalAnalyzer()
    findings, _, hypotheses = analyzer.analyze(
        data_source_path=str(path),
        target_column="y",
        analysis_types=["descriptive"],
        parameters={"exclude_columns": ["id"]},
    )

    id_actions = {
        h.proposed_action
        for h in hypotheses
        if (h.experiment_delta or {}).get("column") == "id"
        or "id" in ((h.experiment_delta or {}).get("exclude_columns") or [])
    }
    assert "exclude_identifier" in id_actions or any(
        f.column_name == "id" and f.finding_type == "high_cardinality_identifier" for f in findings
    )
    assert "discretize_or_cluster" not in id_actions
    assert not any(
        h.proposed_action == "discretize_or_cluster"
        and (h.experiment_delta or {}).get("column") == "id"
        for h in hypotheses
    )
