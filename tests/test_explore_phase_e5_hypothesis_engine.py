"""Tests for CATML Explore Phase E5: Hypothesis Engine (Scientific Agent & AutoML Link).

Validates:
- Deterministic hypothesis-to-experiment translation (HypothesisExperimentTranslator).
- "Propose ≠ Accept" verification protocol on identical validation split.
- EvidenceLink creation, persistence, and state transitions (proposed -> accepted / rejected).
- MCP tool analysis_verify_hypothesis and resource catml://studies/{study_id}/evidence.
- CLI command `catml explore verify` and report export with evidence links table.
- REST API endpoints /api/analysis/hypotheses/verify and /api/analysis/evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from automl.application.analysis.commands import (
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetEvidenceLinkQuery,
    GetHypothesisQuery,
    ListEvidenceLinksQuery,
    ListHypothesesQuery,
)
from automl.application.bootstrap import build_application
from automl.domain.analysis.models import AnalysisHypothesis
from automl.engine.analysis.hypothesis_translator import (
    HypothesisExperimentTranslator,
)
from automl.interfaces.cli.explore_cli import run_explore_cli
from automl.interfaces.mcp.server import create_mcp_server


@pytest.fixture
def synthetic_e5_data(tmp_path: Path) -> Path:
    """Creates a dataset where dropping a collinear feature improves model generalization."""
    np.random.seed(123)
    n = 200
    # Clean predictive signal
    x_true = np.random.normal(loc=0.0, scale=1.0, size=n)
    # Collinear feature with heavy noise
    x_noise = x_true * 1.0 + np.random.normal(loc=0.0, scale=0.02, size=n)
    x3 = np.random.normal(loc=2.0, scale=1.5, size=n)

    # Ground truth depends cleanly on x_true and x3
    prob = 1.0 / (1.0 + np.exp(-(1.5 * x_true + 0.8 * x3)))
    target = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "signal_var": x_true,
        "collinear_noise": x_noise,
        "secondary_var": x3,
        "target": target,
    })
    path = tmp_path / "e5_synthetic.csv"
    df.to_csv(path, index=False)
    return path


# =========================================================================
# 1. TRANSLATOR UNIT TESTS
# =========================================================================

def test_hypothesis_experiment_translator_rules():
    """Verify HypothesisExperimentTranslator applies deterministic mapping rules."""
    available = ["feat_a", "feat_b", "feat_c"]

    # 1. resolve_collinearity: drops second feature
    hyp_collin = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f1",
        description="Drop collinear feature",
        proposed_action="resolve_collinearity",
        experiment_delta={"collinear_pair": ["feat_a", "feat_b"]},
    )
    spec_collin = HypothesisExperimentTranslator.translate(hyp_collin, available, "binary_classification")
    assert "feat_b" not in spec_collin.feature_names
    assert "feat_a" in spec_collin.feature_names
    assert "drop_collinear_feat_b" in spec_collin.name

    # 2. prioritize_feature: places prioritized feature first
    hyp_prio = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f2",
        description="Prioritize feat_c",
        proposed_action="prioritize_feature",
        experiment_delta={"feature": "feat_c"},
    )
    spec_prio = HypothesisExperimentTranslator.translate(hyp_prio, available, "binary_classification")
    assert spec_prio.feature_names[0] == "feat_c"
    assert len(spec_prio.feature_names) == 3

    # 3. nonlinear_transform_or_trees: configures tree ensemble models
    hyp_tree = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f3",
        description="Use tree models",
        proposed_action="nonlinear_transform_or_trees",
        experiment_delta={"suggested_models": ["lightgbm", "random_forest"]},
    )
    spec_tree = HypothesisExperimentTranslator.translate(hyp_tree, available, "binary_classification")
    assert spec_tree.model_ids == ["lightgbm", "random_forest"]

    # 4. Explicit drop_features delta override
    hyp_drop = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f4",
        description="Explicit drop",
        proposed_action="custom",
        experiment_delta={"drop_features": ["feat_a"]},
    )
    spec_drop = HypothesisExperimentTranslator.translate(hyp_drop, available, "binary_classification")
    assert spec_drop.feature_names == ["feat_b", "feat_c"]


# =========================================================================
# 2. "PROPOSE ≠ ACCEPT" VERIFICATION WORKFLOW & EVIDENCE LINK
# =========================================================================

def test_hypothesis_verification_workflow_propose_not_accept(synthetic_e5_data: Path, tmp_path: Path):
    """Verify full hypothesis verification cycle: proposed -> trial -> accepted/rejected -> EvidenceLink."""
    ws_dir = str(tmp_path / "ws_e5_verify")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="e5_ds", path=synthetic_e5_data, target="target", task_type="binary_classification")

    # Run explore study to generate hypotheses
    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="E5 Verification Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    # Fetch generated hypotheses
    hyps = qb.dispatch(ListHypothesesQuery(study_id=study_id))
    assert len(hyps) >= 1

    first_hyp = hyps[0]
    hyp_id = first_hyp["id"]
    assert first_hyp["status"] == "proposed", "Hypotheses must strictly initialize in 'proposed' state"

    # Verify hypothesis with zero improvement threshold
    link_data = cb.dispatch(
        VerifyHypothesisCommand(
            hypothesis_id=hyp_id,
            min_improvement=0.0,
        )
    )

    assert link_data is not None
    assert "id" in link_data
    assert link_data["hypothesis_id"] == hyp_id
    assert link_data["baseline_score"] > 0
    assert link_data["candidate_score"] > 0
    assert "metric" in link_data
    assert "accepted" in link_data
    assert "notes" in link_data

    # Check updated hypothesis state in DB
    updated_hyp = qb.dispatch(GetHypothesisQuery(hypothesis_id=hyp_id))
    assert updated_hyp["status"] in ("accepted", "rejected")
    if link_data["accepted"]:
        assert updated_hyp["status"] == "accepted"
    else:
        assert updated_hyp["status"] == "rejected"

    # Check EvidenceLink persisted in repository
    retrieved_link = qb.dispatch(GetEvidenceLinkQuery(link_id=link_data["id"]))
    assert retrieved_link is not None
    assert retrieved_link["id"] == link_data["id"]

    # Query all evidence links for study
    study_links = qb.dispatch(ListEvidenceLinksQuery(study_id=study_id))
    assert len(study_links) >= 1
    assert any(l["id"] == link_data["id"] for l in study_links)


def test_hypothesis_verification_rejection_negative_learning(synthetic_e5_data: Path, tmp_path: Path):
    """Verify that an unsubstantiated hypothesis is rejected with negative evidence recorded."""
    ws_dir = str(tmp_path / "ws_e5_reject")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="reject_ds", path=synthetic_e5_data, target="target", task_type="binary_classification")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="E5 Reject Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    hyps = qb.dispatch(ListHypothesesQuery(study_id=study_id))
    assert len(hyps) >= 1
    target_hyp = hyps[0]

    # Demand an impossibly high improvement delta (0.80) to force rejection
    link_data = cb.dispatch(
        VerifyHypothesisCommand(
            hypothesis_id=target_hyp["id"],
            min_improvement=0.80,
        )
    )

    assert link_data["accepted"] is False
    assert "Rejected" in link_data["notes"]

    updated_hyp = qb.dispatch(GetHypothesisQuery(hypothesis_id=target_hyp["id"]))
    assert updated_hyp["status"] == "rejected"


# =========================================================================
# 3. MCP SERVER VERIFY TOOL & EVIDENCE RESOURCE
# =========================================================================

def test_mcp_verify_hypothesis_tool_and_resource(synthetic_e5_data: Path, tmp_path: Path):
    """Verify analysis_verify_hypothesis tool and study evidence resource via MCP server."""
    ws_dir = str(tmp_path / "ws_mcp_e5")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="mcp_e5_ds", path=synthetic_e5_data, target="target")
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    # 1. Create and run study via MCP
    c_res = asyncio.run(server.call_tool("analysis_create_study", {"dataset_id": ds.id, "target_column": "target"}))
    study_id = json.loads(c_res.content[0].text)["id"]
    asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))

    # 2. Fetch findings and propose experiment
    f_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id, "limit": 1}))
    findings = json.loads(f_res.content[0].text)
    finding_id = findings[0]["id"]

    prop_res = asyncio.run(
        server.call_tool(
            "analysis_propose_experiment",
            {
                "study_id": study_id,
                "finding_id": finding_id,
                "hypothesis_description": "Drop collinear noise feature to improve test AUC",
                "proposed_action": "resolve_collinearity",
                "transformation_spec": {"collinear_pair": ["signal_var", "collinear_noise"]},
            },
        )
    )
    prop_data = json.loads(prop_res.content[0].text)
    hyp_id = prop_data["hypothesis_id"]
    assert prop_data["status"] == "proposed"

    # 3. Call analysis_verify_hypothesis tool
    v_res = asyncio.run(
        server.call_tool(
            "analysis_verify_hypothesis",
            {"hypothesis_id": hyp_id, "min_improvement": 0.0},
        )
    )
    assert not v_res.is_error
    verify_payload = json.loads(v_res.content[0].text)
    assert verify_payload["hypothesis_id"] == hyp_id
    assert "baseline_score" in verify_payload
    assert "candidate_score" in verify_payload

    # 4. Read evidence resource
    ev_raw = asyncio.run(server.read_resource(f"catml://studies/{study_id}/evidence"))
    assert ev_raw
    ev_str = ev_raw[0].content if isinstance(ev_raw, list) else ev_raw
    ev_list = json.loads(ev_str)
    assert isinstance(ev_list, list)
    assert len(ev_list) >= 1
    assert any(e["hypothesis_id"] == hyp_id for e in ev_list)


# =========================================================================
# 4. CLI EXPLORE VERIFY & EVIDENCE EXPORT
# =========================================================================

def test_cli_explore_verify_and_export(synthetic_e5_data: Path, tmp_path: Path, capsys):
    """Verify `catml explore verify` command and exported report contains Section 5."""
    ws_dir = str(tmp_path / "ws_cli_e5")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="cli_e5_ds", path=synthetic_e5_data, target="target")

    # Create & run study
    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="E5 CLI Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))
    hyps = qb.dispatch(ListHypothesesQuery(study_id=study_id))
    assert len(hyps) >= 1
    hyp_id = hyps[0]["id"]

    # CLI explore verify (JSON mode)
    args_verify_json = argparse.Namespace(
        workspace=ws_dir,
        explore_action="verify",
        hypothesis_id=hyp_id,
        run_id=None,
        min_improvement=0.0,
        json=True,
    )
    rc = run_explore_cli(args_verify_json)
    assert rc == 0
    out_json = capsys.readouterr().out
    link_json = json.loads(out_json)
    assert link_json["hypothesis_id"] == hyp_id
    assert "accepted" in link_json

    # CLI explore verify (Text mode)
    args_verify_txt = argparse.Namespace(
        workspace=ws_dir,
        explore_action="verify",
        hypothesis_id=hyp_id,
        run_id=None,
        min_improvement=0.0,
        json=False,
    )
    rc2 = run_explore_cli(args_verify_txt)
    assert rc2 == 0
    out_txt = capsys.readouterr().out
    assert "Hypothesis Verification Result:" in out_txt
    assert "Baseline Score:" in out_txt

    # CLI explore export markdown: Section 5 must be present and contain evidence
    md_out = tmp_path / "e5_report.md"
    args_export = argparse.Namespace(
        workspace=ws_dir,
        explore_action="export",
        study_id=study_id,
        format="markdown",
        output=str(md_out),
    )
    rc3 = run_explore_cli(args_export)
    assert rc3 == 0
    content = md_out.read_text(encoding="utf-8")
    assert "## 5. Empirical Verification & Evidence Links" in content
    assert link_json["id"] in content
    assert "Empirically Verified Evidence Links:** 2" in content


# =========================================================================
# 5. REST API EVIDENCE ENDPOINTS
# =========================================================================

def test_web_api_evidence_endpoints(synthetic_e5_data: Path, tmp_path: Path):
    """Verify HTTP REST endpoints: /api/analysis/hypotheses/verify and /api/analysis/evidence."""
    from http.server import ThreadingHTTPServer
    import threading
    from urllib.request import Request, urlopen
    from automl.interfaces.web.server import AutoMLWebHandler

    ws_dir = str(tmp_path / "ws_web_e5")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="web_e5_ds", path=synthetic_e5_data, target="target")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="E5 Web Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))
    hyps = qb.dispatch(ListHypothesesQuery(study_id=study_id))
    assert len(hyps) >= 1
    hyp_id = hyps[0]["id"]

    AutoMLWebHandler.workspace_dir = ws_dir
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. POST /api/analysis/hypotheses/verify
        req_verify = Request(
            f"{base_url}/api/analysis/hypotheses/verify",
            data=json.dumps({"hypothesis_id": hyp_id, "min_improvement": 0.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req_verify, timeout=30.0) as resp:
            assert resp.status == 200
            verify_res = json.loads(resp.read().decode("utf-8"))
            assert verify_res["hypothesis_id"] == hyp_id
            assert "accepted" in verify_res
            link_id = verify_res["id"]

        # 2. GET /api/analysis/evidence?study_id=...
        with urlopen(f"{base_url}/api/analysis/evidence?study_id={study_id}", timeout=10.0) as resp:
            assert resp.status == 200
            evidence_res = json.loads(resp.read().decode("utf-8"))
            assert isinstance(evidence_res, list)
            assert len(evidence_res) >= 1
            assert any(e["id"] == link_id for e in evidence_res)

    finally:
        server.shutdown()
        server.server_close()
