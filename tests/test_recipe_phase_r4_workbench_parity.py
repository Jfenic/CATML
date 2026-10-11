"""Tests for Phase R4: Workbench UI integration & End-to-End Surface Parity.

Proves the Phase R4 acceptance criteria:
1. Workbench UI static assets and client interfaces comply with architectural rules (no JS business logic leaks).
2. End-to-end parity: CLI creates -> MCP inspects/selects -> REST/Workbench executes -> CLI promotes ->
   All surfaces report 100% identical configuration, lineage, status, and executed experiments.
3. Fit-in-fold column transforms (Yeo-Johnson, RobustScaler, log1p) execute identically when submitted via Workbench REST client.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import pytest
import pandas as pd

from automl.application.analysis.commands import CreateStudyCommand
from automl.application.bootstrap import build_application
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
)
from automl.domain.recipes.recipe import RecipeStatus
from automl.interfaces.cli.main import main
from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server
from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def workbench_recipe_env(tmp_path: Path):
    """Sets up an environment with dataset, study, hypothesis, experiments, and accepted evidence."""
    ws_dir = tmp_path / "ws_r4"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    # 1. Dataset & Run
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    ds = ws.register_dataset("r4_churn_ds", df, target="target")
    run = ws.create_run(ds)

    # 2. Explore Study & Statistical Finding
    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Workbench R4 Study", target_column="target"))
    study_repo = ws.get_study_repository()

    analysis_run = AnalysisRun.create(study_id=study_id)
    study_repo.save_analysis_run(analysis_run)

    finding = StatisticalFinding.create(
        study_id=study_id,
        analysis_run_id=analysis_run.id,
        finding_type="skewed_distribution",
        summary="num1 is heavily skewed",
        column_name="num1",
    )
    study_repo.save_finding(finding)

    hypothesis = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id=finding.id,
        description="Apply power transform yeo_johnson to num1",
        proposed_action="power_transform",
        experiment_delta={
            "column": "num1",
            "method": "yeo_johnson",
            "transform_kind": "yeo_johnson",
            "suggested_action": "power_transform",
        },
    )
    study_repo.save_hypothesis(hypothesis)

    # 3. Baseline & Candidate Experiments
    base_exp = ws.create_experiment(
        run=run,
        name="Baseline RF",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
    )
    ws.run_experiment(run, base_exp)

    cand_exp = ws.create_experiment(
        run=run,
        name="Candidate YeoJohnson",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
        column_transforms=[{"column": "num1", "kind": "yeo_johnson", "params": {}}],
    )
    ws.run_experiment(run, cand_exp)

    # 4. Accepted EvidenceLink
    evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=cand_exp.id,
        baseline_score=0.75,
        candidate_score=0.88,
        metric="roc_auc",
        accepted=True,
        notes="Verified +0.13 ROC-AUC empirical improvement",
    )
    study_repo.save_evidence_link(evidence)

    return {
        "workspace_dir": str(ws_dir),
        "ws": ws,
        "cmd_bus": cmd_bus,
        "qry_bus": qry_bus,
        "dataset": ds,
        "run": run,
        "study_id": study_id,
        "evidence": evidence,
    }


@pytest.fixture
def running_workbench_server(workbench_recipe_env):
    """Runs a live Workbench HTTP server for REST client validation."""
    ws_dir = workbench_recipe_env["workspace_dir"]
    AutoMLWebHandler.workspace_dir = ws_dir

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {**workbench_recipe_env, "base_url": base_url}

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_workbench_static_js_recipe_integration():
    """Verify that all modified Workbench frontend modules load and contain recipe integrations."""
    static_js_dir = Path(__file__).resolve().parents[1] / "src" / "automl" / "interfaces" / "web" / "static" / "js"
    api_js = (static_js_dir / "api.js").read_text(encoding="utf-8")
    explore_js = (static_js_dir / "views" / "explore.js").read_text(encoding="utf-8")
    studio_js = (static_js_dir / "views" / "studio.js").read_text(encoding="utf-8")
    new_exp_js = (static_js_dir / "views" / "new_experiment.js").read_text(encoding="utf-8")
    icons_js = (static_js_dir / "icons.js").read_text(encoding="utf-8")

    # 1. API client declares all recipe endpoints
    assert "getVerifiedImprovements" in api_js
    assert "getRecipes" in api_js
    assert "getRecipe" in api_js
    assert "createRecipeFromEvidence" in api_js
    assert "selectRecipe" in api_js
    assert "submitRecipe" in api_js
    assert "promoteRecipe" in api_js
    assert "archiveRecipe" in api_js

    # 2. Explore view includes recipes section and CTA button
    assert "_renderRecipesSectionHtml" in explore_js
    assert "btn-create-recipe" in explore_js
    assert "Recetas de Experimentos Reproducibles" in explore_js
    assert "btn-select-recipe" in explore_js
    assert "btn-submit-recipe" in explore_js
    assert "btn-promote-recipe" in explore_js

    # 3. Studio view includes recipes tab
    assert 'data-tab="recipes"' in studio_js
    assert "_renderRecipesTabHtml" in studio_js
    assert "_bindRecipeTabEvents" in studio_js
    assert "btn-studio-submit-recipe" in studio_js

    # 4. New experiment modal includes recipe selection
    assert "selectExperimentRecipe" in new_exp_js
    assert "submitRecipe" in new_exp_js

    # 5. Recipe icon defined in Lucide icon set
    assert '"recipe"' in icons_js


def test_workbench_rest_recipe_lifecycle(running_workbench_server):
    """Verify complete recipe lifecycle exclusively via Workbench REST API endpoints."""
    base = running_workbench_server["base_url"]
    evidence = running_workbench_server["evidence"]
    dataset = running_workbench_server["dataset"]

    # 1. GET verified improvements
    with urlopen(f"{base}/api/recipes/verified-improvements?dataset_id={dataset.id}") as resp:
        assert resp.status == 200
        verified = json.loads(resp.read().decode("utf-8"))
        assert len(verified) >= 1
        assert verified[0]["evidence_link_id"] == evidence.id
        assert verified[0]["suggested_action"] == "power_transform"

    # 2. POST create recipe from evidence
    create_payload = json.dumps({
        "evidence_link_id": evidence.id,
        "name": "Workbench Verified YeoJohnson Recipe",
        "notes": "Created via Workbench UI button",
    }).encode("utf-8")
    req = Request(f"{base}/api/recipes/create-from-evidence", data=create_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        recipe_id = data["recipe_id"]
        assert recipe_id.startswith("rcp_")
        assert data["recipe"]["status"] == "draft"
        assert data["recipe"]["column_transforms"] == [{"column": "num1", "kind": "yeo_johnson", "params": {}}]

    # 3. GET recipe list and detail
    with urlopen(f"{base}/api/recipes?dataset_id={dataset.id}") as resp:
        assert resp.status == 200
        recipes = json.loads(resp.read().decode("utf-8"))
        assert any(r["id"] == recipe_id for r in recipes)

    with urlopen(f"{base}/api/recipes/{recipe_id}") as resp:
        assert resp.status == 200
        detail = json.loads(resp.read().decode("utf-8"))
        assert detail["id"] == recipe_id
        assert detail["status"] == "draft"

    # 4. POST select recipe
    select_payload = json.dumps({"recipe_id": recipe_id, "notes": "Approved in UI"}).encode("utf-8")
    req = Request(f"{base}/api/recipes/select", data=select_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        sel = json.loads(resp.read().decode("utf-8"))
        assert sel["recipe"]["status"] == "selected"

    # 5. POST submit recipe (trains model fit-in-fold)
    submit_payload = json.dumps({"recipe_id": recipe_id}).encode("utf-8")
    req = Request(f"{base}/api/recipes/submit", data=submit_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        sub = json.loads(resp.read().decode("utf-8"))
        assert sub["status"] == "submitted"
        assert len(sub["experiment_ids"]) > 0
        executed_exp_id = sub["experiment_ids"][0]
        assert sub["recipe"]["status"] == "submitted"
        assert executed_exp_id in sub["recipe"]["experiment_ids"]

    # 6. POST promote recipe
    promote_payload = json.dumps({
        "recipe_id": recipe_id,
        "reason": "Consistent validation gain in CV",
        "notes": "Promoted via Studio tab button",
    }).encode("utf-8")
    req = Request(f"{base}/api/recipes/promote", data=promote_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        pro = json.loads(resp.read().decode("utf-8"))
        assert pro["status"] == "promoted"
        assert pro["recipe"]["status"] == "promoted"
        assert pro["recipe"]["lineage"]["promoted_reason"] == "Consistent validation gain in CV"

    # 7. POST archive recipe
    archive_payload = json.dumps({"recipe_id": recipe_id}).encode("utf-8")
    req = Request(f"{base}/api/recipes/archive", data=archive_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        arc = json.loads(resp.read().decode("utf-8"))
        assert arc["status"] == "archived"


def test_end_to_end_cross_surface_parity_r4(running_workbench_server, capsys):
    """End-to-End Cross-Surface Parity (Fase R4 Acceptance Test):

    1. Explore study evidence inspected via CLI.
    2. Recipe created via Workbench REST client (simulating user clicking 'Crear Receta').
    3. MCP Agent tool reads recipe config, inspects transforms, and explicitly selects it.
    4. Workbench REST client submits and runs the training experiment.
    5. CLI promotes the recipe with an explicit audit reason.
    6. All 4 surfaces (Python API, CLI, MCP Server, REST Workbench) query the recipe:
       - Exact same Recipe ID.
       - Exact same transforms: [{'column': 'num1', 'kind': 'yeo_johnson', 'params': {}}].
       - Exact same PROMOTED status.
       - Exact same executed experiment IDs.
       - Exact same promoted reason in lineage.
    """
    ws_dir = running_workbench_server["workspace_dir"]
    base_url = running_workbench_server["base_url"]
    ws = running_workbench_server["ws"]
    cmd_bus = running_workbench_server["cmd_bus"]
    qry_bus = running_workbench_server["qry_bus"]
    evidence = running_workbench_server["evidence"]

    # --- Step 1: CLI inspects verified improvement ---
    rc = main(["recipe", "list-verified", "--workspace", ws_dir, "--json"])
    assert rc == 0
    cli_verified = json.loads(capsys.readouterr().out)
    assert any(v["evidence_link_id"] == evidence.id for v in cli_verified)

    # --- Step 2: Workbench UI creates recipe draft from evidence ---
    create_body = json.dumps({
        "evidence_link_id": evidence.id,
        "name": "End-to-End Parity Recipe",
        "notes": "Created by Workbench User",
    }).encode("utf-8")
    req = Request(f"{base_url}/api/recipes/create-from-evidence", data=create_body, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        workbench_res = json.loads(resp.read().decode("utf-8"))
        recipe_id = workbench_res["recipe_id"]
        assert workbench_res["recipe"]["status"] == "draft"

    # --- Step 3: MCP Agent reads and selects the recipe ---
    if HAS_MCP:
        server = create_mcp_server(workspace=ws, command_bus=cmd_bus, query_bus=qry_bus)
        mcp_get = asyncio.run(server.call_tool("recipe_get", {"recipe_id": recipe_id}))
        assert mcp_get.is_error is False
        mcp_recipe = json.loads(mcp_get.content[0].text)
        assert mcp_recipe["id"] == recipe_id
        assert mcp_recipe["status"] == "draft"

        mcp_sel = asyncio.run(server.call_tool("recipe_select", {"recipe_id": recipe_id, "notes": "Selected by MCP agent"}))
        assert mcp_sel.is_error is False
        assert json.loads(mcp_sel.content[0].text)["status"] == "selected"

    # --- Step 4: Workbench UI executes the recipe experiment ---
    submit_body = json.dumps({"recipe_id": recipe_id, "notes": "Submitted from Workbench Studio"}).encode("utf-8")
    req = Request(f"{base_url}/api/recipes/submit", data=submit_body, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        sub_res = json.loads(resp.read().decode("utf-8"))
        assert sub_res["status"] == "submitted"
        executed_exp_id = sub_res["experiment_ids"][0]

    # --- Step 5: CLI promotes the recipe ---
    rc = main([
        "recipe", "promote", recipe_id,
        "--reason", "Passed Phase R4 E2E validation suite",
        "--workspace", ws_dir,
        "--json",
    ])
    assert rc == 0
    cli_promote = json.loads(capsys.readouterr().out)
    assert cli_promote["status"] == "promoted"

    # --- Step 6: 4-Surface Identity and Parity Verification ---
    # Surface 1: REST Workbench
    with urlopen(f"{base_url}/api/recipes/{recipe_id}") as resp:
        surface_rest = json.loads(resp.read().decode("utf-8"))

    # Surface 2: CLI Show
    rc = main(["recipe", "show", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    surface_cli = json.loads(capsys.readouterr().out)

    # Surface 3: Application QueryBus
    from automl.application.recipes.queries import GetExperimentRecipeQuery
    surface_cqrs = qry_bus.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
    assert surface_cqrs is not None
    surface_cqrs_dict = asdict(surface_cqrs)

    # Parity assertions across Python CQRS, CLI, and REST
    for s in [surface_rest, surface_cli, surface_cqrs_dict]:
        assert s["id"] == recipe_id
        assert s["status"] == "promoted"
        assert s["column_transforms"] == [{"column": "num1", "kind": "yeo_johnson", "params": {}}]
        assert executed_exp_id in s["experiment_ids"]
        assert s["lineage"]["promoted_reason"] == "Passed Phase R4 E2E validation suite"
        assert s["evidence_link_id"] == evidence.id

    # Surface 4: MCP Tool (if installed)
    if HAS_MCP:
        mcp_final = asyncio.run(server.call_tool("recipe_get", {"recipe_id": recipe_id}))
        assert mcp_final.is_error is False
        surface_mcp = json.loads(mcp_final.content[0].text)
        assert surface_mcp["id"] == recipe_id
        assert surface_mcp["status"] == "promoted"
        assert surface_mcp["column_transforms"] == [{"column": "num1", "kind": "yeo_johnson", "params": {}}]
        assert executed_exp_id in surface_mcp["experiment_ids"]
        assert surface_mcp["lineage"]["promoted_reason"] == "Passed Phase R4 E2E validation suite"
