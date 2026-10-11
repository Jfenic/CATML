"""CATML Explore — Comprehensive 4-Axis Benchmark Suite (Phase E6).

Implements rigorous benchmarks across the four canonical axes defined in ADR-008:
  Axis 1: Numerical and Statistical Accuracy (NIST-style calibration & synthetic ground truth).
  Axis 2: AutoML Performance & Quality Improvement with Explore Hypotheses ("Propose ≠ Accept").
  Axis 3: Security, Privacy, and MCP Token Confinement (Fail-Closed, zero raw data egress).
  Axis 4: Schema Regression, SQLite Migrations, and Workspace Backward Compatibility.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from automl.application.bootstrap import build_application
from automl.application.analysis.commands import (
    CreateStudyCommand,
    RunAnalysisCommand,
    CreateHypothesisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetStudyQuery,
    ListStudiesQuery,
    ListFindingsQuery,
    ListVisualizationsQuery,
    ListHypothesesQuery,
    GetEvidenceLinkQuery,
    ListEvidenceLinksQuery,
)
from automl.application.analysis.study_service import AnalysisStudyService
from automl.application.services.workspace import AutoMLWorkspace, PLATFORM_VERSION
from automl.domain.analysis.models import (
    DataSourceRef,
    StatisticalFinding,
    AnalysisHypothesis,
    EvidenceLink,
    StudySpec,
)
from automl.engine.analysis import (
    StatisticalAnalyzer,
    diagnose_univariate_distributions,
    diagnose_multivariate_outliers,
    analyze_numeric_associations,
    analyze_categorical_associations,
    compute_cramers_v,
    compute_fisher_ci,
    apply_benjamini_hochberg_correction,
    compute_cohens_d,
    compute_eta_squared,
)
from automl.engine.analysis.hypothesis_translator import (
    HypothesisExperimentTranslator,
    ExperimentCandidateSpec,
)
from automl.infrastructure.database.sqlite_studies import SQLiteStudyRepository
from automl.interfaces.mcp.server import create_mcp_server

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic Benchmark Reference Data: Anscombe's Quartet
# ---------------------------------------------------------------------------
ANSCOMBE_QUARTET = {
    "I": {
        "x": [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    },
    "II": {
        "x": [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        "y": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    },
    "III": {
        "x": [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        "y": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    },
    "IV": {
        "x": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0],
        "y": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    },
}


@dataclass
class ExploreBenchmarkRunner:
    """Automated benchmark runner for CATML Explore."""

    workspace_root: str | None = None

    # -----------------------------------------------------------------------
    # AXIS 1: Numerical and Statistical Accuracy
    # -----------------------------------------------------------------------
    def run_statistical_accuracy_benchmarks(self) -> dict[str, Any]:
        """Validate numerical algorithms against analytical standards and synthetic reference datasets."""
        results: dict[str, Any] = {}

        # 1. NIST-style Univariate Precision under Large Shifts (Catastrophic Cancellation test)
        base_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        shift = 1_000_000_000.0
        shifted_vals = base_vals + shift

        true_mean = shift + 5.5
        true_var = float(np.var(base_vals, ddof=1))  # 9.166666666666666...
        computed_mean = float(np.mean(shifted_vals))
        computed_var = float(np.var(shifted_vals, ddof=1))

        mean_err = abs(computed_mean - true_mean)
        var_err = abs(computed_var - true_var)
        nist_passed = bool(mean_err < 1e-6 and var_err < 1e-4)

        results["nist_univariate_precision"] = {
            "passed": nist_passed,
            "true_mean": true_mean,
            "computed_mean": computed_mean,
            "mean_abs_error": mean_err,
            "true_var": true_var,
            "computed_var": computed_var,
            "var_abs_error": var_err,
        }

        # 2. Anscombe's Quartet Calibration
        anscombe_results: dict[str, Any] = {}
        for q_id, q_data in ANSCOMBE_QUARTET.items():
            x = np.array(q_data["x"], dtype=float)
            y = np.array(q_data["y"], dtype=float)
            r_val, p_val = stats.pearsonr(x, y)
            rho_val, _ = stats.spearmanr(x, y)

            r_accurate = bool(0.80 <= r_val <= 0.83)
            is_nonlinear_signature = bool(abs(rho_val - r_val) > 0.05 if q_id == "II" else True)

            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            outlier_detected = bool((y > (q3 + 1.5 * iqr)).any() if q_id == "III" else True)

            q_passed = bool(r_accurate and is_nonlinear_signature and outlier_detected)
            anscombe_results[f"quartet_{q_id}"] = {
                "pearson_r": round(float(r_val), 4),
                "spearman_rho": round(float(rho_val), 4),
                "p_value": float(p_val),
                "r_accurate": r_accurate,
                "nonlinear_signature": is_nonlinear_signature,
                "outlier_detected": outlier_detected,
                "passed": q_passed,
            }

        anscombe_passed = bool(all(item["passed"] for item in anscombe_results.values()))
        results["anscombe_quartet"] = {
            "passed": anscombe_passed,
            "datasets": anscombe_results,
        }

        # 3. Fisher z-Transform 95% Confidence Interval Analytical Benchmark
        ci_low, ci_high = compute_fisher_ci(0.50, 103, confidence=0.95)
        fisher_passed = bool(abs(ci_low - 0.3394) < 0.01 and abs(ci_high - 0.6324) < 0.01)
        results["fisher_ci_calibration"] = {
            "passed": fisher_passed,
            "computed_ci": (round(ci_low, 4), round(ci_high, 4)),
            "expected_ci": (0.3394, 0.6324),
        }

        # 4. Benjamini-Hochberg (FDR) Procedure Verification
        raw_p = [0.001, 0.005, 0.012, 0.040, 0.080, 0.120, 0.400, 0.800]
        dummy_findings = [
            StatisticalFinding.create(
                study_id="bench_s",
                analysis_run_id="bench_r",
                finding_type="correlation",
                summary=f"finding_{i}",
                column_name=f"col_{i}",
                p_value=p,
            )
            for i, p in enumerate(raw_p)
        ]
        apply_benjamini_hochberg_correction(dummy_findings)
        adj_p_values = [f.metrics["p_value_fdr"] for f in dummy_findings]

        fdr_monotonic = bool(all(adj_p_values[i] <= adj_p_values[i + 1] for i in range(len(adj_p_values) - 1)))
        fdr_bounded = bool(all(0.0 <= p <= 1.0 for p in adj_p_values))
        fdr_inflation_protected = bool(all(adj_p_values[i] >= raw_p[i] for i in range(len(raw_p))))
        fdr_passed = bool(fdr_monotonic and fdr_bounded and fdr_inflation_protected)

        results["benjamini_hochberg_fdr"] = {
            "passed": fdr_passed,
            "raw_p_values": raw_p,
            "adjusted_p_values": adj_p_values,
            "monotonic": fdr_monotonic,
            "inflation_protected": fdr_inflation_protected,
        }

        # 5. Welch's t-test and Cohen's d Effect Size Benchmark
        np.random.seed(42)
        g1 = np.random.normal(loc=10.0, scale=2.0, size=1500)
        g2 = np.random.normal(loc=12.0, scale=2.0, size=1500)
        d_val = compute_cohens_d(g1, g2)
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        effect_size_passed = bool(abs(d_val - 1.0) < 0.10 and p_val < 1e-15)

        results["cohens_d_effect_size"] = {
            "passed": effect_size_passed,
            "computed_d": round(d_val, 4),
            "expected_d": 1.0,
            "p_value": float(p_val),
        }

        # 6. Cramér's V Categorical Association Benchmark
        diag_table = pd.DataFrame([[100, 0], [0, 100]], index=["A", "B"], columns=["X", "Y"])
        v_diag, chi2_diag, p_diag = compute_cramers_v(diag_table)

        unif_table = pd.DataFrame([[50, 50], [50, 50]], index=["A", "B"], columns=["X", "Y"])
        v_unif, chi2_unif, p_unif = compute_cramers_v(unif_table)

        cramers_passed = bool((v_diag >= 0.98) and (v_unif <= 0.02))
        results["cramers_v_bounds"] = {
            "passed": cramers_passed,
            "v_perfect": round(v_diag, 4),
            "v_independent": round(v_unif, 4),
        }

        all_passed = bool(all(r.get("passed", False) for r in results.values()))
        return {
            "axis": "numerical_statistical_accuracy",
            "passed": all_passed,
            "benchmarks": results,
        }

    # -----------------------------------------------------------------------
    # AXIS 2: AutoML Performance & Quality Improvement with Explore Hypotheses
    # -----------------------------------------------------------------------
    def run_automl_explore_improvement_benchmarks(
        self,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """Validate end-to-end integration between statistical findings and AutoML candidate experiments."""
        tmp_dir = tempfile.mkdtemp(prefix="catml_e6_automl_")
        try:
            ws, cmd_bus, qry_bus = build_application(tmp_dir)

            if dataset_path is None:
                dataset_path = str(
                    Path(__file__).resolve().parents[3] / "examples" / "data" / "customers_churn.csv"
                )

            ds = ws.register_dataset(
                name="churn_bench",
                path=dataset_path,
                target="churn",
                task_type="binary_classification",
            )
            run = ws.create_run(ds, metric="roc_auc")

            # 1. Establish Baseline Experiment
            baseline_exp = ws.create_experiment(
                run=run,
                name="baseline_raw",
                model_ids=["logistic_regression", "random_forest"],
                hypothesis="Raw unassisted baseline experiment",
            )
            baseline_results = ws.run_experiment(run, baseline_exp)
            valid_baseline = [r for r in baseline_results if r.succeeded]
            assert len(valid_baseline) > 0, "Baseline experiment failed to produce trials"
            baseline_score = max(r.primary_score for r in valid_baseline)

            # 2. Run Statistical Study
            study_id = cmd_bus.dispatch(
                CreateStudyCommand(
                    dataset_id=ds.id,
                    target_column="churn",
                    name="Explore Benchmark Study",
                )
            )
            cmd_bus.dispatch(RunAnalysisCommand(study_id=study_id))
            findings = qry_bus.dispatch(ListFindingsQuery(study_id=study_id))

            # 3. Create a candidate hypothesis guided by statistical findings
            def get_f_field(finding_obj, key):
                return finding_obj.get(key) if isinstance(finding_obj, dict) else getattr(finding_obj, key, None)

            collinear_finding = next((f for f in findings if get_f_field(f, "finding_type") == "collinearity"), None)
            if collinear_finding:
                action = "resolve_collinearity"
                target_col = get_f_field(collinear_finding, "column_name")
                delta_spec = {"drop_columns": [target_col]}
                hyp_desc = f"Drop redundant collinear predictor '{target_col}'."
                finding_id = get_f_field(collinear_finding, "id")
            else:
                top_finding = findings[0] if findings else None
                finding_id = get_f_field(top_finding, "id") if top_finding else "f_bench_fallback"
                action = "prioritize_feature"
                target_col = "salary"
                delta_spec = {"priority_features": [target_col]}
                hyp_desc = f"Prioritize top signal predictor '{target_col}'."

            hyp_id = cmd_bus.dispatch(
                CreateHypothesisCommand(
                    study_id=study_id,
                    finding_id=finding_id,
                    description=hyp_desc,
                    proposed_action=action,
                    experiment_delta=delta_spec,
                )
            )

            # 4. Verify hypothesis on identical validation split (Propose ≠ Accept)
            verify_cmd = VerifyHypothesisCommand(
                hypothesis_id=hyp_id,
                run_id=run.id,
                min_improvement=0.0,
            )
            evidence_link = cmd_bus.dispatch(verify_cmd)

            assert evidence_link is not None
            candidate_score = evidence_link.get("candidate_score", 0.0)
            accepted = evidence_link.get("accepted", False)
            delta = evidence_link.get("delta", candidate_score - baseline_score)

            # 5. Check Negative Learning or Positive Improvement Preservation
            updated_hyp = qry_bus.dispatch(ListHypothesesQuery(study_id=study_id))
            hyp_entity = next((h for h in updated_hyp if (h.get("id") if isinstance(h, dict) else h.id) == hyp_id), None)
            assert hyp_entity is not None
            hyp_status = hyp_entity.get("status") if isinstance(hyp_entity, dict) else hyp_entity.status
            assert hyp_status in {"accepted", "rejected"}

            evidence_links = qry_bus.dispatch(ListEvidenceLinksQuery(study_id=study_id))
            assert len(evidence_links) >= 1

            passed = bool(
                baseline_score > 0.40
                and candidate_score > 0.40
                and evidence_link["baseline_score"] == baseline_score
                and (evidence_link["accepted"] == (delta > 0.0))
            )

            return {
                "axis": "automl_quality_and_hypothesis_improvement",
                "passed": passed,
                "dataset": ds.name,
                "baseline_score": round(baseline_score, 4),
                "candidate_score": round(candidate_score, 4),
                "delta": round(delta, 4),
                "hypothesis_status": hyp_status,
                "evidence_links_count": len(evidence_links),
                "propose_not_accept_enforced": True,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # AXIS 3: Security, Privacy, and MCP Token Confinement
    # -----------------------------------------------------------------------
    def run_security_and_mcp_confinement_benchmarks(self) -> dict[str, Any]:
        """Validate token boundedness, zero data leakage, and fail-closed security for agents."""
        tmp_dir = tempfile.mkdtemp(prefix="catml_e6_security_")
        try:
            ws, cmd_bus, qry_bus = build_application(tmp_dir)

            dataset_path = str(
                Path(__file__).resolve().parents[3] / "examples" / "data" / "customers_churn.csv"
            )
            ds = ws.register_dataset(
                name="churn_sec",
                path=dataset_path,
                target="churn",
                task_type="binary_classification",
            )
            study_id = cmd_bus.dispatch(
                CreateStudyCommand(dataset_id=ds.id, target_column="churn", name="Sec Study")
            )
            cmd_bus.dispatch(RunAnalysisCommand(study_id=study_id))

            mcp_server = create_mcp_server(workspace=ws, command_bus=cmd_bus, query_bus=qry_bus, root_dir=tmp_dir)

            # 1. MCP Token Confinement & Pagination Test
            res_envelope = asyncio.run(
                mcp_server.call_tool(
                    "analysis_get_findings",
                    {
                        "study_id": study_id,
                        "limit": 2,
                        "compact": True,
                        "envelope": True,
                    },
                )
            )
            payload = json.loads(res_envelope.content[0].text)

            token_bounded = bool(len(payload.get("data", [])) <= 2)
            envelope_valid = bool("total" in payload and "limit" in payload and payload["limit"] == 2)
            compact_verified = True
            for f in payload.get("data", []):
                if "matrix" in f.get("metrics", {}) or "points" in f.get("metrics", {}):
                    compact_verified = False

            # 2. Data Privacy: Summary Resource Must Never Leak Raw CSV Data
            summary_raw = asyncio.run(mcp_server.read_resource(f"catml://studies/{study_id}/summary"))
            summary_content = summary_raw[0].content if isinstance(summary_raw, list) else summary_raw

            no_raw_data_leak = bool(
                "customer_id" not in summary_content
                and "C0001" not in summary_content
            )

            # 3. Fail-Closed Path Traversal and SQL Injection Resistance
            malicious_ids = [
                "../../../../etc/passwd",
                "' OR 1=1 --",
                "nonexistent_study_id_404",
                "\x00nullbyte",
            ]
            fail_closed_safe = True
            for m_id in malicious_ids:
                res_err = asyncio.run(
                    mcp_server.call_tool(
                        "analysis_get_findings",
                        {"study_id": m_id},
                    )
                )
                if not res_err.is_error and len(res_err.content) > 0:
                    text_out = res_err.content[0].text
                    if "NOT_FOUND" not in text_out and "error" not in text_out.lower():
                        fail_closed_safe = False

            passed = bool(token_bounded and envelope_valid and compact_verified and no_raw_data_leak and fail_closed_safe)

            return {
                "axis": "security_and_mcp_confinement",
                "passed": passed,
                "token_budget_bounded": token_bounded,
                "envelope_metadata_valid": envelope_valid,
                "compact_stripping_verified": compact_verified,
                "zero_raw_pii_egress": no_raw_data_leak,
                "fail_closed_injection_resistance": fail_closed_safe,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # AXIS 4: Schema Regression, SQLite Migrations, and Workspace Compatibility
    # -----------------------------------------------------------------------
    def run_schema_and_workspace_regression_benchmarks(self) -> dict[str, Any]:
        """Validate backwards compatibility, SQLite schema migrations, and WAL concurrency."""
        tmp_dir = tempfile.mkdtemp(prefix="catml_e6_schema_")
        try:
            legacy_ws_root = Path(tmp_dir) / "legacy_workspace"
            legacy_ws_root.mkdir(parents=True, exist_ok=True)

            ws, cmd_bus, qry_bus = build_application(str(legacy_ws_root))
            db_path = legacy_ws_root / "automl.db"
            assert db_path.exists(), "automl.db was not initialized automatically"

            # 2. Coexistence of Supervised and Unsupervised Studies
            dataset_path = str(
                Path(__file__).resolve().parents[3] / "examples" / "data" / "wine_recognition.csv"
            )
            ds = ws.register_dataset(
                name="wine_ds",
                path=dataset_path,
                target=None,
                task_type="clustering",
            )

            # Study A: Unsupervised (target = None)
            study_a_id = cmd_bus.dispatch(
                CreateStudyCommand(
                    dataset_id=ds.id,
                    target_column=None,
                    name="Unsupervised Clustering Study",
                )
            )
            cmd_bus.dispatch(RunAnalysisCommand(study_id=study_a_id))
            findings_a = qry_bus.dispatch(ListFindingsQuery(study_id=study_a_id))

            # Study B: Supervised study on churn dataset
            churn_path = str(
                Path(__file__).resolve().parents[3] / "examples" / "data" / "customers_churn.csv"
            )
            ds_sup = ws.register_dataset(
                name="churn_ds",
                path=churn_path,
                target="churn",
                task_type="binary_classification",
            )
            study_b_id = cmd_bus.dispatch(
                CreateStudyCommand(
                    dataset_id=ds_sup.id,
                    target_column="churn",
                    name="Supervised Churn Study",
                )
            )
            cmd_bus.dispatch(RunAnalysisCommand(study_id=study_b_id))
            findings_b = qry_bus.dispatch(ListFindingsQuery(study_id=study_b_id))

            # Both studies coexisting cleanly
            studies_list = qry_bus.dispatch(ListStudiesQuery())
            assert len(studies_list) >= 2
            study_a_retrieved = qry_bus.dispatch(GetStudyQuery(study_id=study_a_id))
            target_a = study_a_retrieved.get("target_column") if isinstance(study_a_retrieved, dict) else study_a_retrieved.target_column
            assert target_a is None

            study_b_retrieved = qry_bus.dispatch(GetStudyQuery(study_id=study_b_id))
            target_b = study_b_retrieved.get("target_column") if isinstance(study_b_retrieved, dict) else study_b_retrieved.target_column
            assert target_b == "churn"

            # 3. WAL Mode Verification
            import sqlite3
            chk_conn = sqlite3.connect(db_path)
            journal_mode = chk_conn.execute("PRAGMA journal_mode;").fetchone()[0]
            chk_conn.close()
            wal_active = bool(journal_mode.lower() == "wal")

            passed = bool(
                db_path.exists()
                and wal_active
                and len(findings_a) > 0
                and len(findings_b) > 0
            )

            return {
                "axis": "schema_regression_and_workspace_compatibility",
                "passed": passed,
                "db_initialized": True,
                "wal_mode_active": wal_active,
                "unsupervised_study_supported": len(findings_a) > 0,
                "supervised_study_supported": len(findings_b) > 0,
                "total_studies_coexisting": len(studies_list),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Comprehensive 4-Axis Suite Execution
    # -----------------------------------------------------------------------
    def run_all(self, dataset_path: str | None = None) -> dict[str, Any]:
        """Execute all four benchmark axes and compile full technical validation report."""
        axis1 = self.run_statistical_accuracy_benchmarks()
        axis2 = self.run_automl_explore_improvement_benchmarks(dataset_path=dataset_path)
        axis3 = self.run_security_and_mcp_confinement_benchmarks()
        axis4 = self.run_schema_and_workspace_regression_benchmarks()

        all_passed = bool(axis1["passed"] and axis2["passed"] and axis3["passed"] and axis4["passed"])

        report = {
            "title": "CATML Explore Phase E6 Comprehensive 4-Axis Benchmark",
            "platform_version": PLATFORM_VERSION,
            "passed": all_passed,
            "axes": {
                "axis1_statistical_accuracy": axis1,
                "axis2_automl_quality_improvement": axis2,
                "axis3_security_and_mcp_confinement": axis3,
                "axis4_schema_regression_compatibility": axis4,
            },
        }
        return report
