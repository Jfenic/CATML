"""Test suite for CATML Explore Phase E6 — Comprehensive 4-Axis Benchmark Suite.

Verifies:
  1. Axis 1: Numerical and statistical accuracy (NIST-style large shift, Anscombe Quartet, Fisher CI, FDR).
  2. Axis 2: AutoML performance with hypothesis verification (Propose ≠ Accept, EvidenceLink, deltas).
  3. Axis 3: Security, privacy, and MCP token confinement (fail-closed, compact stripping, zero raw PII).
  4. Axis 4: Schema regression, legacy workspace compatibility, and WAL concurrency.
  5. Full run_all() report aggregation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from automl.benchmarks.explore_benchmarks import (
    ANSCOMBE_QUARTET,
    ExploreBenchmarkRunner,
)
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer


def test_explore_benchmark_axis1_statistical_accuracy() -> None:
    runner = ExploreBenchmarkRunner()
    res = runner.run_statistical_accuracy_benchmarks()

    assert res["axis"] == "numerical_statistical_accuracy"
    assert res["passed"] is True

    benchmarks = res["benchmarks"]
    # Check NIST univariate precision
    nist = benchmarks["nist_univariate_precision"]
    assert nist["passed"] is True
    assert nist["mean_abs_error"] < 1e-6
    assert nist["var_abs_error"] < 1e-4

    # Check Anscombe Quartet calibration
    anscombe = benchmarks["anscombe_quartet"]
    assert anscombe["passed"] is True
    for q_id, q_res in anscombe["datasets"].items():
        assert q_res["passed"] is True
        assert 0.80 <= q_res["pearson_r"] <= 0.83

    # Check Fisher CI calibration
    fisher = benchmarks["fisher_ci_calibration"]
    assert fisher["passed"] is True
    low, high = fisher["computed_ci"]
    assert 0.33 <= low <= 0.35
    assert 0.62 <= high <= 0.64

    # Check Benjamini-Hochberg FDR
    fdr = benchmarks["benjamini_hochberg_fdr"]
    assert fdr["passed"] is True
    assert fdr["monotonic"] is True
    assert fdr["inflation_protected"] is True

    # Check Cohen's d
    cohen = benchmarks["cohens_d_effect_size"]
    assert cohen["passed"] is True
    assert abs(cohen["computed_d"] - 1.0) < 0.10

    # Check Cramér's V
    cramer = benchmarks["cramers_v_bounds"]
    assert cramer["passed"] is True
    assert cramer["v_perfect"] >= 0.98
    assert cramer["v_independent"] <= 0.02


def test_explore_benchmark_axis2_automl_quality_and_hypothesis() -> None:
    runner = ExploreBenchmarkRunner()
    res = runner.run_automl_explore_improvement_benchmarks()

    assert res["axis"] == "automl_quality_and_hypothesis_improvement"
    assert res["passed"] is True
    assert res["baseline_score"] > 0.40
    assert res["candidate_score"] > 0.40
    assert res["hypothesis_status"] in {"accepted", "rejected"}
    assert res["evidence_links_count"] >= 1
    assert res["propose_not_accept_enforced"] is True


def test_explore_benchmark_axis3_security_and_mcp_confinement() -> None:
    runner = ExploreBenchmarkRunner()
    res = runner.run_security_and_mcp_confinement_benchmarks()

    assert res["axis"] == "security_and_mcp_confinement"
    assert res["passed"] is True
    assert res["token_budget_bounded"] is True
    assert res["envelope_metadata_valid"] is True
    assert res["compact_stripping_verified"] is True
    assert res["zero_raw_pii_egress"] is True
    assert res["fail_closed_injection_resistance"] is True


def test_explore_benchmark_axis4_schema_regression_compatibility() -> None:
    runner = ExploreBenchmarkRunner()
    res = runner.run_schema_and_workspace_regression_benchmarks()

    assert res["axis"] == "schema_regression_and_workspace_compatibility"
    assert res["passed"] is True
    assert res["db_initialized"] is True
    assert res["wal_mode_active"] is True
    assert res["unsupervised_study_supported"] is True
    assert res["supervised_study_supported"] is True
    assert res["total_studies_coexisting"] >= 2


def test_explore_benchmark_run_all() -> None:
    runner = ExploreBenchmarkRunner()
    report = runner.run_all()

    assert report["passed"] is True
    assert "axes" in report
    axes = report["axes"]
    assert "axis1_statistical_accuracy" in axes
    assert "axis2_automl_quality_improvement" in axes
    assert "axis3_security_and_mcp_confinement" in axes
    assert "axis4_schema_regression_compatibility" in axes

    assert axes["axis1_statistical_accuracy"]["passed"] is True
    assert axes["axis2_automl_quality_improvement"]["passed"] is True
    assert axes["axis3_security_and_mcp_confinement"]["passed"] is True
    assert axes["axis4_schema_regression_compatibility"]["passed"] is True


def test_anscombe_quartet_direct_statistical_analyzer(tmp_path: Path) -> None:
    q2 = ANSCOMBE_QUARTET["II"]
    df = pd.DataFrame({"x": q2["x"], "y": q2["y"]})
    csv_file = tmp_path / "anscombe_ii.csv"
    df.to_csv(csv_file, index=False)

    analyzer = StatisticalAnalyzer()
    findings, viz_specs, hypotheses = analyzer.analyze(
        data_source_path=str(csv_file),
        target_column=None,  # Unsupervised pairwise associations between numeric columns
    )

    assert len(findings) > 0
    correl_findings = [f for f in findings if f.finding_type in {"high_collinearity", "monotonic_nonlinear_relationship", "correlation", "association"}]
    assert len(correl_findings) > 0
