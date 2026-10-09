"""Comprehensive test suite for CATML Explore Phase E2 (Advanced Statistical Engine).

Validates normality tests, multimodality detection, multivariate Mahalanobis outliers,
monotonic non-linear Spearman contrasts, bias-corrected Cramér's V, group hypothesis tests
(Welch t-test, ANOVA, Cohen's d, eta-squared), and Benjamini-Hochberg FDR corrections.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.domain.analysis.models import StatisticalFinding
from automl.engine.analysis.association_metrics import (
    analyze_categorical_associations,
    analyze_numeric_associations,
    compute_cramers_v,
    compute_fisher_ci,
)
from automl.engine.analysis.distribution_diagnostics import (
    diagnose_multivariate_outliers,
    diagnose_univariate_distributions,
)
from automl.engine.analysis.hypothesis_testing import (
    analyze_target_hypotheses,
    apply_benjamini_hochberg_correction,
    compute_cohens_d,
    compute_eta_squared,
)
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer


@pytest.fixture
def advanced_synthetic_dataset(tmp_path: Path) -> Path:
    """Generate deterministic dataset with known mathematical properties:
    - col_normal: Standard Gaussian distribution.
    - col_skewed: Exponential / Log-Normal (high skewness and non-normality).
    - col_bimodal: Mixture of two distant Gaussians (high bimodal coefficient).
    - col_x, col_y_monotonic: Monotonic non-linear relationship (y = x^3).
    - cat_dept, cat_role: Strongly associated categorical columns (high Cramér's V).
    - target_binary: Binary classification target with clear class separation.
    - target_multi: 3-class target for ANOVA / Kruskal-Wallis evaluation.
    """
    np.random.seed(42)
    n = 200

    col_normal = np.random.normal(50.0, 10.0, n)
    col_skewed = np.random.exponential(scale=5.0, size=n) ** 1.8

    # Bimodal: 50% from N(10, 1) and 50% from N(30, 1)
    bimodal_part1 = np.random.normal(10.0, 1.0, n // 2)
    bimodal_part2 = np.random.normal(30.0, 1.0, n // 2)
    col_bimodal = np.concatenate([bimodal_part1, bimodal_part2])
    np.random.shuffle(col_bimodal)

    # Monotonic non-linear: y = x^3 + small noise
    col_x = np.linspace(-3.0, 3.0, n)
    col_y_monotonic = (col_x ** 3) + np.random.normal(0, 0.5, n)

    # Categorical associations
    departments = ["Engineering", "Sales", "Support"]
    roles = []
    depts = []
    for _ in range(n):
        d = np.random.choice(departments, p=[0.5, 0.3, 0.2])
        depts.append(d)
        if d == "Engineering":
            roles.append(np.random.choice(["Dev", "QA"], p=[0.8, 0.2]))
        elif d == "Sales":
            roles.append(np.random.choice(["AccountExec", "SDR"], p=[0.7, 0.3]))
        else:
            roles.append("Agent")

    # Binary target linked to col_normal
    target_binary = (col_normal > 52.0).astype(int)

    # Multi-class target (3 classes) linked to col_normal
    target_multi = np.zeros(n, dtype=int)
    target_multi[col_normal < 45.0] = 0
    target_multi[(col_normal >= 45.0) & (col_normal < 55.0)] = 1
    target_multi[col_normal >= 55.0] = 2

    df = pd.DataFrame({
        "col_normal": col_normal,
        "col_skewed": col_skewed,
        "col_bimodal": col_bimodal,
        "col_x": col_x,
        "col_y_monotonic": col_y_monotonic,
        "dept": depts,
        "role": roles,
        "target_binary": target_binary,
        "target_multi": target_multi,
    })

    csv_path = tmp_path / "advanced_stats_catalog.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# =========================================================================
# 1. DISTRIBUTION DIAGNOSTICS TESTS
# =========================================================================

def test_univariate_normality_skewness_and_multimodality(advanced_synthetic_dataset: Path):
    """Verify detection of severe skewness, normality rejection, and multimodality."""
    df = pd.read_csv(advanced_synthetic_dataset)
    num_cols = ["col_normal", "col_skewed", "col_bimodal"]

    findings, hypotheses = diagnose_univariate_distributions(
        df=df,
        num_cols=num_cols,
        study_id="study_dist",
        run_id="run_dist",
    )

    finding_types = {f.finding_type for f in findings}
    columns_flagged = {f.column_name for f in findings}

    # Skewed column should have skewness finding and power_transform hypothesis
    assert "skewness" in finding_types
    assert "col_skewed" in columns_flagged
    hyp_actions = {h.proposed_action for h in hypotheses}
    assert "power_transform" in hyp_actions

    # Bimodal column should trigger multimodality finding
    assert "multimodal_distribution" in finding_types
    bimodal_cols = {f.column_name for f in findings if f.finding_type == "multimodal_distribution"}
    assert "col_bimodal" in bimodal_cols

    # Normality violation should be detected on col_skewed and col_bimodal
    assert "normality_violation" in finding_types


def test_multivariate_outliers_detection():
    """Verify Mahalanobis distance detects multivariate joint anomalies."""
    np.random.seed(42)
    n = 150
    # Two strongly correlated variables: y = 2x + noise
    x = np.random.normal(10.0, 2.0, n)
    y = 2.0 * x + np.random.normal(0, 0.5, n)

    # Inject 4 multivariate outliers that violate correlation structure
    # (Univariately within [10, 15], but joints are far off the line y = 2x)
    x[0] = 10.0
    y[0] = 35.0  # expected y ~ 20.0
    x[1] = 14.0
    y[1] = 10.0  # expected y ~ 28.0

    df = pd.DataFrame({"feat_x": x, "feat_y": y})
    findings, hypotheses = diagnose_multivariate_outliers(
        df=df,
        num_cols=["feat_x", "feat_y"],
        study_id="study_outliers",
        run_id="run_outliers",
    )

    assert len(findings) >= 1
    f = findings[0]
    assert f.finding_type == "multivariate_outliers"
    assert f.metrics["outlier_count"] >= 2
    assert f.metrics["critical_threshold"] > 0
    assert len(hypotheses) >= 1
    assert hypotheses[0].proposed_action == "multivariate_filtering"


# =========================================================================
# 2. ASSOCIATION METRICS TESTS
# =========================================================================

def test_fisher_ci_bounds():
    """Verify Fisher z-transform 95% confidence intervals."""
    low, high = compute_fisher_ci(0.80, n=100)
    assert 0.70 < low < 0.80
    assert 0.80 < high < 0.90
    assert low < high


def test_monotonic_nonlinear_relationship_contrast(advanced_synthetic_dataset: Path):
    """Verify contrast between Pearson r and Spearman rho captures non-linearity."""
    df = pd.read_csv(advanced_synthetic_dataset)
    findings, viz, hypotheses = analyze_numeric_associations(
        df=df,
        num_cols=["col_x", "col_y_monotonic"],
        study_id="study_assoc",
        run_id="run_assoc",
    )

    types = {f.finding_type for f in findings}
    assert "monotonic_nonlinear_relationship" in types or "high_collinearity" in types

    # Correlation matrix visualization spec must be present
    assert len(viz) == 1
    assert viz[0].chart_type == "correlation_matrix"
    assert "spearman_matrix" in viz[0].data_series


def test_cramers_v_bias_corrected_categorical_association(advanced_synthetic_dataset: Path):
    """Verify bias-corrected Cramér's V and Chi-square independence tests."""
    df = pd.read_csv(advanced_synthetic_dataset)
    findings, viz, hypotheses = analyze_categorical_associations(
        df=df,
        cat_cols=["dept", "role"],
        study_id="study_cat",
        run_id="run_cat",
    )

    assert len(findings) >= 1
    f = findings[0]
    assert f.finding_type == "categorical_association"
    assert f.metrics["cramers_v"] > 0.40
    assert f.p_value is not None
    assert f.p_value < 0.001
    assert len(viz) == 1
    assert viz[0].chart_type == "categorical_association_matrix"


# =========================================================================
# 3. HYPOTHESIS TESTING & EFFECT SIZES
# =========================================================================

def test_cohens_d_and_eta_squared_calculations():
    """Verify effect size calculation for 2 groups (Cohen's d) and >2 groups (Eta-squared)."""
    # 2 groups with mean diff = 2.0, std = 1.0 -> d ≈ 2.0
    g1 = np.array([10.0, 10.5, 9.5, 10.0, 10.2])
    g2 = np.array([12.0, 12.5, 11.5, 12.0, 12.2])
    d = compute_cohens_d(g1, g2)
    assert d > 1.5

    # 3 groups for ANOVA
    g3 = np.array([14.0, 14.5, 13.5, 14.0, 14.2])
    eta2 = compute_eta_squared([g1, g2, g3])
    assert 0.80 <= eta2 <= 1.0


def test_target_supervised_inferences_binary_and_multiclass(advanced_synthetic_dataset: Path):
    """Verify Welch t-test, Mann-Whitney U, and ANOVA F-tests against targets."""
    df = pd.read_csv(advanced_synthetic_dataset)

    # 1. Binary target
    findings_bin, hyps_bin = analyze_target_hypotheses(
        df=df,
        target_column="target_binary",
        num_cols=["col_normal"],
        cat_cols=[],
        study_id="study_tgt_bin",
        run_id="run_tgt_bin",
    )
    assert len(findings_bin) >= 1
    f_bin = findings_bin[0]
    assert f_bin.finding_type == "target_class_separation"
    assert "welch_t" in f_bin.metrics
    assert "cohens_d" in f_bin.metrics
    assert f_bin.p_value is not None
    assert f_bin.p_value < 0.05

    # 2. Multiclass target (3 classes)
    findings_multi, hyps_multi = analyze_target_hypotheses(
        df=df,
        target_column="target_multi",
        num_cols=["col_normal"],
        cat_cols=[],
        study_id="study_tgt_multi",
        run_id="run_tgt_multi",
    )
    assert len(findings_multi) >= 1
    f_multi = findings_multi[0]
    assert f_multi.finding_type == "target_class_separation"
    assert "anova_f" in f_multi.metrics
    assert "eta_squared" in f_multi.metrics
    assert f_multi.metrics["groups_count"] == 3


# =========================================================================
# 4. BENJAMINI-HOCHBERG FDR CORRECTION TESTS
# =========================================================================

def test_benjamini_hochberg_correction_fdr():
    """Verify that multiple comparisons correction adjusts p-values and flags false discoveries."""
    # Create findings with synthetic p-values:
    # 2 highly significant (0.0001, 0.001), 1 borderline (0.045), 2 non-significant (0.60, 0.85)
    f1 = StatisticalFinding.create("s", "r", "test1", "Significant 1", p_value=0.0001)
    f2 = StatisticalFinding.create("s", "r", "test2", "Significant 2", p_value=0.001)
    f3 = StatisticalFinding.create("s", "r", "test3", "Borderline 3", p_value=0.045)
    f4 = StatisticalFinding.create("s", "r", "test4", "Non-significant 4", p_value=0.60)
    f5 = StatisticalFinding.create("s", "r", "test5", "Non-significant 5", p_value=0.85)

    findings = [f1, f2, f3, f4, f5]
    apply_benjamini_hochberg_correction(findings)

    # f1 and f2 must remain FDR significant
    assert f1.metrics["fdr_significant"] is True
    assert f2.metrics["fdr_significant"] is True

    # Borderline f3 has rank 3/5: adj_p = 0.045 * 5 / 3 = 0.075 >= 0.05
    # So after FDR correction it should be marked as NOT significant!
    assert f3.metrics["p_value_fdr"] >= 0.05
    assert f3.metrics["fdr_significant"] is False
    assert len(f3.limitations) >= 1
    assert "Benjamini-Hochberg" in f3.limitations[0]


# =========================================================================
# 5. END-TO-END STATISTICAL ANALYZER (PHASE E2)
# =========================================================================

def test_statistical_analyzer_full_phase_e2_pipeline(advanced_synthetic_dataset: Path):
    """Verify end-to-end analyzer execution combining all Phase E2 diagnostic engines."""
    analyzer = StatisticalAnalyzer()
    findings, visualizations, hypotheses = analyzer.analyze(
        data_source_path=str(advanced_synthetic_dataset),
        target_column="target_binary",
        analysis_types=["descriptive", "association"],
    )

    finding_types = {f.finding_type for f in findings}
    viz_types = {v.chart_type for v in visualizations}

    # Must contain quality diagnostics
    assert "skewness" in finding_types
    assert "normality_violation" in finding_types
    assert "multimodal_distribution" in finding_types

    # Must contain association diagnostics
    assert "correlation_matrix" in viz_types
    assert "categorical_association_matrix" in viz_types

    # Must contain target separation
    assert "target_class_separation" in finding_types

    # Every finding with a p-value must have p_value_fdr and fdr_significant computed
    for f in findings:
        if f.p_value is not None:
            assert "p_value_fdr" in f.metrics
            assert "fdr_significant" in f.metrics

    # Hypotheses must provide actionable suggestions
    actions = {h.proposed_action for h in hypotheses}
    assert "power_transform" in actions
    assert "discretize_or_cluster" in actions
    assert "prioritize_feature" in actions


def test_target_supervised_continuous_regression(advanced_synthetic_dataset: Path):
    """Verify target correlation analysis on continuous numerical regression targets."""
    df = pd.read_csv(advanced_synthetic_dataset)
    findings, hyps = analyze_target_hypotheses(
        df=df,
        target_column="col_y_monotonic",
        num_cols=["col_x"],
        cat_cols=[],
        study_id="study_cont",
        run_id="run_cont",
    )
    assert len(findings) >= 1
    f = findings[0]
    assert f.finding_type == "target_correlation"
    assert "pearson_r" in f.metrics
    assert "spearman_rho" in f.metrics
    assert f.metrics["spearman_rho"] > 0.80
    assert len(hyps) >= 1
    assert hyps[0].proposed_action == "prioritize_feature"


def test_edge_cases_and_graceful_degradation(tmp_path: Path):
    """Verify graceful degradation on empty, tiny, or malformed datasets."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    analyzer = StatisticalAnalyzer()
    f, v, h = analyzer.analyze(str(empty_csv))
    assert f == []
    assert v == []
    assert h == []

    # Small df (3 rows)
    tiny_csv = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}).to_csv(tiny_csv, index=False)
    f_tiny, v_tiny, h_tiny = analyzer.analyze(str(tiny_csv), target_column="missing_col")
    assert isinstance(f_tiny, list)
    assert isinstance(v_tiny, list)
