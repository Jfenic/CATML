"""CATML Explore — Statistical and Exploratory Engine.

Encapsulates mathematical, distribution, correlation, and hypothesis testing
algorithms for exploratory studies.
"""
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer
from automl.engine.analysis.distribution_diagnostics import (
    diagnose_univariate_distributions,
    diagnose_multivariate_outliers,
)
from automl.engine.analysis.association_metrics import (
    analyze_numeric_associations,
    analyze_categorical_associations,
    compute_cramers_v,
    compute_fisher_ci,
)
from automl.engine.analysis.hypothesis_testing import (
    analyze_target_hypotheses,
    apply_benjamini_hochberg_correction,
    compute_cohens_d,
    compute_eta_squared,
)

__all__ = [
    "StatisticalAnalyzer",
    "diagnose_univariate_distributions",
    "diagnose_multivariate_outliers",
    "analyze_numeric_associations",
    "analyze_categorical_associations",
    "compute_cramers_v",
    "compute_fisher_ci",
    "analyze_target_hypotheses",
    "apply_benjamini_hochberg_correction",
    "compute_cohens_d",
    "compute_eta_squared",
]
