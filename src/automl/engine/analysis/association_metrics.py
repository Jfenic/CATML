"""Bivariate linear, monotonic, and categorical association analysis for CATML Explore."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression

from automl.domain.analysis.models import (
    AnalysisHypothesis,
    StatisticalFinding,
    VisualizationSpec,
)

logger = logging.getLogger(__name__)


def compute_fisher_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Fisher z-transform confidence interval for Pearson correlation."""
    if n <= 3 or abs(r) >= 1.0:
        return (r, r)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1.0 + confidence) / 2.0)
    z_low = z - z_crit * se
    z_high = z + z_crit * se
    return (float(np.tanh(z_low)), float(np.tanh(z_high)))


def compute_cramers_v(table: pd.DataFrame) -> tuple[float, float, float]:
    """Compute bias-corrected Cramér's V, Chi-square statistic and p-value.
    
    Returns (v_corr, chi2, p_val).
    """
    n = float(table.values.sum())
    r, k = table.shape
    if n <= 1 or r <= 1 or k <= 1:
        return 0.0, 0.0, 1.0

    chi2, p_val, _, _ = stats.chi2_contingency(table)
    phi2 = chi2 / n

    # Bias correction (Bergsma & Wicher, 2013)
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    min_dim = min(r_corr - 1, k_corr - 1)

    if min_dim <= 0:
        return 0.0, float(chi2), float(p_val)

    v_corr = float(np.sqrt(phi2_corr / min_dim))
    return min(1.0, max(0.0, v_corr)), float(chi2), float(p_val)


def analyze_numeric_associations(
    df: pd.DataFrame,
    num_cols: list[str],
    study_id: str,
    run_id: str,
) -> tuple[list[StatisticalFinding], list[VisualizationSpec], list[AnalysisHypothesis]]:
    """Analyze Pearson, Spearman, and non-linear associations among numeric features."""
    findings: list[StatisticalFinding] = []
    visualizations: list[VisualizationSpec] = []
    hypotheses: list[AnalysisHypothesis] = []

    if len(num_cols) < 2:
        return findings, visualizations, hypotheses

    # Limit maximum columns for n x n pair evaluations to 20 highest variance features
    cols_to_eval = num_cols
    if len(num_cols) > 20:
        vars_series = df[num_cols].var().fillna(0)
        cols_to_eval = list(vars_series.nlargest(20).index)

    # 1. Compute Pearson and Spearman matrices
    clean_df = df[cols_to_eval]
    pearson_df = clean_df.corr(method="pearson").round(3)
    spearman_df = clean_df.corr(method="spearman").round(3)

    # Create Pearson correlation matrix VisualizationSpec
    matrix_data = {
        "columns": cols_to_eval,
        "matrix": [[float(pearson_df.iloc[r, c]) if not np.isnan(pearson_df.iloc[r, c]) else 0.0
                    for c in range(len(cols_to_eval))] for r in range(len(cols_to_eval))],
        "spearman_matrix": [[float(spearman_df.iloc[r, c]) if not np.isnan(spearman_df.iloc[r, c]) else 0.0
                             for c in range(len(cols_to_eval))] for r in range(len(cols_to_eval))],
    }
    visualizations.append(
        VisualizationSpec.create(
            study_id=study_id,
            analysis_run_id=run_id,
            chart_type="correlation_matrix",
            title="Matriz de Correlación Bivariada (Pearson & Spearman)",
            data_series=matrix_data,
            axes_config={"method": "pearson", "range": [-1.0, 1.0]},
        )
    )

    # 2. Evaluate pairwise associations
    n_features = len(cols_to_eval)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            col1 = cols_to_eval[i]
            col2 = cols_to_eval[j]

            valid = (~df[col1].isna()) & (~df[col2].isna())
            n_valid = int(valid.sum())
            if n_valid < 10:
                continue

            s1 = df.loc[valid, col1]
            s2 = df.loc[valid, col2]

            try:
                r_val, p_val = stats.pearsonr(s1, s2)
                rho_val, p_spearman = stats.spearmanr(s1, s2)
                r_float = float(r_val)
                p_float = float(p_val)
                rho_float = float(rho_val)
                ci_low, ci_high = compute_fisher_ci(r_float, n_valid)

                # A) Collinearity (Pearson r >= 0.70)
                if abs(r_float) >= 0.70 and p_float < 0.05:
                    f = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="high_collinearity",
                        summary=(
                            f"Alta colinealidad bivariada entre '{col1}' y '{col2}' "
                            f"(Pearson r={r_float:.3f}, IC95%=[{ci_low:.3f}, {ci_high:.3f}], p={p_float:.4e})."
                        ),
                        column_name=col1,
                        secondary_column=col2,
                        method_name="pearson_correlation",
                        metrics={
                            "pearson_r": round(r_float, 4),
                            "ci_lower": round(ci_low, 4),
                            "ci_upper": round(ci_high, 4),
                            "spearman_rho": round(rho_float, 4),
                            "sample_size": n_valid,
                        },
                        p_value=p_float,
                        effect_size=abs(round(r_float, 3)),
                    )
                    findings.append(f)
                    hyp = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f.id,
                        description=f"Aplicar regularización L1 o eliminar variable redundante entre '{col1}' y '{col2}'.",
                        proposed_action="resolve_collinearity",
                        experiment_delta={"regularization": "l1", "collinear_pair": [col1, col2]},
                    )
                    hypotheses.append(hyp)

                # B) Monotonic Non-linear Relationship (Spearman >> Pearson)
                if abs(rho_float) >= 0.60 and (abs(rho_float) - abs(r_float) >= 0.20 or abs(r_float) < 0.40):
                    f_nonlin = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="monotonic_nonlinear_relationship",
                        summary=(
                            f"Relación monótona no lineal detectada entre '{col1}' y '{col2}': "
                            f"Spearman ρ={rho_float:.3f} significativamente superior a Pearson r={r_float:.3f}."
                        ),
                        column_name=col1,
                        secondary_column=col2,
                        method_name="spearman_rank_contrast",
                        metrics={
                            "spearman_rho": round(rho_float, 4),
                            "pearson_r": round(r_float, 4),
                            "delta_monotonic": round(abs(rho_float) - abs(r_float), 4),
                        },
                        p_value=float(p_spearman),
                        effect_size=abs(round(rho_float, 3)),
                    )
                    findings.append(f_nonlin)
                    hyp_nonlin = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f_nonlin.id,
                        description=f"Modelos basados en árboles o splines para capturar relación no lineal entre '{col1}' y '{col2}'.",
                        proposed_action="nonlinear_transform_or_trees",
                        experiment_delta={"pair": [col1, col2], "suggested_models": ["lightgbm", "xgboost"]},
                    )
                    hypotheses.append(hyp_nonlin)

            except Exception as exc:
                logger.debug("Correlation check failed between %s and %s: %s", col1, col2, exc)

    return findings, visualizations, hypotheses


def analyze_categorical_associations(
    df: pd.DataFrame,
    cat_cols: list[str],
    study_id: str,
    run_id: str,
) -> tuple[list[StatisticalFinding], list[VisualizationSpec], list[AnalysisHypothesis]]:
    """Analyze Cramér's V and Chi-square independence among categorical variables."""
    findings: list[StatisticalFinding] = []
    visualizations: list[VisualizationSpec] = []
    hypotheses: list[AnalysisHypothesis] = []

    if len(cat_cols) < 2:
        return findings, visualizations, hypotheses

    # Select up to 15 categorical columns with cardinality between 2 and 50
    valid_cats = [c for c in cat_cols if 2 <= df[c].nunique(dropna=True) <= 50][:15]
    if len(valid_cats) < 2:
        return findings, visualizations, hypotheses

    n_cats = len(valid_cats)
    v_matrix = np.eye(n_cats)

    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            c1 = valid_cats[i]
            c2 = valid_cats[j]

            table = pd.crosstab(df[c1].dropna(), df[c2].dropna())
            if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
                continue

            try:
                v_val, chi2, p_val = compute_cramers_v(table)
                v_matrix[i, j] = v_val
                v_matrix[j, i] = v_val

                if v_val >= 0.40 and p_val < 0.05:
                    f = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="categorical_association",
                        summary=(
                            f"Fuerte asociación categórica entre '{c1}' y '{c2}' "
                            f"(Cramér's V={v_val:.3f}, χ²={chi2:.2f}, p={p_val:.4e})."
                        ),
                        column_name=c1,
                        secondary_column=c2,
                        method_name="cramers_v_bias_corrected",
                        metrics={"cramers_v": round(v_val, 4), "chi2_stat": round(chi2, 2)},
                        p_value=p_val,
                        effect_size=round(v_val, 3),
                    )
                    findings.append(f)
                    hyp = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f.id,
                        description=f"Evaluar fusión o codificación conjunta de interacciones entre '{c1}' y '{c2}'.",
                        proposed_action="categorical_interaction",
                        experiment_delta={"interaction_pair": [c1, c2]},
                    )
                    hypotheses.append(hyp)
            except Exception as exc:
                logger.debug("Cramer V calculation error between %s and %s: %s", c1, c2, exc)

    visualizations.append(
        VisualizationSpec.create(
            study_id=study_id,
            analysis_run_id=run_id,
            chart_type="categorical_association_matrix",
            title="Matriz de Asociación Categórica (Cramér's V corregido)",
            data_series={
                "columns": valid_cats,
                "matrix": [[round(float(v_matrix[r, c]), 3) for c in range(n_cats)] for r in range(n_cats)],
            },
            axes_config={"range": [0.0, 1.0]},
        )
    )

    return findings, visualizations, hypotheses
