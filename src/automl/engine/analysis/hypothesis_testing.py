"""Inferential hypothesis testing and multiple testing corrections for CATML Explore."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression

from automl.domain.analysis.models import AnalysisHypothesis, StatisticalFinding

logger = logging.getLogger(__name__)


def compute_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent groups."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled < 1e-9:
        return 0.0
    return float(abs(np.mean(g1) - np.mean(g2)) / s_pooled)


def compute_eta_squared(groups: list[np.ndarray]) -> float:
    """Compute Eta-squared (η²) effect size for one-way ANOVA."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total < 1e-9:
        return 0.0
    ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups)
    return float(min(1.0, max(0.0, ss_between / ss_total)))


def apply_benjamini_hochberg_correction(findings: list[StatisticalFinding]) -> None:
    """Apply Benjamini-Hochberg (FDR) multiple comparisons correction in-place.
    
    Adjusts raw p-values across all statistical tests conducted during the study run
    to protect against data dredging / type I error inflation.
    """
    tests_with_p = [f for f in findings if f.p_value is not None]
    m = len(tests_with_p)
    if m == 0:
        return

    # Sort tests by ascending p-value
    indexed = sorted(enumerate(tests_with_p), key=lambda x: x[1].p_value or 1.0)
    
    # Compute Benjamini-Hochberg step-up adjusted p-values
    # p_adj(i) = min_{j >= i} (p(j) * m / j)
    raw_p_values = [item[1].p_value or 1.0 for item in indexed]
    adjusted = [0.0] * m
    cum_min = 1.0

    for rank_idx in range(m - 1, -1, -1):
        rank = rank_idx + 1  # 1-based rank
        curr_p = raw_p_values[rank_idx]
        adj = (curr_p * m) / rank
        cum_min = min(cum_min, adj)
        adjusted[rank_idx] = min(1.0, max(0.0, cum_min))

    # Store FDR values back into findings
    for (orig_idx, finding), adj_p in zip(indexed, adjusted):
        finding.metrics["p_value_fdr"] = round(float(adj_p), 6)
        finding.metrics["fdr_significant"] = bool(adj_p < 0.05)
        if (finding.p_value or 1.0) < 0.05 and adj_p >= 0.05:
            finding.limitations.append(
                f"El hallazgo no supera la corrección FDR Benjamini-Hochberg (p_fdr={adj_p:.4f} ≥ 0.05). Posible falso positivo."
            )


def analyze_target_hypotheses(
    df: pd.DataFrame,
    target_column: str,
    num_cols: list[str],
    cat_cols: list[str],
    study_id: str,
    run_id: str,
) -> tuple[list[StatisticalFinding], list[AnalysisHypothesis]]:
    """Conduct automated hypothesis tests against target (Welch t-test, ANOVA, Mann-Whitney, Levene)."""
    findings: list[StatisticalFinding] = []
    hypotheses: list[AnalysisHypothesis] = []

    if target_column not in df.columns:
        return findings, hypotheses

    target_series = df[target_column].dropna()
    if len(target_series) < 10:
        return findings, hypotheses

    n_unique_target = target_series.nunique()
    is_target_num = pd.api.types.is_numeric_dtype(target_series) and n_unique_target > 10

    if is_target_num:
        # Continuous regression target: evaluate linear correlation and mutual information
        y = target_series
        for col in num_cols:
            valid = (~df[col].isna()) & (~df[target_column].isna())
            if valid.sum() < 10:
                continue

            x = df.loc[valid, col]
            y_valid = df.loc[valid, target_column]

            try:
                r_val, p_val = stats.pearsonr(x, y_valid)
                rho_val, p_spearman = stats.spearmanr(x, y_valid)
                r_float, p_float = float(r_val), float(p_val)
                rho_float = float(rho_val)

                if abs(r_float) >= 0.20 and p_float < 0.05:
                    f = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="target_correlation",
                        summary=(
                            f"Asociación lineal predictiva entre '{col}' y la variable objetivo '{target_column}' "
                            f"(Pearson r={r_float:.3f}, Spearman ρ={rho_float:.3f}, p={p_float:.4e})."
                        ),
                        column_name=col,
                        secondary_column=target_column,
                        method_name="pearson_correlation",
                        metrics={
                            "pearson_r": round(r_float, 4),
                            "spearman_rho": round(rho_float, 4),
                        },
                        p_value=p_float,
                        effect_size=abs(round(r_float, 3)),
                    )
                    findings.append(f)
                    hyp = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f.id,
                        description=f"Priorizar la variable '{col}' como predictor numérico primario de '{target_column}'.",
                        proposed_action="prioritize_feature",
                        experiment_delta={"feature": col, "priority": "high"},
                    )
                    hypotheses.append(hyp)

            except Exception as exc:
                logger.debug("Target correlation failed for %s: %s", col, exc)

    else:
        # Categorical / Discrete classification target (binary or multiclass)
        for col in num_cols:
            groups_dict: dict[Any, np.ndarray] = {}
            for cls_val, grp in df.groupby(target_column)[col]:
                arr = grp.dropna().values.astype(float)
                if len(arr) >= 3:
                    groups_dict[cls_val] = arr

            groups = list(groups_dict.values())
            if len(groups) < 2:
                continue

            try:
                # 1. Homoscedasticity test (Levene)
                levene_p: float | None = None
                try:
                    _, lp = stats.levene(*groups)
                    levene_p = float(lp)
                except Exception:
                    pass

                # 2. Binary classification (2 groups)
                if len(groups) == 2:
                    g1, g2 = groups[0], groups[1]
                    welch_t, welch_p = stats.ttest_ind(g1, g2, equal_var=False)
                    u_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    d_val = compute_cohens_d(g1, g2)
                    p_chosen = float(welch_p)

                    if p_chosen < 0.05 or mw_p < 0.05:
                        f = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="target_class_separation",
                            summary=(
                                f"Separación univariante significativa en '{col}' entre clases de '{target_column}' "
                                f"(Welch t={float(welch_t):.2f}, p={p_chosen:.4e}, Mann-Whitney p={float(mw_p):.4e}, Cohen's d={d_val:.2f})."
                            ),
                            column_name=col,
                            secondary_column=target_column,
                            method_name="welch_ttest_and_mannwhitney",
                            metrics={
                                "welch_t": round(float(welch_t), 3),
                                "mann_whitney_u": round(float(u_stat), 1),
                                "cohens_d": round(d_val, 3),
                                "levene_p_variance_homogeneity": round(levene_p, 4) if levene_p is not None else None,
                            },
                            p_value=p_chosen,
                            effect_size=round(d_val, 3),
                        )
                        findings.append(f)
                        hyp = AnalysisHypothesis.create(
                            study_id=study_id,
                            finding_id=f.id,
                            description=f"Priorizar la variable discriminativa '{col}' con alta separación de clases.",
                            proposed_action="prioritize_feature",
                            experiment_delta={"feature": col, "priority": "high"},
                        )
                        hypotheses.append(hyp)

                # 3. Multiclass classification (> 2 groups)
                elif len(groups) > 2:
                    f_stat, anova_p = stats.f_oneway(*groups)
                    kw_stat, kw_p = stats.kruskal(*groups)
                    eta2 = compute_eta_squared(groups)
                    p_chosen = float(anova_p)

                    if p_chosen < 0.05 or kw_p < 0.05:
                        f = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="target_class_separation",
                            summary=(
                                f"Separación multiclase significativa en '{col}' para '{target_column}' "
                                f"(ANOVA F={float(f_stat):.2f}, p={p_chosen:.4e}, Kruskal-Wallis p={float(kw_p):.4e}, η²={eta2:.3f})."
                            ),
                            column_name=col,
                            secondary_column=target_column,
                            method_name="oneway_anova_and_kruskal",
                            metrics={
                                "anova_f": round(float(f_stat), 3),
                                "kruskal_h": round(float(kw_stat), 3),
                                "eta_squared": round(eta2, 4),
                                "groups_count": len(groups),
                            },
                            p_value=p_chosen,
                            effect_size=round(eta2, 3),
                        )
                        findings.append(f)
                        hyp = AnalysisHypothesis.create(
                            study_id=study_id,
                            finding_id=f.id,
                            description=f"Priorizar variable discriminativa multiclase '{col}' en modelos supervisados.",
                            proposed_action="prioritize_feature",
                            experiment_delta={"feature": col, "priority": "high"},
                        )
                        hypotheses.append(hyp)

            except Exception as exc:
                logger.debug("Hypothesis test failed for column %s: %s", col, exc)

    return findings, hypotheses
