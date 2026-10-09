"""Mathematical statistical analyzer implementation for CATML Explore (Advanced Engine - Phase E2)."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from automl.domain.analysis.models import (
    AnalysisHypothesis,
    StatisticalFinding,
    VisualizationSpec,
)
from automl.domain.analysis.ports import StatisticalEnginePort
from automl.engine.analysis.association_metrics import (
    analyze_categorical_associations,
    analyze_numeric_associations,
)
from automl.engine.analysis.distribution_diagnostics import (
    diagnose_multivariate_outliers,
    diagnose_univariate_distributions,
)
from automl.engine.analysis.hypothesis_testing import (
    analyze_target_hypotheses,
    apply_benjamini_hochberg_correction,
)
from automl.engine.analysis.visualizations.builder import VisualizationBuilder
from automl.engine.profiling.dataset_profiler import load_dataframe

logger = logging.getLogger(__name__)


class StatisticalAnalyzer(StatisticalEnginePort):
    """Deterministic, rigorous numerical analyzer for exploratory studies (Phase E2 Advanced Engine)."""

    def analyze(
        self,
        data_source_path: str,
        target_column: str | None = None,
        analysis_types: list[str] | None = None,
        parameters: dict | None = None,
    ) -> tuple[list[StatisticalFinding], list[VisualizationSpec], list[AnalysisHypothesis]]:
        findings: list[StatisticalFinding] = []
        visualizations: list[VisualizationSpec] = []
        hypotheses: list[AnalysisHypothesis] = []

        try:
            df = load_dataframe(data_source_path)
        except Exception as exc:
            logger.warning("Could not load dataframe from %s: %s", data_source_path, exc)
            return findings, visualizations, hypotheses

        types = set(analysis_types or ["descriptive", "association"])
        params = parameters or {}

        study_id = params.get("study_id", "study_default")
        run_id = params.get("run_id", "run_default")

        n_rows, n_cols = df.shape
        if n_rows == 0:
            return findings, visualizations, hypotheses

        # Identify column types
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target_column]

        # ---------------------------------------------------------
        # 1. Descriptive & Distribution Quality Diagnostics
        # ---------------------------------------------------------
        if "descriptive" in types:
            # A) Missingness
            for col in df.columns:
                if col == target_column:
                    continue
                missing_cnt = int(df[col].isna().sum())
                if missing_cnt > 0:
                    rate = float(missing_cnt / n_rows)
                    if rate >= 0.01:
                        f = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="missingness",
                            summary=f"Columna '{col}' tiene {missing_cnt} valores faltantes ({rate * 100:.1f}%).",
                            column_name=col,
                            method_name="count_nulls",
                            metrics={"missing_count": missing_cnt, "missing_rate": round(rate, 4)},
                        )
                        findings.append(f)
                        if rate >= 0.05:
                            hyp = AnalysisHypothesis.create(
                                study_id=study_id,
                                finding_id=f.id,
                                description=f"Imputar '{col}' mediante indicador explícito de nulo o imputación robusta.",
                                proposed_action="impute_with_indicator",
                                experiment_delta={"imputer": "simple", "add_indicator": True, "column": col},
                            )
                            hypotheses.append(hyp)

            # B) Univariate distributions (Normality, Skewness, Multimodality, IQR Outliers)
            uni_findings, uni_hyps = diagnose_univariate_distributions(
                df=df,
                num_cols=num_cols,
                study_id=study_id,
                run_id=run_id,
            )
            findings.extend(uni_findings)
            hypotheses.extend(uni_hyps)

            # Generate histogram and boxplot specs for distribution findings
            for f in uni_findings:
                if f.finding_type in ("skewness", "multimodal_distribution") and f.column_name:
                    h_spec = VisualizationBuilder.build_histogram_spec(df, f.column_name, study_id, run_id)
                    if h_spec:
                        visualizations.append(h_spec)
                elif f.finding_type == "outlier_density" and f.column_name:
                    b_spec = VisualizationBuilder.build_boxplot_spec(df, f.column_name, study_id, run_id)
                    if b_spec:
                        visualizations.append(b_spec)

            # C) Multivariate anomalies (Mahalanobis Distance with Chi-square test)
            multi_findings, multi_hyps = diagnose_multivariate_outliers(
                df=df,
                num_cols=num_cols,
                study_id=study_id,
                run_id=run_id,
            )
            findings.extend(multi_findings)
            hypotheses.extend(multi_hyps)

            # D) High Cardinality Identifier Candidates
            for col in df.columns:
                if col == target_column:
                    continue
                nunique = int(df[col].nunique(dropna=True))
                card_ratio = float(nunique / n_rows)
                # Primary key candidate or high cardinality ID
                if nunique >= 50 and card_ratio >= 0.95 and ("id" in col.lower() or "uuid" in col.lower() or card_ratio >= 0.99):
                    f = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="high_cardinality_identifier",
                        summary=f"Columna '{col}' presenta cardinalidad casi única ({nunique}/{n_rows}, {card_ratio * 100:.1f}%), probable clave primaria o ID.",
                        column_name=col,
                        method_name="cardinality_ratio",
                        metrics={"unique_count": nunique, "cardinality_ratio": round(card_ratio, 4)},
                    )
                    findings.append(f)
                    hyp = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f.id,
                        description=f"Excluir identificador de alta cardinalidad '{col}' de las variables predictivas para evitar sobreajuste.",
                        proposed_action="exclude_identifier",
                        experiment_delta={"exclude_columns": [col]},
                    )
                    hypotheses.append(hyp)

        # ---------------------------------------------------------
        # 2. Association Diagnostics (Linear, Monotonic, Categorical)
        # ---------------------------------------------------------
        if "association" in types:
            # Numeric associations (Pearson with Fisher 95% CI, Spearman monotonic non-linear, Viz)
            num_findings, num_viz, num_hyps = analyze_numeric_associations(
                df=df,
                num_cols=num_cols,
                study_id=study_id,
                run_id=run_id,
            )
            findings.extend(num_findings)
            visualizations.extend(num_viz)
            hypotheses.extend(num_hyps)

            # Generate bivariate scatter specs for top collinear/monotonic relationships
            for f in num_findings[:3]:
                if f.finding_type in ("high_collinearity", "monotonic_nonlinear_relationship") and f.column_name and f.secondary_column:
                    sc_spec = VisualizationBuilder.build_bivariate_scatter_spec(
                        df, f.column_name, f.secondary_column, study_id, run_id
                    )
                    if sc_spec:
                        visualizations.append(sc_spec)

            # Categorical associations (Cramér's V bias-corrected, Chi-square, Viz)
            cat_findings, cat_viz, cat_hyps = analyze_categorical_associations(
                df=df,
                cat_cols=cat_cols,
                study_id=study_id,
                run_id=run_id,
            )
            findings.extend(cat_findings)
            visualizations.extend(cat_viz)
            hypotheses.extend(cat_hyps)

        # ---------------------------------------------------------
        # 3. Target Inferences & Group Comparisons (Supervised context)
        # ---------------------------------------------------------
        if target_column and target_column in df.columns:
            target_findings, target_hyps = analyze_target_hypotheses(
                df=df,
                target_column=target_column,
                num_cols=num_cols,
                cat_cols=cat_cols,
                study_id=study_id,
                run_id=run_id,
            )
            findings.extend(target_findings)
            hypotheses.extend(target_hyps)

        # ---------------------------------------------------------
        # 4. Multiple Testing Correction (Benjamini-Hochberg FDR)
        # ---------------------------------------------------------
        apply_benjamini_hochberg_correction(findings)

        return findings, visualizations, hypotheses
