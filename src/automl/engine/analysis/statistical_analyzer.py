"""Mathematical statistical analyzer implementation for CATML Explore."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats

from automl.domain.analysis.models import (
    AnalysisHypothesis,
    StatisticalFinding,
    VisualizationSpec,
)
from automl.domain.analysis.ports import StatisticalEnginePort
from automl.engine.profiling.dataset_profiler import load_dataframe

logger = logging.getLogger(__name__)


class StatisticalAnalyzer(StatisticalEnginePort):
    """Deterministic, rigorous numerical analyzer for exploratory studies."""

    def analyze(
        self,
        data_source_path: str,
        target_column: str | None = None,
        analysis_types: list[str] | None = None,
        parameters: dict | None = None,
    ) -> tuple[list[StatisticalFinding], list[VisualizationSpec], list[AnalysisHypothesis]]:
        """Compute statistical findings, visualizations, and actionable hypotheses."""
        df = load_dataframe(data_source_path)
        types = set(analysis_types or ["descriptive", "association"])
        params = parameters or {}

        study_id = params.get("study_id", "study_default")
        run_id = params.get("run_id", "run_default")

        findings: list[StatisticalFinding] = []
        visualizations: list[VisualizationSpec] = []
        hypotheses: list[AnalysisHypothesis] = []

        n_rows, n_cols = df.shape
        if n_rows == 0:
            return findings, visualizations, hypotheses

        # Identify column types
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target_column]

        # ---------------------------------------------------------
        # 1. Univariate Data Quality & Descriptive Findings
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

            # B) Skewness, Kurtosis & Outliers (Numeric columns)
            for col in num_cols:
                series = df[col].dropna()
                if len(series) < 5:
                    continue

                # Skewness
                try:
                    skew_val = float(stats.skew(series, nan_policy="omit"))
                    kurt_val = float(stats.kurtosis(series, nan_policy="omit"))
                    if abs(skew_val) >= 1.2:
                        direction = "positiva (cola derecha)" if skew_val > 0 else "negativa (cola izquierda)"
                        f = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="skewness",
                            summary=f"Asimetría severa {direction} detectada en '{col}' (skewness={skew_val:.2f}, curtosis={kurt_val:.2f}).",
                            column_name=col,
                            method_name="scipy.stats.skew",
                            metrics={"skewness": round(skew_val, 3), "kurtosis": round(kurt_val, 3)},
                            effect_size=abs(round(skew_val, 3)),
                        )
                        findings.append(f)
                        hyp = AnalysisHypothesis.create(
                            study_id=study_id,
                            finding_id=f.id,
                            description=f"Aplicar transformación matemática (Log1p / Yeo-Johnson) en '{col}' para normalizar distribución.",
                            proposed_action="power_transform",
                            experiment_delta={"transform": "yeo_johnson" if (series <= 0).any() else "log1p", "column": col},
                        )
                        hypotheses.append(hyp)
                except Exception as e:
                    logger.debug("Could not compute skewness for %s: %s", col, e)

                # Outliers using IQR
                try:
                    q25 = float(np.percentile(series, 25))
                    q75 = float(np.percentile(series, 75))
                    iqr = q75 - q25
                    if iqr > 1e-9:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                        outlier_rate = float(outliers / len(series))
                        if outlier_rate >= 0.03:
                            f = StatisticalFinding.create(
                                study_id=study_id,
                                analysis_run_id=run_id,
                                finding_type="outlier_density",
                                summary=f"Alta densidad de valores atípicos en '{col}' ({outliers} observaciones, {outlier_rate * 100:.1f}%).",
                                column_name=col,
                                method_name="iqr_bounds",
                                metrics={"outlier_count": int(outliers), "outlier_rate": round(outlier_rate, 4), "iqr": round(iqr, 4)},
                                effect_size=round(outlier_rate, 3),
                            )
                            findings.append(f)
                            hyp = AnalysisHypothesis.create(
                                study_id=study_id,
                                finding_id=f.id,
                                description=f"Usar RobustScaler o winsorización de cuantiles en '{col}' para mitigar influencia de atípicos.",
                                proposed_action="robust_scaler",
                                experiment_delta={"scaler": "robust", "column": col},
                            )
                            hypotheses.append(hyp)
                except Exception as e:
                    logger.debug("Could not compute IQR for %s: %s", col, e)

            # C) Pseudo-identifiers in categorical features
            for col in cat_cols:
                n_uniq = int(df[col].nunique(dropna=True))
                if n_uniq >= 10 and (n_uniq / n_rows) >= 0.85:
                    f = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="high_cardinality_identifier",
                        summary=f"Columna '{col}' posee cardinalidad extremadamente alta ({n_uniq} valores únicos para {n_rows} filas), consistente con un ID o clave única.",
                        column_name=col,
                        method_name="cardinality_ratio",
                        metrics={"unique_count": n_uniq, "cardinality_ratio": round(n_uniq / n_rows, 4)},
                    )
                    findings.append(f)
                    hyp = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f.id,
                        description=f"Excluir '{col}' del espacio de entrenamiento para prevenir memorización.",
                        proposed_action="exclude_identifier",
                        experiment_delta={"exclude_columns": [col]},
                    )
                    hypotheses.append(hyp)

        # ---------------------------------------------------------
        # 2. Associations & Collinearity Findings
        # ---------------------------------------------------------
        if "association" in types and len(num_cols) >= 2:
            # Pairwise collinearity check
            corr_matrix = df[num_cols].corr(method="pearson")
            visited_pairs = set()

            for i, col1 in enumerate(num_cols):
                for j in range(i + 1, len(num_cols)):
                    col2 = num_cols[j]
                    val = corr_matrix.loc[col1, col2]
                    if pd.isna(val):
                        continue
                    r_val = float(val)
                    if abs(r_val) >= 0.70:
                        pair_key = tuple(sorted([col1, col2]))
                        if pair_key in visited_pairs:
                            continue
                        visited_pairs.add(pair_key)

                        p_val = None
                        try:
                            s1 = df[col1]
                            s2 = df[col2]
                            valid = (~s1.isna()) & (~s2.isna())
                            if valid.sum() >= 5:
                                _, p = stats.pearsonr(s1[valid], s2[valid])
                                p_val = float(p)
                        except Exception:
                            pass

                        f = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="high_collinearity",
                            summary=f"Alta colinealidad bivariada entre '{col1}' y '{col2}' (r={r_val:.3f}).",
                            column_name=col1,
                            secondary_column=col2,
                            method_name="pearson_correlation",
                            metrics={"pearson_r": round(r_val, 4)},
                            p_value=p_val,
                            effect_size=abs(round(r_val, 3)),
                        )
                        findings.append(f)
                        hyp = AnalysisHypothesis.create(
                            study_id=study_id,
                            finding_id=f.id,
                            description=f"Aplicar penalización L1 o evaluar eliminación de una variable entre '{col1}' y '{col2}'.",
                            proposed_action="resolve_collinearity",
                            experiment_delta={"regularization": "l1", "collinear_pair": [col1, col2]},
                        )
                        hypotheses.append(hyp)

        # ---------------------------------------------------------
        # 3. Target Association Findings (Supervised context)
        # ---------------------------------------------------------
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            is_target_num = pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10

            if is_target_num:
                # Numerical target -> Pearson correlation with features
                for col in num_cols:
                    try:
                        valid = (~df[col].isna()) & (~target_series.isna())
                        if valid.sum() >= 5:
                            r, p = stats.pearsonr(df.loc[valid, col], target_series.loc[valid])
                            r_val = float(r)
                            p_val = float(p)
                            if abs(r_val) >= 0.25 and p_val < 0.05:
                                f = StatisticalFinding.create(
                                    study_id=study_id,
                                    analysis_run_id=run_id,
                                    finding_type="target_correlation",
                                    summary=f"Asociación lineal predictiva entre '{col}' y la variable objetivo '{target_column}' (r={r_val:.3f}, p={p_val:.4e}).",
                                    column_name=col,
                                    secondary_column=target_column,
                                    method_name="pearson_target_correlation",
                                    metrics={"pearson_r": round(r_val, 4)},
                                    p_value=p_val,
                                    effect_size=abs(round(r_val, 3)),
                                )
                                findings.append(f)
                    except Exception as e:
                        logger.debug("Target correlation failed for %s: %s", col, e)
            else:
                # Categorical / classification target -> ANOVA F-test
                clean_target = target_series.dropna()
                classes = clean_target.unique()
                if len(classes) >= 2 and len(classes) <= 20:
                    for col in num_cols:
                        try:
                            groups = [df.loc[clean_target == c, col].dropna() for c in classes]
                            groups = [g for g in groups if len(g) >= 2]
                            if len(groups) >= 2:
                                f_stat, p_val = stats.f_oneway(*groups)
                                if not np.isnan(f_stat) and p_val < 0.01:
                                    f = StatisticalFinding.create(
                                        study_id=study_id,
                                        analysis_run_id=run_id,
                                        finding_type="target_class_separation",
                                        summary=f"Separación univariante altamente significativa de clases en '{col}' frente a '{target_column}' (F={f_stat:.2f}, p={p_val:.4e}).",
                                        column_name=col,
                                        secondary_column=target_column,
                                        method_name="anova_f_oneway",
                                        metrics={"f_statistic": round(float(f_stat), 3)},
                                        p_value=float(p_val),
                                        effect_size=round(float(f_stat), 2),
                                    )
                                    findings.append(f)
                        except Exception as e:
                            logger.debug("ANOVA target association failed for %s: %s", col, e)

        # ---------------------------------------------------------
        # 4. Declarative Visualizations (VisualizationSpec)
        # ---------------------------------------------------------
        # A) Histograms for top 4 numeric features
        for col in num_cols[:4]:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            try:
                counts, bin_edges = np.histogram(series, bins=10)
                bins_payload = []
                for b_idx in range(len(counts)):
                    bins_payload.append({
                        "bin_start": round(float(bin_edges[b_idx]), 3),
                        "bin_end": round(float(bin_edges[b_idx + 1]), 3),
                        "count": int(counts[b_idx]),
                    })
                v = VisualizationSpec.create(
                    study_id=study_id,
                    analysis_run_id=run_id,
                    chart_type="histogram",
                    title=f"Distribución empírica: {col}",
                    data_series={"bins": bins_payload, "total": len(series)},
                    axes_config={"x_label": col, "y_label": "Frecuencia"},
                )
                visualizations.append(v)
            except Exception as e:
                logger.debug("Failed building histogram for %s: %s", col, e)

        # B) Correlation Matrix Spec if multiple numeric features
        if len(num_cols) >= 2:
            matrix_cols = num_cols[:8]  # Limit to 8 to avoid overwhelming visualizations
            corr_sub = df[matrix_cols].corr().fillna(0.0)
            corr_list = []
            for r_col in matrix_cols:
                row_vals = [round(float(corr_sub.loc[r_col, c_col]), 3) for c_col in matrix_cols]
                corr_list.append(row_vals)
            v = VisualizationSpec.create(
                study_id=study_id,
                analysis_run_id=run_id,
                chart_type="correlation_matrix",
                title="Matriz de Correlación Bivariada (Pearson)",
                data_series={"columns": matrix_cols, "matrix": corr_list},
                axes_config={"x_label": "Características", "y_label": "Características"},
            )
            visualizations.append(v)

        # C) Target Distribution Spec (if available)
        if target_column and target_column in df.columns:
            tgt_series = df[target_column].dropna()
            if not pd.api.types.is_numeric_dtype(tgt_series) or tgt_series.nunique() <= 10:
                vc = tgt_series.value_counts()
                bars = [{"category": str(k), "count": int(v)} for k, v in vc.items()]
                v = VisualizationSpec.create(
                    study_id=study_id,
                    analysis_run_id=run_id,
                    chart_type="bar",
                    title=f"Balance de Clases: {target_column}",
                    data_series={"categories": bars},
                    axes_config={"x_label": target_column, "y_label": "Instancias"},
                )
                visualizations.append(v)

        return findings, visualizations, hypotheses
