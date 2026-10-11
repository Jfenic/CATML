"""Declarative client-agnostic visualization specification builder for CATML Explore."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats

from automl.domain.analysis.models import VisualizationSpec

logger = logging.getLogger(__name__)


class VisualizationBuilder:
    """Builds aggregated, client-safe VisualizationSpec objects without overloading frontend DOM."""

    @staticmethod
    def build_histogram_spec(
        df: pd.DataFrame,
        column: str,
        study_id: str,
        run_id: str,
        bins: int = 15,
    ) -> VisualizationSpec | None:
        """Construct aggregated histogram specification for a continuous feature."""
        if column not in df.columns:
            return None
        series = df[column].dropna().values
        if len(series) < 5:
            return None

        try:
            counts, bin_edges = np.histogram(series, bins=bins)
            mean_val = float(np.mean(series))
            std_val = float(np.std(series))
            median_val = float(np.median(series))

            return VisualizationSpec.create(
                study_id=study_id,
                analysis_run_id=run_id,
                chart_type="histogram",
                title=f"Histograma de Frecuencia: {column}",
                data_series={
                    "column": column,
                    "counts": [int(c) for c in counts],
                    "bin_edges": [round(float(e), 3) for e in bin_edges],
                    "mean": round(mean_val, 3),
                    "std": round(std_val, 3),
                    "median": round(median_val, 3),
                    "total_points": int(len(series)),
                },
                axes_config={
                    "x_label": column,
                    "y_label": "Frecuencia",
                    "x_range": [round(float(bin_edges[0]), 3), round(float(bin_edges[-1]), 3)],
                },
            )
        except Exception as exc:
            logger.debug("Failed to build histogram for %s: %s", column, exc)
            return None

    @staticmethod
    def build_boxplot_spec(
        df: pd.DataFrame,
        column: str,
        study_id: str,
        run_id: str,
    ) -> VisualizationSpec | None:
        """Construct five-number summary boxplot specification with sampled outliers."""
        if column not in df.columns:
            return None
        series = df[column].dropna().values
        if len(series) < 5:
            return None

        try:
            q25 = float(np.percentile(series, 25))
            median = float(np.percentile(series, 50))
            q75 = float(np.percentile(series, 75))
            iqr = q75 - q25

            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            whisker_low = float(np.min(series[series >= lower_bound])) if (series >= lower_bound).any() else float(np.min(series))
            whisker_high = float(np.max(series[series <= upper_bound])) if (series <= upper_bound).any() else float(np.max(series))

            outliers = series[(series < lower_bound) | (series > upper_bound)]
            # Sample at most 25 outliers to keep JSON payload lean and crisp
            sample_outliers = [round(float(v), 3) for v in outliers[:25]]

            return VisualizationSpec.create(
                study_id=study_id,
                analysis_run_id=run_id,
                chart_type="boxplot",
                title=f"Diagrama de Caja y Outliers: {column}",
                data_series={
                    "column": column,
                    "q25": round(q25, 3),
                    "median": round(median, 3),
                    "q75": round(q75, 3),
                    "whisker_low": round(whisker_low, 3),
                    "whisker_high": round(whisker_high, 3),
                    "outliers_sample": sample_outliers,
                    "outliers_count": int(len(outliers)),
                    "total_points": int(len(series)),
                },
                axes_config={
                    "y_label": column,
                    "range": [round(float(np.min(series)), 3), round(float(np.max(series)), 3)],
                },
            )
        except Exception as exc:
            logger.debug("Failed to build boxplot for %s: %s", column, exc)
            return None

    @staticmethod
    def build_bivariate_scatter_spec(
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
        study_id: str,
        run_id: str,
        max_points: int = 200,
    ) -> VisualizationSpec | None:
        """Construct aggregated scatter plot specification with linear trend line."""
        if col_x not in df.columns or col_y not in df.columns:
            return None

        valid = (~df[col_x].isna()) & (~df[col_y].isna())
        if valid.sum() < 5:
            return None

        x_vals = df.loc[valid, col_x].values.astype(float)
        y_vals = df.loc[valid, col_y].values.astype(float)
        n = len(x_vals)

        try:
            # Deterministic downsampling for large sample sizes
            if n > max_points:
                step = n // max_points
                indices = np.arange(0, n, step)[:max_points]
                x_sample = x_vals[indices]
                y_sample = y_vals[indices]
            else:
                x_sample = x_vals
                y_sample = y_vals

            # Linear regression line
            slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)

            return VisualizationSpec.create(
                study_id=study_id,
                analysis_run_id=run_id,
                chart_type="scatter",
                title=f"Dispersión Bivariada: {col_x} vs {col_y}",
                data_series={
                    "col_x": col_x,
                    "col_y": col_y,
                    "x_points": [round(float(v), 3) for v in x_sample],
                    "y_points": [round(float(v), 3) for v in y_sample],
                    "trend_slope": round(float(slope), 4),
                    "trend_intercept": round(float(intercept), 4),
                    "pearson_r": round(float(r_val), 3),
                    "sample_size": n,
                },
                axes_config={
                    "x_label": col_x,
                    "y_label": col_y,
                    "x_range": [round(float(np.min(x_vals)), 3), round(float(np.max(x_vals)), 3)],
                    "y_range": [round(float(np.min(y_vals)), 3), round(float(np.max(y_vals)), 3)],
                },
            )
        except Exception as exc:
            logger.debug("Failed to build scatter for %s vs %s: %s", col_x, col_y, exc)
            return None
