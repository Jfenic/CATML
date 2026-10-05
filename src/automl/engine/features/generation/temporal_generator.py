"""Automated temporal dynamics and sequential feature generator for CATML."""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.selection_strategy import FeatureSetCandidate
from automl.domain.features.temporal import (
    CyclicalSpec,
    DeltaSpec,
    GeneratedTemporalFeature,
    LagSpec,
    RollingWindowSpec,
    TemporalPeriodicity,
    TemporalStructure,
)
from automl.engine.profiling.dataset_profiler import detect_sequential_structure


class TemporalDynamicsGenerator:
    """
    Automated temporal and sequential dynamics generator for CATML.
    Discovers candidate features by generating:
      1. Sensor/Feature autoregressive lags: X_{t-k}
      2. Trend deltas / rate-of-change: Delta X = X_t - X_{t-k}
      3. Trigonometric cyclical projections: sin(2pi * t / T) and cos(2pi * t / T)
      4. Rolling window moving statistics: moving mean

    Strictly obeys 'Propose != Accept': Original datasets are never mutated,
    and generated features are proposed as candidate FeatureSets for hypothesis testing.
    """

    def __init__(
        self,
        max_lags: int = 1,
        include_lags: bool = True,
        include_deltas: bool = True,
        include_cyclical: bool = True,
        include_rolling: bool = False,
        rolling_windows: list[int] | None = None,
        max_numerical_features: int = 20,
    ) -> None:
        self.max_lags = max(1, int(max_lags))
        self.include_lags = include_lags
        self.include_deltas = include_deltas
        self.include_cyclical = include_cyclical
        self.include_rolling = include_rolling
        self.rolling_windows = list(rolling_windows or [3, 5])
        self.max_numerical_features = max_numerical_features

        # Learned context for test transformations
        self.fitted: bool = False
        self._fitted_tails: dict[str, pd.Series] = {}
        self._fitted_medians: dict[str, float] = {}
        self._order_column: str | None = None

    def propose_features(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        target_column: str | None = None,
        temporal_structure: TemporalStructure | None = None,
    ) -> list[GeneratedTemporalFeature]:
        """
        Scans dataframe and feature names to propose candidate temporal features
        (lags, deltas, cyclical sine/cosine projections, rolling windows)
        without mutating the input dataframe.
        """
        if temporal_structure is None:
            temporal_structure = detect_sequential_structure(df, target_column=target_column)

        candidates: list[GeneratedTemporalFeature] = []
        valid_cols = [c for c in feature_names if c in df.columns and c != target_column]

        # Order column should not be lagged if it is a pure counter/timestamp
        order_col = temporal_structure.order_column
        numeric_cols = [
            c for c in valid_cols
            if pd.api.types.is_numeric_dtype(df[c]) and c != order_col
        ][: self.max_numerical_features]

        # 1. Autoregressive lags: X_{t-k}
        if self.include_lags:
            for col in numeric_cols:
                for k in range(1, self.max_lags + 1):
                    candidates.append(
                        GeneratedTemporalFeature(
                            name=f"temp_lag_{k}_{col}",
                            feature_type="lag",
                            source_column=col,
                            lag=k,
                            description=f"Autoregressive lag {k} step(s) of '{col}'",
                        )
                    )

        # 2. Trend deltas / Rate-of-change: Delta X = X_t - X_{t-k}
        if self.include_deltas:
            for col in numeric_cols:
                for k in range(1, self.max_lags + 1):
                    candidates.append(
                        GeneratedTemporalFeature(
                            name=f"temp_delta_{k}_{col}",
                            feature_type="delta",
                            source_column=col,
                            lag=k,
                            description=f"Trend delta (X_t - X_{{t-{k}}}) of '{col}'",
                        )
                    )

        # 3. Trigonometric cyclical projections: sin(2*pi*t/T) and cos(2*pi*t/T)
        if self.include_cyclical and temporal_structure.detected_periodicities:
            for p in temporal_structure.detected_periodicities:
                col = p.column
                if col in df.columns and col != target_column:
                    period = p.period
                    period_tag = int(period) if period.is_integer() else str(period).replace(".", "_")
                    sin_name = f"temp_sin_{period_tag}_{col}"
                    cos_name = f"temp_cos_{period_tag}_{col}"

                    candidates.append(
                        GeneratedTemporalFeature(
                            name=sin_name,
                            feature_type="cyclical_sin",
                            source_column=col,
                            period=period,
                            description=f"Trigonometric sine cyclical projection (T={period}) of '{col}'",
                        )
                    )
                    candidates.append(
                        GeneratedTemporalFeature(
                            name=cos_name,
                            feature_type="cyclical_cos",
                            source_column=col,
                            period=period,
                            description=f"Trigonometric cosine cyclical projection (T={period}) of '{col}'",
                        )
                    )

        # 4. Rolling window moving statistics
        if self.include_rolling:
            for col in numeric_cols:
                for window in self.rolling_windows:
                    candidates.append(
                        GeneratedTemporalFeature(
                            name=f"temp_roll_mean_{window}_{col}",
                            feature_type="rolling_mean",
                            source_column=col,
                            window=window,
                            agg="mean",
                            description=f"Rolling mean over window {window} of '{col}'",
                        )
                    )

        return candidates

    def fit(
        self,
        df: pd.DataFrame,
        features: list[GeneratedTemporalFeature],
        order_column: str | None = None,
    ) -> TemporalDynamicsGenerator:
        """
        Fits baseline tail observations and imputation medians on the training set
        to allow seamless, non-leaking transformations on subsequent test sets.
        """
        self._order_column = order_column
        self._fitted_tails.clear()
        self._fitted_medians.clear()

        tail_size = self.max_lags + max(self.rolling_windows, default=5) + 5
        for feat in features:
            col = feat.source_column
            if col in df.columns and col not in self._fitted_tails:
                self._fitted_tails[col] = df[col].tail(tail_size).copy()
                clean_vals = df[col].dropna()
                median_val = float(clean_vals.median()) if len(clean_vals) > 0 else 0.0
                self._fitted_medians[col] = median_val

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        features: list[GeneratedTemporalFeature],
        is_continuation: bool = False,
    ) -> pd.DataFrame:
        """
        Applies generated temporal feature transformations without mutating the original dataframe.
        Returns a new DataFrame containing original and transformed temporal columns.
        """
        out_df = df.copy()

        for feat in features:
            col = feat.source_column
            if col not in out_df.columns:
                continue

            series = out_df[col]
            median_fill = self._fitted_medians.get(col, 0.0)

            if feat.feature_type == "lag":
                k = feat.lag or 1
                if is_continuation and col in self._fitted_tails:
                    tail_s = self._fitted_tails[col]
                    combined = pd.concat([tail_s, series], ignore_index=True)
                    shifted = combined.shift(k).iloc[len(tail_s):].reset_index(drop=True)
                    out_df[feat.name] = shifted.bfill().fillna(median_fill).to_numpy(dtype=float)
                else:
                    out_df[feat.name] = series.shift(k).bfill().fillna(median_fill).astype(float)

            elif feat.feature_type == "delta":
                k = feat.lag or 1
                if is_continuation and col in self._fitted_tails:
                    tail_s = self._fitted_tails[col]
                    combined = pd.concat([tail_s, series], ignore_index=True)
                    shifted = combined.shift(k).iloc[len(tail_s):].reset_index(drop=True)
                    delta_vals = series.to_numpy(dtype=float) - shifted.bfill().fillna(median_fill).to_numpy(dtype=float)
                    out_df[feat.name] = delta_vals.astype(float)
                else:
                    shifted = series.shift(k).bfill().fillna(series.iloc[0] if len(series) > 0 else median_fill)
                    out_df[feat.name] = (series.astype(float) - shifted.astype(float)).fillna(0.0).astype(float)

            elif feat.feature_type == "cyclical_sin":
                period = feat.period or 24.0
                rads = 2.0 * np.pi * series.astype(float) / period
                sin_vals = np.sin(rads)
                out_df[feat.name] = np.nan_to_num(sin_vals, nan=0.0).astype(float)

            elif feat.feature_type == "cyclical_cos":
                period = feat.period or 24.0
                rads = 2.0 * np.pi * series.astype(float) / period
                cos_vals = np.cos(rads)
                out_df[feat.name] = np.nan_to_num(cos_vals, nan=0.0).astype(float)

            elif feat.feature_type == "rolling_mean":
                window = feat.window or 3
                out_df[feat.name] = (
                    series.rolling(window=window, min_periods=1)
                    .mean()
                    .bfill()
                    .fillna(median_fill)
                    .astype(float)
                )

        return out_df

    def propose_candidate_feature_sets(
        self,
        base_features: list[str],
        generated_features: list[GeneratedTemporalFeature],
        dataset_id: str,
    ) -> list[FeatureSet]:
        """
        Packages generated feature candidates into explicit FeatureSet versions
        ready for hypothesis-driven experimentation ('Propose != Accept').
        """
        candidates: list[FeatureSet] = []

        all_gen_names = [f.name for f in generated_features]
        lag_names = [f.name for f in generated_features if f.feature_type == "lag"]
        delta_names = [f.name for f in generated_features if f.feature_type == "delta"]
        cyclical_names = [
            f.name for f in generated_features if f.feature_type in {"cyclical_sin", "cyclical_cos"}
        ]

        # 1. Base + All Temporal Features
        if all_gen_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_temporal_all_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_temporal_all",
                    feature_names=base_features + all_gen_names,
                    created_by="temporal_dynamics_generator",
                    lineage=f"Base ({len(base_features)}) + {len(all_gen_names)} temporal features (lags, deltas, cyclical)",
                )
            )

        # 2. Base + Autoregressive Lags Only
        if lag_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_temporal_lags_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_temporal_lags",
                    feature_names=base_features + lag_names,
                    created_by="temporal_dynamics_generator",
                    lineage=f"Base ({len(base_features)}) + {len(lag_names)} autoregressive lag features",
                )
            )

        # 3. Base + Trend Deltas Only
        if delta_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_temporal_deltas_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_temporal_deltas",
                    feature_names=base_features + delta_names,
                    created_by="temporal_dynamics_generator",
                    lineage=f"Base ({len(base_features)}) + {len(delta_names)} rate-of-change trend delta features",
                )
            )

        # 4. Base + Cyclical Trigonometric Projections Only
        if cyclical_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_temporal_cyclical_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_temporal_cyclical",
                    feature_names=base_features + cyclical_names,
                    created_by="temporal_dynamics_generator",
                    lineage=f"Base ({len(base_features)}) + {len(cyclical_names)} cyclical sine/cosine projections",
                )
            )

        return candidates

    def propose_candidate_sets_as_candidates(
        self,
        base_features: list[str],
        generated_features: list[GeneratedTemporalFeature],
    ) -> list[FeatureSetCandidate]:
        """Converts proposed feature sets into FeatureSetCandidate instances for workspace registry."""
        feature_sets = self.propose_candidate_feature_sets(base_features, generated_features, "dataset")
        return [
            FeatureSetCandidate.create(
                name=fs.name,
                feature_names=fs.feature_names,
                method="temporal_dynamics",
                k=len(fs.feature_names),
                metadata={"lineage": fs.lineage},
            )
            for fs in feature_sets
        ]
