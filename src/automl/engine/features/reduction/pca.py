from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from automl.domain.ports import DimensionalityReducerPort, FeatureAnalysisContext
from automl.engine.features.common import prepare_feature_matrix
from automl.engine.profiling.dataset_profiler import load_dataframe


class PCAReducer(DimensionalityReducerPort):
    method_id: str = "pca"

    def __init__(self) -> None:
        self._pca: PCA | None = None
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        context: FeatureAnalysisContext,
        n_components: int | float | None = None,
    ) -> None:
        df = load_dataframe(context.dataset_path)
        X, _ = prepare_feature_matrix(
            df,
            feature_names=context.active_feature_names,
            target_column=context.target_column,
            task_type=context.task_type,
        )
        self._feature_names = list(X.columns)

        # Build pipeline: imputer -> standard scaler -> PCA
        self._pca = PCA(n_components=n_components, random_state=context.random_seed)
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", self._pca),
        ])
        self._pipeline.fit(X)

    def transform(self, data: Any) -> np.ndarray:
        if self._pipeline is None or self._pca is None:
            raise ValueError("PCAReducer has not been fitted yet.")
        if isinstance(data, pd.DataFrame):
            # Select matching features if present
            matching = [c for c in self._feature_names if c in data.columns]
            if len(matching) == len(self._feature_names):
                data = data[self._feature_names]
        return self._pipeline.transform(data)

    def n_components(self) -> int:
        if self._pca is None:
            raise ValueError("PCAReducer has not been fitted yet.")
        return int(self._pca.n_components_)

    def explained_variance_ratio(self) -> list[float]:
        if self._pca is None:
            raise ValueError("PCAReducer has not been fitted yet.")
        return [float(v) for v in self._pca.explained_variance_ratio_]
