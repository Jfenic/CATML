from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineNode


class FeatureFusionNode:
    """Combines heterogeneous representations (tabular columns + dense embeddings) into a unified matrix."""

    def __init__(
        self,
        node: PipelineNode | None = None,
        node_id: str = "feature_fusion",
        strategy: str = "concat",
        default_prefix: str = "emb",
    ) -> None:
        if node is not None:
            self.node_id = node.node_id
            params = dict(node.parameters)
            self.strategy = str(params.get("strategy", strategy))
            self.default_prefix = str(params.get("default_prefix", default_prefix))
        else:
            self.node_id = node_id
            self.strategy = strategy
            self.default_prefix = default_prefix

    @classmethod
    def from_pipeline_node(cls, node: PipelineNode) -> FeatureFusionNode:
        return cls(node=node)

    def to_pipeline_node(self) -> PipelineNode:
        return PipelineNode(
            node_id=self.node_id,
            node_type=NodeType.FUSION,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
            name=f"FeatureFusion({self.strategy})",
            parameters={
                "strategy": self.strategy,
                "default_prefix": self.default_prefix,
            },
        )

    def execute(self, inputs: Any) -> pd.DataFrame:
        """Executes node logic within a Pipeline DAG runner."""
        return self.fuse(inputs)

    def fuse(
        self,
        inputs: dict[str, pd.DataFrame | np.ndarray] | list[pd.DataFrame | np.ndarray],
    ) -> pd.DataFrame:
        """Horizontally concatenates multiple modal representations into a single DataFrame.
        
        Args:
            inputs: Either a dictionary mapping branch names to data arrays/DataFrames,
                    or a list of data arrays/DataFrames.
                    
        Returns:
            pd.DataFrame: Merged feature table of shape (N, total_features).
            
        Raises:
            ValueError: If input is empty, has differing row counts, or unsupported format.
        """
        if not inputs:
            raise ValueError("FeatureFusionNode received empty inputs; at least one representation is required.")

        normalized_inputs: list[tuple[str, pd.DataFrame]] = []

        if isinstance(inputs, dict):
            items = list(inputs.items())
        else:
            items = [(f"{self.default_prefix}_{idx}", item) for idx, item in enumerate(inputs)]

        expected_rows: int | None = None
        base_index: pd.Index | None = None

        for name, data in items:
            df = self._to_dataframe(data, prefix=name)
            n_rows = len(df)

            if expected_rows is None:
                expected_rows = n_rows
                base_index = df.index
            elif n_rows != expected_rows:
                raise ValueError(
                    f"Row count mismatch in FeatureFusionNode '{self.node_id}': "
                    f"input '{name}' has {n_rows} rows, but previous inputs have {expected_rows} rows."
                )

            normalized_inputs.append((name, df))

        assert expected_rows is not None

        # Build concatenated dataframe with disambiguated columns
        dfs_to_concat: list[pd.DataFrame] = []
        seen_columns: set[str] = set()

        for name, df in normalized_inputs:
            # Reindex to base_index if indices differ in values
            if base_index is not None and not df.index.equals(base_index):
                df = df.copy()
                df.index = base_index

            new_columns = []
            for col in df.columns:
                col_name = str(col)
                if col_name in seen_columns:
                    col_name = f"{name}_{col_name}"
                seen_columns.add(col_name)
                new_columns.append(col_name)

            df.columns = pd.Index(new_columns)
            dfs_to_concat.append(df)

        return pd.concat(dfs_to_concat, axis=1)

    def _to_dataframe(self, data: pd.DataFrame | np.ndarray | Any, prefix: str) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            return data.to_frame(name=prefix)

        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            # Flatten higher dimensional representations to 2D
            arr = arr.reshape(arr.shape[0], -1)

        cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
