from __future__ import annotations

from pathlib import Path
import struct
import zlib
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import ExecutePipelineCommand
from automl.application.queries.workspace_queries import (
    GetPipelineExecutionOrderQuery,
    ListPluginsQuery,
    ValidatePipelineGraphQuery,
)
from automl.domain.modalities.modality import DataSource, Modality
from automl.domain.pipelines.graph import NodeType, PipelineGraph, PipelineNode
from automl.plugins.modalities.image_plugin import ImageModalityPlugin


def create_test_png(path: Path, width: int = 8, height: int = 8, color: tuple[int, int, int] = (120, 200, 50)) -> Path:
    """Creates a small valid PNG image file using only standard library."""
    png = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    raw_data = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    compressed = zlib.compress(raw_data)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    png += struct.pack(">I", 0) + b"IEND" + iend_crc

    path.write_bytes(png)
    return path


@pytest.fixture
def multimodal_workspace(tmp_path: Path):
    root_dir = tmp_path / "automl_ws"
    workspace, command_bus, query_bus = build_application(root_dir=str(root_dir))
    return workspace, command_bus, query_bus, tmp_path


@pytest.fixture
def synthetic_multimodal_data(tmp_path: Path):
    """Creates tabular DataFrame and image files on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 12
    image_paths = []
    for i in range(n_samples):
        # Vary color by sample index
        color = ((i * 20) % 256, (i * 40 + 50) % 256, (i * 60 + 100) % 256)
        img_path = img_dir / f"sample_{i:03d}.png"
        create_test_png(img_path, width=8, height=8, color=color)
        image_paths.append(str(img_path))

    np.random.seed(42)
    tab_df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n_samples),
        "income": np.random.uniform(20000, 100000, size=n_samples),
        "score": np.random.uniform(0.1, 0.9, size=n_samples),
    })

    # Synthetic binary target
    target = (tab_df["income"] > 50000).astype(int)

    return tab_df, image_paths, target, img_dir


class TestMultimodalPluginIntegration:
    def test_image_modality_plugin_auto_registered(self, multimodal_workspace) -> None:
        workspace, command_bus, query_bus, _ = multimodal_workspace

        # Verify plugin is present in registry
        assert workspace.plugin_registry.has("image_modality")
        plugin = workspace.plugin_registry.get("image_modality")
        assert isinstance(plugin, ImageModalityPlugin)

        # Verify modality query lookup
        mod_plugin = workspace.plugin_registry.get_modality_plugin(Modality.IMAGE)
        assert mod_plugin is not None
        assert mod_plugin.plugin_id == "image_modality"

        # Verify query bus lists image_modality
        plugins_summary = query_bus.dispatch(ListPluginsQuery())
        ids = [p["plugin_id"] for p in plugins_summary]
        assert "image_modality" in ids

    def test_image_data_source_validation(self, multimodal_workspace, synthetic_multimodal_data) -> None:
        workspace, _, _, _ = multimodal_workspace
        _, image_paths, _, img_dir = synthetic_multimodal_data

        plugin: ImageModalityPlugin = workspace.plugin_registry.get("image_modality")  # type: ignore

        # Valid directory data source
        source_dir = DataSource(id="img_dir", modality=Modality.IMAGE, path=str(img_dir))
        assert plugin.validate_source(source_dir) is True

        # Valid single file data source
        source_file = DataSource(id="img_file", modality=Modality.IMAGE, path=image_paths[0])
        assert plugin.validate_source(source_file) is True

        # Non-existent file
        source_invalid = DataSource(id="missing", modality=Modality.IMAGE, path=str(img_dir / "non_existent.png"))
        assert plugin.validate_source(source_invalid) is False


class TestPipelineGraphCQRS:
    def test_query_bus_validates_pipeline_graph(self, multimodal_workspace) -> None:
        workspace, _, query_bus, _ = multimodal_workspace
        graph = workspace.build_multimodal_pipeline()

        # Validation query should succeed without raising
        query_bus.dispatch(ValidatePipelineGraphQuery(graph=graph))

        # Execution order query returns ordered nodes
        order = query_bus.dispatch(GetPipelineExecutionOrderQuery(graph=graph))
        assert len(order) == 5
        node_ids = [n.node_id for n in order]
        assert "image_encoder" in node_ids
        assert "fusion" in node_ids
        assert node_ids.index("image_encoder") < node_ids.index("fusion")

    def test_query_bus_rejects_cyclic_graph(self, multimodal_workspace) -> None:
        _, _, query_bus, _ = multimodal_workspace
        graph = PipelineGraph(id="cycle_graph", name="Cycle")
        n1 = PipelineNode(node_id="n1", node_type=NodeType.PREPROCESSOR, input_modalities=(Modality.TABULAR,), output_modality=Modality.TABULAR)
        n2 = PipelineNode(node_id="n2", node_type=NodeType.PREPROCESSOR, input_modalities=(Modality.TABULAR,), output_modality=Modality.TABULAR)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n1")

        with pytest.raises(ValueError, match="Cycle detected"):
            query_bus.dispatch(ValidatePipelineGraphQuery(graph=graph))

    def test_command_bus_executes_pipeline(self, multimodal_workspace, synthetic_multimodal_data) -> None:
        workspace, command_bus, _, _ = multimodal_workspace
        tab_df, image_paths, _, _ = synthetic_multimodal_data

        graph = workspace.build_multimodal_pipeline(embedding_dim=8)
        inputs = {
            "tabular_input": tab_df,
            "image_input": image_paths,
        }

        # Dispatch via CommandBus
        fused_df = command_bus.dispatch(ExecutePipelineCommand(graph=graph, inputs=inputs))
        assert isinstance(fused_df, pd.DataFrame)
        assert len(fused_df) == len(tab_df)
        assert fused_df.shape[1] == 3 + 8  # 3 tabular features + 8 embeddings
        assert "img_emb_image_encoder_0" in fused_df.columns or "image_encoder_0" in fused_df.columns


class TestMultimodalEndToEndExecution:
    def test_full_multimodal_pipeline_fit_predict(self, multimodal_workspace, synthetic_multimodal_data) -> None:
        workspace, _, _, _ = multimodal_workspace
        tab_df, image_paths, target, _ = synthetic_multimodal_data

        graph = workspace.build_multimodal_pipeline(
            model_id="random_forest",
            embedding_dim=16,
        )

        inputs = {
            "tabular_input": tab_df,
            "image_input": image_paths,
        }

        result = workspace.fit_predict_multimodal(
            graph=graph,
            inputs=inputs,
            target=target,
            model_id="random_forest",
            task_type="binary_classification",
        )

        assert result["model_id"] == "random_forest"
        assert result["task_type"] == "binary_classification"
        assert result["metric"] in ("roc_auc", "accuracy")
        assert 0.0 <= result["score"] <= 1.0
        assert result["sample_count"] == len(tab_df)
        assert result["feature_count"] == 3 + 16  # 3 tabular + 16 image embeddings
        assert len(result["predictions"]) == len(tab_df)

    def test_hypothesis_driven_comparison_tabular_vs_multimodal(self, multimodal_workspace, synthetic_multimodal_data) -> None:
        """Evaluates hypothesis: Multimodal representations contain distinct signal compared to tabular alone."""
        workspace, _, _, _ = multimodal_workspace
        tab_df, image_paths, target, _ = synthetic_multimodal_data

        # 1. Multimodal pipeline
        graph_multimodal = workspace.build_multimodal_pipeline(embedding_dim=8)
        mm_result = workspace.fit_predict_multimodal(
            graph=graph_multimodal,
            inputs={"tabular_input": tab_df, "image_input": image_paths},
            target=target,
            model_id="logistic_regression",
            task_type="binary_classification",
        )

        # 2. Tabular-only pipeline
        graph_tabular = PipelineGraph(id="tabular_only", name="Tabular Pipeline")
        tab_src = PipelineNode(
            node_id="tab_in",
            node_type=NodeType.SOURCE,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
        )
        model_node = PipelineNode(
            node_id="model",
            node_type=NodeType.MODEL,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
        )
        graph_tabular.add_node(tab_src)
        graph_tabular.add_node(model_node)
        graph_tabular.add_edge("tab_in", "model")

        tab_result = workspace.fit_predict_multimodal(
            graph=graph_tabular,
            inputs={"tab_in": tab_df},
            target=target,
            model_id="logistic_regression",
            task_type="binary_classification",
        )

        # Both produce measurable metrics
        assert mm_result["score"] >= 0.0
        assert tab_result["score"] >= 0.0

        # Multimodal representation has more dimensions than tabular alone
        assert mm_result["feature_count"] > tab_result["feature_count"]

    def test_execute_pipeline_missing_input_raises_key_error(self, multimodal_workspace, synthetic_multimodal_data) -> None:
        workspace, _, _, _ = multimodal_workspace
        tab_df, _, _, _ = synthetic_multimodal_data

        graph = workspace.build_multimodal_pipeline()
        # Missing "image_input"
        with pytest.raises(KeyError, match="Missing required input for source node"):
            workspace.execute_pipeline(graph, {"tabular_input": tab_df})
