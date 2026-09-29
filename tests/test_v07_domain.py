from __future__ import annotations

import sys
from typing import Any
import pytest

from automl.domain.modalities import DataSource, Modality
from automl.domain.pipelines import NodeType, PipelineEdge, PipelineGraph, PipelineNode
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModalityPluginPort, PluginPort


def test_modality_enum_values() -> None:
    assert Modality.TABULAR == "tabular"
    assert Modality.IMAGE == "image"
    assert Modality.TEXT == "text"
    assert Modality.AUDIO == "audio"
    assert Modality.TIMESERIES == "timeseries"
    assert str(Modality.IMAGE) == "image"


def test_data_source_creation() -> None:
    source = DataSource(
        id="img_src_1",
        modality=Modality.IMAGE,
        path="data/images/",
        metadata={"format": "png", "channels": 3},
    )
    assert source.id == "img_src_1"
    assert source.modality == Modality.IMAGE
    assert source.path == "data/images/"
    assert source.metadata["format"] == "png"


def test_pipeline_graph_construction_and_traversal() -> None:
    graph = PipelineGraph(id="multimodal_dag_1", name="Tabular + Image DAG")

    # Source nodes
    tab_src = PipelineNode(
        node_id="tab_src",
        node_type=NodeType.SOURCE,
        input_modalities=(),
        output_modality=Modality.TABULAR,
    )
    img_src = PipelineNode(
        node_id="img_src",
        node_type=NodeType.SOURCE,
        input_modalities=(),
        output_modality=Modality.IMAGE,
    )

    # Image encoder node
    img_enc = PipelineNode(
        node_id="img_enc",
        node_type=NodeType.ENCODER,
        input_modalities=(Modality.IMAGE,),
        output_modality=Modality.TABULAR,  # embeddings are numerical tabular vectors
        parameters={"model_name": "resnet18", "dim": 512},
    )

    # Fusion node
    fusion = PipelineNode(
        node_id="fusion",
        node_type=NodeType.FUSION,
        input_modalities=(Modality.TABULAR, Modality.TABULAR),
        output_modality=Modality.TABULAR,
        parameters={"strategy": "concat"},
    )

    # Model node
    model = PipelineNode(
        node_id="clf_model",
        node_type=NodeType.MODEL,
        input_modalities=(Modality.TABULAR,),
        output_modality=Modality.TABULAR,
        parameters={"model_id": "lightgbm"},
    )

    for node in [tab_src, img_src, img_enc, fusion, model]:
        graph.add_node(node)

    # Connect DAG edges
    graph.add_edge("img_src", "img_enc")
    graph.add_edge("img_enc", "fusion")
    graph.add_edge("tab_src", "fusion")
    graph.add_edge("fusion", "clf_model")

    assert len(graph.nodes) == 5
    assert len(graph.edges) == 4

    # Traversal tests
    source_nodes = graph.get_source_nodes()
    assert {n.node_id for n in source_nodes} == {"tab_src", "img_src"}

    terminal_nodes = graph.get_terminal_nodes()
    assert {n.node_id for n in terminal_nodes} == {"clf_model"}

    incoming_to_fusion = graph.get_incoming_nodes("fusion")
    assert {n.node_id for n in incoming_to_fusion} == {"img_enc", "tab_src"}

    outgoing_from_img_src = graph.get_outgoing_nodes("img_src")
    assert [n.node_id for n in outgoing_from_img_src] == ["img_enc"]


def test_pipeline_graph_errors() -> None:
    graph = PipelineGraph(id="g", name="test")
    node = PipelineNode("n1", NodeType.SOURCE, (), Modality.TABULAR)
    graph.add_node(node)

    # Duplicate node
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(node)

    # Invalid edges
    with pytest.raises(KeyError, match="Source node 'nonexistent'"):
        graph.add_edge("nonexistent", "n1")

    with pytest.raises(KeyError, match="Target node 'nonexistent'"):
        graph.add_edge("n1", "nonexistent")


def test_modality_plugin_protocol_compliance() -> None:
    class DummyImagePlugin(ModalityPluginPort):
        plugin_id = "mock_vision"
        name = "Mock Vision Plugin"
        version = "1.0.0"
        plugin_type = PluginType.PREPROCESSOR
        capabilities = PluginCapability(
            supported_modalities=(Modality.IMAGE,)
        )
        modality = Modality.IMAGE

        def validate_source(self, source: DataSource) -> bool:
            return source.modality == Modality.IMAGE

        def load_data(self, source: DataSource) -> Any:
            return f"loaded_{source.path}"

    plugin = DummyImagePlugin()
    assert plugin.plugin_id == "mock_vision"
    assert plugin.modality == Modality.IMAGE
    assert plugin.validate_source(DataSource("s1", Modality.IMAGE, "path/to/img.png")) is True
    assert plugin.load_data(DataSource("s1", Modality.IMAGE, "test.png")) == "loaded_test.png"


def test_domain_purity_no_framework_dependencies() -> None:
    import automl.domain.modalities as dm
    import automl.domain.pipelines as dp

    forbidden_modules = ["sklearn", "pandas", "numpy", "torch", "scipy", "sqlite3"]
    for mod_name in forbidden_modules:
        assert mod_name not in dm.__dict__
        assert mod_name not in dp.__dict__
