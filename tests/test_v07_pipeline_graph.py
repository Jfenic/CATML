import numpy as np
import pandas as pd
import pytest

from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineGraph, PipelineNode
from automl.engine.pipeline.fusion import FeatureFusionNode
from automl.engine.pipeline.graph_validator import GraphValidator


def _create_node(
    node_id: str,
    node_type: NodeType = NodeType.PREPROCESSOR,
    input_modalities: tuple[Modality, ...] | None = None,
    output_modality: Modality = Modality.TABULAR,
) -> PipelineNode:
    return PipelineNode(
        node_id=node_id,
        node_type=node_type,
        name=node_id.replace("_", " ").title(),
        input_modalities=input_modalities if input_modalities is not None else (Modality.TABULAR,),
        output_modality=output_modality,
    )


class TestGraphValidator:
    def test_empty_graph_fails(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="empty_graph", name="Empty")
        with pytest.raises(ValueError, match="cannot be empty"):
            validator.validate(graph)

    def test_single_node_valid(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="single_node_graph", name="Single Node")
        node = _create_node("input_node", NodeType.SOURCE)
        graph.add_node(node)

        # Should pass validation and return order of 1 node
        validator.validate(graph)
        order = validator.get_execution_order(graph)
        assert len(order) == 1
        assert order[0].node_id == "input_node"

    def test_isolated_node_fails_in_multi_node_graph(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="isolated_test", name="Isolated Test")
        n1 = _create_node("n1", NodeType.SOURCE)
        n2 = _create_node("n2", NodeType.MODEL)
        n3 = _create_node("n3", NodeType.PREPROCESSOR)  # Isolated

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge("n1", "n2")

        with pytest.raises(ValueError, match="Isolated node detected: node 'n3'"):
            validator.validate(graph)

    def test_direct_cycle_detected(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="direct_cycle", name="Direct Cycle")
        n1 = _create_node("n1")
        n2 = _create_node("n2")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n1")

        with pytest.raises(ValueError, match=r"Cycle detected.*n1 -> n2 -> n1|Cycle detected.*n2 -> n1 -> n2"):
            validator.validate(graph)

    def test_indirect_cycle_detected(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="indirect_cycle", name="Indirect Cycle")
        n1 = _create_node("n1")
        n2 = _create_node("n2")
        n3 = _create_node("n3")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        graph.add_edge("n3", "n1")

        with pytest.raises(ValueError, match="Cycle detected"):
            validator.validate(graph)

    def test_modality_mismatch_fails(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="modality_mismatch", name="Modality Mismatch")
        image_node = _create_node("image_in", NodeType.SOURCE, output_modality=Modality.IMAGE)
        tabular_node = _create_node("tabular_prep", NodeType.PREPROCESSOR, input_modalities=(Modality.TABULAR,))

        graph.add_node(image_node)
        graph.add_node(tabular_node)
        graph.add_edge("image_in", "tabular_prep")

        with pytest.raises(ValueError, match="Modality mismatch between node 'image_in' and 'tabular_prep'"):
            validator.validate(graph)

    def test_diamond_graph_valid_execution_order(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="diamond_dag", name="Diamond DAG")
        # Diamond: start -> branch_a, branch_b -> merge
        start = _create_node("start", NodeType.SOURCE)
        branch_a = _create_node("branch_a", NodeType.PREPROCESSOR)
        branch_b = _create_node("branch_b", NodeType.PREPROCESSOR)
        merge = _create_node("merge", NodeType.MODEL)

        for n in [start, branch_a, branch_b, merge]:
            graph.add_node(n)

        graph.add_edge("start", "branch_a")
        graph.add_edge("start", "branch_b")
        graph.add_edge("branch_a", "merge")
        graph.add_edge("branch_b", "merge")

        validator.validate(graph)
        order = validator.get_execution_order(graph)
        order_ids = [n.node_id for n in order]

        assert order_ids[0] == "start"
        assert order_ids[-1] == "merge"
        assert set(order_ids[1:3]) == {"branch_a", "branch_b"}

    def test_multimodal_dag_execution_order(self) -> None:
        validator = GraphValidator()
        graph = PipelineGraph(id="multimodal_dag", name="Multimodal Pipeline")

        tab_in = _create_node("tab_in", NodeType.SOURCE, output_modality=Modality.TABULAR)
        img_in = _create_node("img_in", NodeType.SOURCE, output_modality=Modality.IMAGE)
        img_encoder = _create_node(
            "img_encoder",
            NodeType.ENCODER,
            input_modalities=(Modality.IMAGE,),
            output_modality=Modality.TABULAR,
        )
        fusion = _create_node(
            "fusion",
            NodeType.FUSION,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
        )
        classifier = _create_node("classifier", NodeType.MODEL, input_modalities=(Modality.TABULAR,))

        for n in [tab_in, img_in, img_encoder, fusion, classifier]:
            graph.add_node(n)

        graph.add_edge("tab_in", "fusion")
        graph.add_edge("img_in", "img_encoder")
        graph.add_edge("img_encoder", "fusion")
        graph.add_edge("fusion", "classifier")

        validator.validate(graph)
        order = validator.get_execution_order(graph)
        order_ids = [n.node_id for n in order]

        # Sources must come before downstream nodes
        assert order_ids.index("img_in") < order_ids.index("img_encoder")
        assert order_ids.index("img_encoder") < order_ids.index("fusion")
        assert order_ids.index("tab_in") < order_ids.index("fusion")
        assert order_ids.index("fusion") < order_ids.index("classifier")


class TestFeatureFusionNode:
    def test_fuse_empty_inputs_raises_value_error(self) -> None:
        node = FeatureFusionNode()
        with pytest.raises(ValueError, match="empty inputs"):
            node.fuse({})

        with pytest.raises(ValueError, match="empty inputs"):
            node.fuse([])

    def test_fuse_dataframes(self) -> None:
        node = FeatureFusionNode()
        df1 = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        df2 = pd.DataFrame({"feat_c": [7, 8, 9]})

        fused = node.fuse({"tab1": df1, "tab2": df2})
        assert fused.shape == (3, 3)
        assert list(fused.columns) == ["feat_a", "feat_b", "feat_c"]
        assert (fused["feat_c"] == [7, 8, 9]).all()

    def test_fuse_dataframe_and_dense_array(self) -> None:
        node = FeatureFusionNode()
        df = pd.DataFrame({"age": [25, 30, 45], "income": [50000, 60000, 80000]})
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        fused = node.fuse({"tabular": df, "image_emb": embeddings})
        assert fused.shape == (3, 5)
        assert list(fused.columns) == ["age", "income", "image_emb_0", "image_emb_1", "image_emb_2"]
        assert fused.iloc[0]["image_emb_1"] == 0.2

    def test_fuse_with_list_input(self) -> None:
        node = FeatureFusionNode(default_prefix="branch")
        arr1 = np.ones((4, 2))
        arr2 = np.zeros((4, 3))

        fused = node.fuse([arr1, arr2])
        assert fused.shape == (4, 5)
        assert list(fused.columns) == [
            "branch_0_0", "branch_0_1",
            "branch_1_0", "branch_1_1", "branch_1_2",
        ]

    def test_fuse_handles_1d_and_series(self) -> None:
        node = FeatureFusionNode()
        s = pd.Series([10, 20, 30], name="score")
        arr_1d = np.array([100, 200, 300])

        fused = node.fuse({"series_branch": s, "arr_branch": arr_1d})
        assert fused.shape == (3, 2)
        assert "score" in fused.columns or "series_branch" in fused.columns
        assert "arr_branch_0" in fused.columns

    def test_fuse_handles_3d_flattening(self) -> None:
        node = FeatureFusionNode()
        arr_3d = np.zeros((5, 2, 4))  # 5 samples, flattened to 8 features

        fused = node.fuse({"spatial_emb": arr_3d})
        assert fused.shape == (5, 8)
        assert list(fused.columns) == [f"spatial_emb_{i}" for i in range(8)]

    def test_fuse_column_name_collision_disambiguation(self) -> None:
        node = FeatureFusionNode()
        df1 = pd.DataFrame({"feature": [1, 2], "value": [10, 20]})
        df2 = pd.DataFrame({"feature": [3, 4], "value": [30, 40]})

        fused = node.fuse({"b1": df1, "b2": df2})
        assert fused.shape == (2, 4)
        assert list(fused.columns) == ["feature", "value", "b2_feature", "b2_value"]

    def test_fuse_mismatched_row_counts_raises_value_error(self) -> None:
        node = FeatureFusionNode()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2]})

        with pytest.raises(ValueError, match=r"Row count mismatch.*input 'branch_2' has 2 rows.*3 rows"):
            node.fuse({"branch_1": df1, "branch_2": df2})

    def test_fuse_reindexes_to_base_index(self) -> None:
        node = FeatureFusionNode()
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["row1", "row2", "row3"])
        df2 = pd.DataFrame({"b": [4, 5, 6]}, index=[0, 1, 2])

        fused = node.fuse({"first": df1, "second": df2})
        assert list(fused.index) == ["row1", "row2", "row3"]
        assert fused["b"].tolist() == [4, 5, 6]
