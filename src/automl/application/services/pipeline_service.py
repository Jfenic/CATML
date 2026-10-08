from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineGraph, PipelineNode
from automl.domain.tasks.task_type import default_metric_for
from automl.engine.pipeline.fusion import FeatureFusionNode
from automl.engine.pipeline.graph_validator import GraphValidator
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.engine.vision.image_encoder import ImageEncoderNode

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class PipelineExecutionService:
    """Application service for validating, building, and executing multimodal pipeline DAGs."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def validate_pipeline_graph(self, graph: PipelineGraph) -> None:
        """Validates that a pipeline graph is well-formed, acyclic, and modality-consistent."""
        validator = GraphValidator()
        validator.validate(graph)

    def get_pipeline_execution_order(self, graph: PipelineGraph) -> list[PipelineNode]:
        """Returns the topologically sorted execution order of nodes in the pipeline graph."""
        validator = GraphValidator()
        return validator.get_execution_order(graph)

    def build_multimodal_pipeline(
        self,
        tabular_source_id: str = "tabular_input",
        image_source_id: str = "image_input",
        model_id: str = "random_forest",
        embedding_dim: int = 16,
        pipeline_id: str = "multimodal_pipeline",
        pipeline_name: str = "Multimodal Tabular + Vision Pipeline",
    ) -> PipelineGraph:
        """Constructs a validated standard multimodal DAG with tabular and vision branches."""
        graph = PipelineGraph(id=pipeline_id, name=pipeline_name)

        tab_node = PipelineNode(
            node_id=tabular_source_id,
            node_type=NodeType.SOURCE,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
            name="Tabular Data Source",
        )
        img_node = PipelineNode(
            node_id=image_source_id,
            node_type=NodeType.SOURCE,
            input_modalities=(Modality.IMAGE,),
            output_modality=Modality.IMAGE,
            name="Image Data Source",
        )
        encoder_node = ImageEncoderNode(
            node_id="image_encoder",
            output_dim=embedding_dim,
        ).to_pipeline_node()
        fusion_node = FeatureFusionNode(
            node_id="fusion",
            default_prefix="img_emb",
        ).to_pipeline_node()
        model_node = PipelineNode(
            node_id="model",
            node_type=NodeType.MODEL,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
            name=f"Model({model_id})",
            parameters={"model_id": model_id},
        )

        for n in [tab_node, img_node, encoder_node, fusion_node, model_node]:
            graph.add_node(n)

        graph.add_edge(image_source_id, "image_encoder")
        graph.add_edge(tabular_source_id, "fusion")
        graph.add_edge("image_encoder", "fusion")
        graph.add_edge("fusion", "model")

        self.validate_pipeline_graph(graph)
        return graph

    def execute_pipeline(self, graph: PipelineGraph, inputs: dict[str, Any]) -> Any:
        """Executes a pipeline DAG given modal inputs, returning the terminal node's output."""
        validator = GraphValidator()
        execution_order = validator.get_execution_order(graph)

        node_outputs: dict[str, Any] = {}

        for node in execution_order:
            if node.node_type == NodeType.SOURCE:
                data = inputs.get(node.node_id)
                if data is None:
                    for k, v in inputs.items():
                        if k in node.node_id or (
                            hasattr(node.output_modality, "value") and k == node.output_modality.value
                        ):
                            data = v
                            break
                if data is None:
                    raise KeyError(f"Missing required input for source node '{node.node_id}' in pipeline execution.")
                node_outputs[node.node_id] = data

            elif node.node_type == NodeType.ENCODER:
                incoming = graph.get_incoming_nodes(node.node_id)
                if not incoming:
                    raise ValueError(f"Encoder node '{node.node_id}' has no incoming inputs.")
                src_data = node_outputs[incoming[0].node_id]
                encoder = ImageEncoderNode.from_pipeline_node(node)
                node_outputs[node.node_id] = encoder.transform(src_data)

            elif node.node_type == NodeType.FUSION:
                incoming = graph.get_incoming_nodes(node.node_id)
                if not incoming:
                    raise ValueError(f"Fusion node '{node.node_id}' has no incoming inputs.")
                fusion_dict = {inc.node_id: node_outputs[inc.node_id] for inc in incoming}
                fusion = FeatureFusionNode.from_pipeline_node(node)
                node_outputs[node.node_id] = fusion.fuse(fusion_dict)

            elif node.node_type == NodeType.PREPROCESSOR:
                incoming = graph.get_incoming_nodes(node.node_id)
                src_data = node_outputs[incoming[0].node_id]
                node_outputs[node.node_id] = src_data

            elif node.node_type == NodeType.MODEL:
                incoming = graph.get_incoming_nodes(node.node_id)
                src_data = node_outputs[incoming[0].node_id]
                node_outputs[node.node_id] = src_data

        terminal_nodes = graph.get_terminal_nodes()
        if len(terminal_nodes) == 1:
            return node_outputs[terminal_nodes[0].node_id]
        return {tn.node_id: node_outputs[tn.node_id] for tn in terminal_nodes}

    def fit_predict_multimodal(
        self,
        graph: PipelineGraph,
        inputs: dict[str, Any],
        target: Any,
        model_id: str = "random_forest",
        task_type: str = "binary_classification",
    ) -> dict[str, Any]:
        """Trains a model on the fused representations produced by a multimodal DAG pipeline."""
        fused_df = self.execute_pipeline(graph, inputs)
        if not isinstance(fused_df, pd.DataFrame):
            fused_df = pd.DataFrame(fused_df)

        y = np.asarray(target)
        model_plugin = self.workspace.plugin_registry.get_model_plugin(model_id)

        if model_plugin is not None:
            model = model_plugin.build_estimator()
            model.fit(fused_df, y)
            preds = model.predict(fused_df)
        else:
            trainer = SklearnTrainer(model_registry=self.workspace.model_registry)
            trial_exec = trainer.train(
                task_type=task_type,
                model_id=model_id,
                hyperparameters={},
                X_train=fused_df,
                y_train=y,
                metric=default_metric_for(task_type),
            )
            model = trial_exec.model
            preds = model.predict(fused_df)

        metric_name = default_metric_for(task_type)
        if task_type == "binary_classification":
            from sklearn.metrics import accuracy_score

            score = float(accuracy_score(y, preds))
        elif task_type == "regression":
            from sklearn.metrics import r2_score

            score = float(r2_score(y, preds))
        else:
            from sklearn.metrics import accuracy_score

            score = float(accuracy_score(y, preds))

        return {
            "model_id": model_id,
            "task_type": task_type,
            "metric": metric_name,
            "score": score,
            "fused_feature_names": list(fused_df.columns),
            "feature_count": fused_df.shape[1],
            "sample_count": fused_df.shape[0],
            "predictions": preds,
            "graph_id": graph.id,
        }
