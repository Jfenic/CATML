from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from automl.domain.modalities.modality import Modality


class NodeType(str, Enum):
    """Categorization of pipeline graph stages."""
    SOURCE = "source"
    ENCODER = "encoder"
    PREPROCESSOR = "preprocessor"
    FUSION = "fusion"
    MODEL = "model"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PipelineNode:
    """Represents an atomic execution unit in a pipeline DAG."""
    node_id: str
    node_type: str | NodeType
    input_modalities: tuple[Modality, ...]
    output_modality: Modality
    name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEdge:
    """Directed connection from source_node to target_node."""
    source_node_id: str
    target_node_id: str


@dataclass
class PipelineGraph:
    """Encapsulates a Directed Acyclic Graph (DAG) for multi-modal processing."""
    id: str
    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[PipelineEdge] = field(default_factory=list)

    def add_node(self, node: PipelineNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in pipeline graph '{self.id}'.")
        self.nodes[node.node_id] = node

    def add_edge(self, source_node_id: str, target_node_id: str) -> None:
        if source_node_id not in self.nodes:
            raise KeyError(f"Source node '{source_node_id}' not found in pipeline graph.")
        if target_node_id not in self.nodes:
            raise KeyError(f"Target node '{target_node_id}' not found in pipeline graph.")
        edge = PipelineEdge(source_node_id=source_node_id, target_node_id=target_node_id)
        if edge not in self.edges:
            self.edges.append(edge)

    def get_node(self, node_id: str) -> PipelineNode | None:
        return self.nodes.get(node_id)

    def get_incoming_nodes(self, node_id: str) -> list[PipelineNode]:
        source_ids = [e.source_node_id for e in self.edges if e.target_node_id == node_id]
        return [self.nodes[sid] for sid in source_ids if sid in self.nodes]

    def get_outgoing_nodes(self, node_id: str) -> list[PipelineNode]:
        target_ids = [e.target_node_id for e in self.edges if e.source_node_id == node_id]
        return [self.nodes[tid] for tid in target_ids if tid in self.nodes]

    def get_source_nodes(self) -> list[PipelineNode]:
        """Nodes with no incoming edges (entry points)."""
        nodes_with_incoming = {e.target_node_id for e in self.edges}
        return [node for nid, node in self.nodes.items() if nid not in nodes_with_incoming]

    def get_terminal_nodes(self) -> list[PipelineNode]:
        """Nodes with no outgoing edges (exit points / models)."""
        nodes_with_outgoing = {e.source_node_id for e in self.edges}
        return [node for nid, node in self.nodes.items() if nid not in nodes_with_outgoing]
