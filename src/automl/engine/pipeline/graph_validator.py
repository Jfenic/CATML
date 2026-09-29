from __future__ import annotations

from collections import deque
from typing import Any

from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineGraph, PipelineNode


class GraphValidator:
    """Validates structural integrity, absence of cycles, and modal typing in a PipelineGraph."""

    def validate(self, graph: PipelineGraph) -> None:
        """Runs full validation suite on the graph. Raises ValueError on any inconsistency."""
        self._validate_non_empty(graph)
        self._detect_cycles(graph)
        self._validate_connectivity(graph)
        self._validate_modality_compatibility(graph)

    def _validate_non_empty(self, graph: PipelineGraph) -> None:
        if not graph.nodes:
            raise ValueError(f"Pipeline graph '{graph.id}' cannot be empty; it must contain at least one node.")

    def _validate_connectivity(self, graph: PipelineGraph) -> None:
        sources = graph.get_source_nodes()
        terminals = graph.get_terminal_nodes()

        if not sources:
            raise ValueError(f"Pipeline graph '{graph.id}' must have at least one source node (no incoming edges).")

        if not terminals:
            raise ValueError(f"Pipeline graph '{graph.id}' must have at least one terminal node (no outgoing edges).")

        # In graphs with multiple nodes, ensure there are no isolated nodes (nodes with 0 incoming AND 0 outgoing edges)
        if len(graph.nodes) > 1:
            for node_id, node in graph.nodes.items():
                incoming = graph.get_incoming_nodes(node_id)
                outgoing = graph.get_outgoing_nodes(node_id)
                if not incoming and not outgoing:
                    raise ValueError(
                        f"Isolated node detected: node '{node_id}' in graph '{graph.id}' has no incoming or outgoing connections."
                    )

    def _detect_cycles(self, graph: PipelineGraph) -> None:
        """Detects directed cycles using depth-first search with 3-color state tracking."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node_id: WHITE for node_id in graph.nodes}
        parent: dict[str, str | None] = {node_id: None for node_id in graph.nodes}

        def dfs(node_id: str, path: list[str]) -> None:
            color[node_id] = GRAY
            path.append(node_id)

            for outgoing in graph.get_outgoing_nodes(node_id):
                neighbor_id = outgoing.node_id
                if color[neighbor_id] == GRAY:
                    # Cycle found - reconstruct the cycle path
                    cycle_start_idx = path.index(neighbor_id)
                    cycle_loop = path[cycle_start_idx:] + [neighbor_id]
                    cycle_str = " -> ".join(cycle_loop)
                    raise ValueError(
                        f"Cycle detected in pipeline graph '{graph.id}': {cycle_str}. Pipelines must be acyclic (DAG)."
                    )
                if color[neighbor_id] == WHITE:
                    parent[neighbor_id] = node_id
                    dfs(neighbor_id, path)

            path.pop()
            color[node_id] = BLACK

        for node_id in graph.nodes:
            if color[node_id] == WHITE:
                dfs(node_id, [])

    def _validate_modality_compatibility(self, graph: PipelineGraph) -> None:
        """Ensures that the output modality of the source matches an accepted input modality of the target."""
        for edge in graph.edges:
            src_node = graph.get_node(edge.source_node_id)
            tgt_node = graph.get_node(edge.target_node_id)

            if src_node is None or tgt_node is None:
                continue

            src_out = src_node.output_modality
            tgt_inputs = tgt_node.input_modalities

            if src_out not in tgt_inputs:
                raise ValueError(
                    f"Modality mismatch between node '{src_node.node_id}' and '{tgt_node.node_id}': "
                    f"source '{src_node.node_id}' outputs '{src_out}', but target '{tgt_node.node_id}' "
                    f"only accepts {[str(m) for m in tgt_inputs]}."
                )

    def get_execution_order(self, graph: PipelineGraph) -> list[PipelineNode]:
        """Calculates topological sort of nodes. Preconditions: Graph must be a valid DAG."""
        self.validate(graph)

        # Kahn's algorithm for topological sorting
        in_degree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        for edge in graph.edges:
            in_degree[edge.target_node_id] += 1

        queue: deque[str] = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        ordered_nodes: list[PipelineNode] = []

        while queue:
            current_id = queue.popleft()
            current_node = graph.get_node(current_id)
            if current_node:
                ordered_nodes.append(current_node)

            for outgoing in graph.get_outgoing_nodes(current_id):
                in_degree[outgoing.node_id] -= 1
                if in_degree[outgoing.node_id] == 0:
                    queue.append(outgoing.node_id)

        if len(ordered_nodes) != len(graph.nodes):
            raise ValueError(f"Failed to establish topological execution order in pipeline graph '{graph.id}'.")

        return ordered_nodes
