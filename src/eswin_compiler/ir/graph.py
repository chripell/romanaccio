"""Computational Directed Acyclic Graph (DAG) for ESWIN IR (EIR)."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from .node import EIRNode
from .op_types import EngineType, OpType
from .tensor import EIRTensor


class EIRGraph:
    """Represents a neural network computational graph in EIR.
    
    Contains nodes, tensors, user-visible network inputs/outputs,
    and provides topological sorting and graph integrity validation.
    """

    def __init__(self, name: str = "eswin_graph"):
        self.name: str = name
        self.inputs: List[EIRTensor] = []
        self.outputs: List[EIRTensor] = []
        self.nodes: List[EIRNode] = []
        self.tensors: Dict[str, EIRTensor] = {}

    def add_tensor(self, tensor: EIRTensor) -> EIRTensor:
        """Registers a tensor in the graph dictionary. Returns the tensor."""
        self.tensors[tensor.name] = tensor
        return tensor

    def get_or_create_tensor(self, name: str, shape=(1, 1, 1, 1), dtype="float16") -> EIRTensor:
        """Retrieves an existing tensor or registers a new one."""
        if name in self.tensors:
            return self.tensors[name]
        t = EIRTensor(name=name, shape=shape, dtype=dtype)
        return self.add_tensor(t)

    def add_node(self, node: EIRNode) -> EIRNode:
        """Appends an operation node to the graph and registers its tensors."""
        self.nodes.append(node)
        for t in node.inputs:
            if t.name not in self.tensors:
                self.tensors[t.name] = t
        for t in node.outputs:
            if t.name not in self.tensors:
                self.tensors[t.name] = t
        return node

    def add_input(self, tensor: EIRTensor) -> None:
        """Marks a tensor as a graph-level input."""
        if tensor.name not in self.tensors:
            self.add_tensor(tensor)
        if tensor not in self.inputs:
            self.inputs.append(tensor)

    def add_output(self, tensor: EIRTensor) -> None:
        """Marks a tensor as a graph-level output."""
        if tensor.name not in self.tensors:
            self.add_tensor(tensor)
        if tensor not in self.outputs:
            self.outputs.append(tensor)

    def topological_sort(self) -> List[EIRNode]:
        """Sorts nodes into valid execution order using Kahn's algorithm."""
        in_degree: Dict[str, int] = {node.name: 0 for node in self.nodes}
        node_map: Dict[str, EIRNode] = {node.name: node for node in self.nodes}

        # Initialize in-degree based on non-constant tensor dependencies
        for node in self.nodes:
            deg = 0
            for inp in node.inputs:
                if inp.producer is not None:
                    deg += 1
            in_degree[node.name] = deg

        # Queue nodes with in_degree == 0
        queue = deque([node_map[name] for name, deg in in_degree.items() if deg == 0])
        sorted_nodes: List[EIRNode] = []

        while queue:
            curr = queue.popleft()
            sorted_nodes.append(curr)

            for out_t in curr.outputs:
                for consumer in out_t.consumers:
                    if consumer.name in in_degree:
                        in_degree[consumer.name] -= 1
                        if in_degree[consumer.name] == 0:
                            queue.append(consumer)

        if len(sorted_nodes) != len(self.nodes):
            missing = set(node_map.keys()) - set(n.name for n in sorted_nodes)
            raise RuntimeError(f"Graph cycle detected or unresolved dependencies for nodes: {missing}")

        self.nodes = sorted_nodes
        return sorted_nodes

    def validate(self) -> None:
        """Validates graph connectivity and tensor shapes."""
        if not self.inputs:
            raise ValueError("Graph has no declared inputs.")
        if not self.outputs:
            raise ValueError("Graph has no declared outputs.")

        # Ensure all tensor shapes are 4D
        for t_name, t in self.tensors.items():
            if len(t.shape) != 4:
                raise ValueError(f"Tensor '{t_name}' shape {t.shape} is not 4D [N, C, H, W].")

        # Ensure topological order is valid
        self.topological_sort()

    def summary(self) -> str:
        """Generates a text summary of graph structure and operator statistics."""
        lines = [
            f"=== EIR Graph: {self.name} ===",
            f"Inputs:  {[f'{t.name} {list(t.shape)} ({t.dtype})' for t in self.inputs]}",
            f"Outputs: {[f'{t.name} {list(t.shape)} ({t.dtype})' for t in self.outputs]}",
            f"Total Nodes: {len(self.nodes)}, Total Tensors: {len(self.tensors)}",
            "Nodes:"
        ]
        for i, n in enumerate(self.nodes):
            lines.append(f"  [{i:02d}] {n.name:<30} {n.op_type.name:<15} Engine: {n.engine.name:<6} In: {[t.name for t in n.inputs]} Out: {[t.name for t in n.outputs]}")
        return "\n".join(lines)
