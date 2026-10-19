"""Dual-Die Tensor Parallelism and Pipeline Partitioning Pass.

The FML13V03 / EIC7700X SoC features two identical NPU dies:
  - Die 0: NPU 0 (/dev/npu0), 4MB on-chip SPRAM, 1024 FP16 / 2048 INT8 MAC units
  - Die 1: NPU 1 (/dev/npu1), 4MB on-chip SPRAM, 1024 FP16 / 2048 INT8 MAC units

This pass handles dual-die parallelization:
1. Vocabulary Tensor Parallelism:
   Splits wide projection layers (e.g. Language Model Head with vocabulary size
   49,152 for SmolLM or 152,064 for Qwen2.5) along the channel/vocabulary dimension.
   - Die 0 computes logits [0 .. V/2 - 1]
   - Die 1 computes logits [V/2 .. V - 1]
   Both dies execute concurrently over PCIe/die interconnect, doubling effective
   memory bandwidth and compute throughput.

2. Pipeline Parallelism:
   Partitions sequential Transformer layers across dies:
   - Die 0: layers [0 .. N/2 - 1]
   - Die 1: layers [N/2 .. N - 1]
"""

from copy import deepcopy
from typing import List, Optional, Tuple
import numpy as np

from .base import GraphPass
try:
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..ir.tensor import EIRTensor
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from ir.tensor import EIRTensor


class DualDieSplitterPass(GraphPass):
    """Splits wide linear / convolution layers across Die 0 and Die 1."""

    def __init__(self, vocab_split_threshold: int = 2048):
        """
        Args:
            vocab_split_threshold: Minimum output channels to trigger dual-die
                                   vocabulary tensor parallelism.
        """
        self.vocab_split_threshold = vocab_split_threshold

    def run(self, graph: EIRGraph) -> bool:
        """Standard pass interface - annotates nodes suitable for dual-die splitting."""
        modified = False
        for node in graph.nodes:
            if node.op_type in (OpType.GEMM, OpType.CONV, OpType.MATMUL):
                if len(node.inputs) >= 2:
                    weight = node.inputs[1]
                    c_out = weight.shape[0] if len(weight.shape) >= 2 else 0
                    if c_out >= self.vocab_split_threshold:
                        node.attributes["dual_die_splittable"] = True
                        modified = True
        return modified

    def split_graph_tensor_parallel(
        self, graph: EIRGraph, target_node_name: Optional[str] = None
    ) -> Tuple[EIRGraph, EIRGraph]:
        """Splits an EIRGraph containing a wide projection into Die 0 and Die 1 subgraphs.
        
        Args:
            graph: The input computation graph.
            target_node_name: Optional specific node to split. If None, splits the first
                              node exceeding vocab_split_threshold.
                              
        Returns:
            Tuple of (graph_die0, graph_die1).
        """
        # Find candidate node
        target_node: Optional[EIRNode] = None
        for node in graph.nodes:
            if target_node_name is not None:
                if node.name == target_node_name:
                    target_node = node
                    break
            else:
                if len(node.inputs) >= 2:
                    w = node.inputs[1]
                    c_out = w.shape[0] if len(w.shape) >= 2 else 0
                    if c_out >= self.vocab_split_threshold:
                        target_node = node
                        break

        if target_node is None:
            raise ValueError(
                f"No suitable node found for dual-die splitting in graph '{graph.name}'."
            )

        weight_tensor = target_node.inputs[1]
        bias_tensor = target_node.inputs[2] if len(target_node.inputs) > 2 else None
        c_out = weight_tensor.shape[0]
        half_cout = c_out // 2

        # ----------------------------------------------------------------------
        # Construct Die 0 Graph
        # ----------------------------------------------------------------------
        graph_d0 = deepcopy(graph)
        graph_d0.name = f"{graph.name}_d0"
        node_d0 = [n for n in graph_d0.nodes if n.name == target_node.name][0]
        w_d0 = node_d0.inputs[1]

        # Slice weight for Die 0
        w_d0.shape = (half_cout, *weight_tensor.shape[1:])
        if w_d0.data is not None:
            w_d0.data = np.ascontiguousarray(w_d0.data[:half_cout])

        if bias_tensor is not None:
            b_d0 = node_d0.inputs[2]
            b_d0.shape = (half_cout, *bias_tensor.shape[1:])
            if b_d0.data is not None:
                b_d0.data = np.ascontiguousarray(b_d0.data[:half_cout])

        # Update output tensor shape for Die 0
        out_d0 = node_d0.outputs[0]
        out_d0.shape = (out_d0.shape[0], half_cout, *out_d0.shape[2:])

        # ----------------------------------------------------------------------
        # Construct Die 1 Graph
        # ----------------------------------------------------------------------
        graph_d1 = deepcopy(graph)
        graph_d1.name = f"{graph.name}_d1"
        node_d1 = [n for n in graph_d1.nodes if n.name == target_node.name][0]
        w_d1 = node_d1.inputs[1]

        # Slice weight for Die 1
        w_d1.shape = (c_out - half_cout, *weight_tensor.shape[1:])
        if w_d1.data is not None:
            w_d1.data = np.ascontiguousarray(w_d1.data[half_cout:])

        if bias_tensor is not None:
            b_d1 = node_d1.inputs[2]
            b_d1.shape = (c_out - half_cout, *bias_tensor.shape[1:])
            if b_d1.data is not None:
                b_d1.data = np.ascontiguousarray(b_d1.data[half_cout:])

        # Update output tensor shape for Die 1
        out_d1 = node_d1.outputs[0]
        out_d1.shape = (out_d1.shape[0], c_out - half_cout, *out_d1.shape[2:])

        return graph_d0, graph_d1

    @staticmethod
    def partition_pipeline_layers(layers: List[EIRGraph]) -> Tuple[List[EIRGraph], List[EIRGraph]]:
        """Partitions a sequential pipeline of transformer layer graphs across Die 0 and Die 1.
        
        Args:
            layers: List of per-layer EIR graphs [L0, L1, ..., LN-1].
            
        Returns:
            Tuple of (die0_layers, die1_layers).
        """
        n_layers = len(layers)
        half = n_layers // 2
        d0_layers = deepcopy(layers[:half])
        d1_layers = deepcopy(layers[half:])

        for i, l in enumerate(d0_layers):
            l.name = f"{l.name}_d0"
        for i, l in enumerate(d1_layers):
            l.name = f"{l.name}_d1"

        return d0_layers, d1_layers
