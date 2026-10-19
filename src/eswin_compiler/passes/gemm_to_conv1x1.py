"""Optimization pass lowering GEMM and MatMul operations into 1x1 Convolutions.

The EIC7700X CMAC (Convolution Multiply-Accumulate) hardware engine natively executes
2D convolutions with high MAC throughput. Fully connected linear projections, LLM
attention heads, and MLP feed-forward layers are lowered into 1x1 pointwise convolutions:
  Conv2D(input=[N, C_in, 1, 1], weight=[C_out, C_in, 1, 1], kernel=[1, 1], stride=[1, 1])
"""

import numpy as np
from typing import List

from .base import GraphPass
try:
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..ir.tensor import EIRTensor, normalize_to_4d
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from ir.tensor import EIRTensor, normalize_to_4d


class GemmToConv1x1Pass(GraphPass):
    """Transforms GEMM and MatMul nodes into 1x1 Convolution nodes for NPU CMAC execution."""

    def run(self, graph: EIRGraph) -> bool:
        modified = False

        for node in list(graph.nodes):
            if node.op_type in (OpType.GEMM, OpType.MATMUL):
                self._lower_to_conv1x1(graph, node)
                modified = True

        return modified

    def _lower_to_conv1x1(self, graph: EIRGraph, node: EIRNode) -> None:
        """Lowers a single GEMM/MatMul node into a 1x1 Conv."""
        inputs = node.inputs
        if len(inputs) < 2:
            return

        act_tensor = inputs[0]
        weight_tensor = inputs[1]
        bias_tensor = inputs[2] if len(inputs) > 2 else None

        trans_b = bool(node.attributes.get("transB", 0))

        # Ensure weight data is transposed to (C_out, C_in) if needed
        if weight_tensor.data is not None:
            w_data = weight_tensor.data
            # If weight is 2D
            if w_data.ndim == 2:
                if not trans_b and node.op_type == OpType.MATMUL:
                    # In standard MatMul: input [1, C_in], weight [C_in, C_out] -> transpose to [C_out, C_in]
                    w_data = np.ascontiguousarray(w_data.T)
                elif trans_b:
                    # Already [C_out, C_in]
                    w_data = np.ascontiguousarray(w_data)
            elif w_data.ndim == 4:
                # If already 4D, squeeze spatial dims to check
                if w_data.shape[2] == 1 and w_data.shape[3] == 1:
                    w_data_2d = w_data.reshape(w_data.shape[0], w_data.shape[1])
                    if not trans_b and node.op_type == OpType.MATMUL:
                        w_data = np.ascontiguousarray(w_data_2d.T)
                    else:
                        w_data = np.ascontiguousarray(w_data_2d)

            c_out = w_data.shape[0]
            c_in = w_data.shape[1]

            # Reshape weight array to 4D [C_out, C_in, 1, 1]
            w_data_4d = w_data.reshape((c_out, c_in, 1, 1))
            weight_tensor.data = w_data_4d
            weight_tensor.shape = (c_out, c_in, 1, 1)

        # Mutate node to CONV
        node.op_type = OpType.CONV
        node.engine = EngineType.NPU
        node.attributes = {
            "kernel_shape": [1, 1],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "dilations": [1, 1],
            "group": 1,
        }

        # Update output tensor shape if needed
        for out_t in node.outputs:
            if weight_tensor.data is not None:
                c_out = weight_tensor.data.shape[0]
                n_batch = act_tensor.shape[0]
                out_t.shape = (n_batch, c_out, 1, 1)
