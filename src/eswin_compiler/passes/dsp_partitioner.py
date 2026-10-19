"""Graph partitioning pass assigning non-NPU vector operations to Cadence Vision Q7 DSP cores."""

from typing import Dict, Optional, Tuple
from .base import GraphPass
try:
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType


# Mapping from EIR OpType and dtype to canonical DSP shared library kernel names
DSP_KERNEL_CATALOG: Dict[Tuple[OpType, str], str] = {
    (OpType.RMSNORM, "float16"): "libes_onnx_rms_normalization_f16f16.so",
    (OpType.RMSNORM, "int8"): "libes_onnx_rms_normalization_f16s8.so",
    (OpType.LAYERNORM, "float16"): "libes_onnx_layer_normalization_f16f16.so",
    (OpType.SOFTMAX, "float16"): "libes_onnx_softmax.so",
    (OpType.SOFTMAX, "float32"): "libes_onnx_softmax.so",
    (OpType.ROPE, "float16"): "libes_qk_rotary_position_embedding_f16f16.so",
    (OpType.ROPE, "int8"): "libes_qk_rotary_position_embedding_s8f16.so",
}


class DSPPartitionerPass(GraphPass):
    """Partitions the computational DAG between NPU hardware engines and DSP vector cores.
    
    Operations that cannot run on NVDLA fixed-function engines (RMSNorm, LayerNorm,
    Softmax, RoPE) are assigned to `EngineType.DSP` and tagged with the exact vendor
    DSP kernel shared object (.so).
    """

    def run(self, graph: EIRGraph) -> bool:
        modified = False

        for node in graph.nodes:
            kernel_name = self._resolve_dsp_kernel(node)
            if kernel_name is not None:
                if node.engine != EngineType.DSP or node.dsp_kernel_name != kernel_name:
                    node.engine = EngineType.DSP
                    node.dsp_kernel_name = kernel_name
                    modified = True

        return modified

    def _resolve_dsp_kernel(self, node: EIRNode) -> Optional[str]:
        """Resolves target DSP kernel library based on operator type and input precision."""
        if not node.inputs:
            return None

        primary_input = node.inputs[0]
        dtype = primary_input.dtype

        # Exact match
        key = (node.op_type, dtype)
        if key in DSP_KERNEL_CATALOG:
            return DSP_KERNEL_CATALOG[key]

        # Generic match on OpType with default float16
        fallback_key = (node.op_type, "float16")
        if fallback_key in DSP_KERNEL_CATALOG:
            return DSP_KERNEL_CATALOG[fallback_key]

        return None
