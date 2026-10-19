"""DSP Submodel & UMD Parameter Struct Builder for Cadence Vision Q7 Cores.

Formats binary execution descriptors, kernel parameter structures, and synchronization
events for operations offloaded to the DSP cores (RMSNorm, Softmax, RoPE).
"""

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from .flatbuffer_emitter import (
        AddressEntry,
        BlobEntry,
        EventEntry,
        LoadableModel,
        MemoryEntry,
        ModelVersion,
        SubModel,
        TensorDescEntry,
    )
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..ir.tensor import EIRTensor
except (ImportError, ValueError):
    from backend.flatbuffer_emitter import (
        AddressEntry,
        BlobEntry,
        EventEntry,
        LoadableModel,
        MemoryEntry,
        ModelVersion,
        SubModel,
        TensorDescEntry,
    )
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from ir.tensor import EIRTensor


# ==============================================================================
# DSP Parameter Structure Encoders
# ==============================================================================

def pack_rmsnorm_params(epsilon: float = 1e-5, hidden_size: int = 576) -> bytes:
    """Packs parameter block for `libes_onnx_rms_normalization_f16f16.so`.
    
    Structure:
      +0: float32  epsilon
      +4: uint32   hidden_size
      +8: uint32   padding[2]
    """
    return struct.pack("<fIII", float(epsilon), int(hidden_size), 0, 0)


def pack_softmax_params(axis: int = -1) -> bytes:
    """Packs parameter block for `libes_onnx_softmax.so`.
    
    Structure:
      +0: int32    axis
      +4: uint32   padding[3]
    """
    return struct.pack("<iIII", int(axis), 0, 0, 0)


def pack_rope_params(
    head_dim: int = 64,
    rotary_dim: int = 64,
    theta: float = 10000.0,
    max_position: int = 2048
) -> bytes:
    """Packs parameter block for `libes_qk_rotary_position_embedding_f16f16.so`.
    
    Structure:
      +0: uint32   head_dim
      +4: uint32   rotary_dim
      +8: float32  theta
      +12: uint32  max_position
    """
    return struct.pack("<IIfI", int(head_dim), int(rotary_dim), float(theta), int(max_position))


# ==============================================================================
# DSP Submodel Builder
# ==============================================================================

class DSPSubmodelBuilder:
    """Constructs UMD task descriptors, parameter blobs, and event barriers for DSP nodes."""

    @staticmethod
    def build_dsp_task(
        node: EIRNode,
        task_id: int = 0,
        input_address_ids: Optional[List[int]] = None,
        output_address_ids: Optional[List[int]] = None,
    ) -> Tuple[BlobEntry, BlobEntry, List[EventEntry]]:
        """Assembles parameter blob, surface descriptor, and sync events for a single DSP node.
        
        Returns:
            Tuple[param_blob, surf_blob, event_list]
        """
        if node.engine != EngineType.DSP:
            raise ValueError(f"Node '{node.name}' is not marked for DSP execution.")

        kernel_name = node.dsp_kernel_name or "libes_unknown.so"

        # 1. Encode parameter payload based on operator type
        if node.op_type == OpType.RMSNORM:
            eps = node.attributes.get("epsilon", 1e-5)
            h_size = node.inputs[0].shape[1] if node.inputs else 576
            param_bytes = pack_rmsnorm_params(epsilon=eps, hidden_size=h_size)
        elif node.op_type == OpType.SOFTMAX:
            ax = node.attributes.get("axis", -1)
            param_bytes = pack_softmax_params(axis=ax)
        elif node.op_type == OpType.ROPE:
            h_dim = node.attributes.get("head_dim", 64)
            r_dim = node.attributes.get("rotary_dim", 64)
            th = node.attributes.get("theta", 10000.0)
            param_bytes = pack_rope_params(head_dim=h_dim, rotary_dim=r_dim, theta=th)
        else:
            param_bytes = b"\x00" * 32

        param_blob = BlobEntry(
            name=f"dsp-params-{node.name}",
            size=len(param_bytes),
            interface=1,  # UMD interface
            sub_interface=5,
            data=param_bytes
        )

        # 2. Build UMD surface mapping descriptor (linking input/output buffer address IDs)
        in_addrs = input_address_ids or [0]
        out_addrs = output_address_ids or [1]

        # UMD surface packet: num_inputs, in_addrs..., num_outputs, out_addrs...
        surf_header = struct.pack("<HH", len(in_addrs), len(out_addrs))
        in_bytes = struct.pack(f"<{len(in_addrs)}H", *in_addrs)
        out_bytes = struct.pack(f"<{len(out_addrs)}H", *out_addrs)
        surf_raw = surf_header + in_bytes + out_bytes
        pad_len = ((len(surf_raw) + 15) & ~15) - len(surf_raw)
        aligned_surf = surf_raw + b"\x00" * pad_len

        surf_blob = BlobEntry(
            name=f"umd-surf-{node.name}",
            size=len(aligned_surf),
            interface=1,
            sub_interface=4,
            data=aligned_surf
        )

        # 3. Build synchronization events (wait upstream, signal downstream)
        events = [
            EventEntry(id=task_id * 2, type=0, target=task_id, val=1, op=0),      # Wait event
            EventEntry(id=task_id * 2 + 1, type=0, target=task_id, val=1, op=1),  # Signal event
        ]

        return param_blob, surf_blob, events
