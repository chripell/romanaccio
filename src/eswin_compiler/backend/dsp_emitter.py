"""DSP Sub-Model and Kernel Descriptor Emitter for ESWIN EIC7700X.

Emits DSP task descriptors and operator configurations matching the Cadence Vision Q7
hardware execution runtime (libes_dsp_runtime_ll.so) and firmware kernel specifications
(/usr/lib/firmware/dsp_kernels/*.pkg.lib) for non-GEMM transformer and vision ops:
  - RMSNorm (es_rms_norm_f16f16.pkg.lib)
  - RoPE (es_qk_rotary_position_embedding_f16f16.pkg.lib)
  - Softmax (es_onnx_softmax.pkg.lib)
  - LayerNorm (es_onnx_layer_normalization_f16f16.pkg.lib)
  - YOLOv11 Post-processing & NMS (es_detection_out_f16f16_yolov11.pkg.lib)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union


class DSPKernelType(Enum):
    RMS_NORM = "rms_norm_f16f16"
    SOFTMAX = "onnx_softmax"
    ROPE = "qk_rotary_position_embedding_f16f16"
    LAYER_NORM = "onnx_layer_normalization_f16f16"
    YOLO_DETECTION = "detection_out_f16f16_yolov11"
    TOPK = "onnx_topk_f16f16"


@dataclass
class DSPOperatorDesc:
    """Hardware descriptor for a Cadence Vision Q7 DSP execution task."""
    op_name: str
    kernel_lib_name: str
    buffer_cnt_cfg: int
    buffer_cnt_input: int
    buffer_cnt_output: int
    cfg_data: bytes
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    dtype: str = "float16"

    @property
    def total_desc_size(self) -> int:
        """Size of ES_DSP_OPERATOR_DESC_S + paramData."""
        # sizeof(ES_DSP_OPERATOR_DESC_S) = 4 + 128 + 4*3 + 64*4 = 400 bytes
        return 400 + len(self.cfg_data)

    def pack_operator_desc(self) -> bytes:
        """Serializes into binary C struct ES_DSP_OPERATOR_DESC_S."""
        name_bytes = self.op_name.encode("utf-8")[:127]
        name_padded = name_bytes.ljust(128, b"\x00")

        buf_sizes = [0] * 64
        # Offset 0..cfg
        buf_sizes[0] = len(self.cfg_data)
        # Inputs
        elem_bytes = 2 if self.dtype == "float16" else 4
        for i, shape in enumerate(self.input_shapes):
            size = 1
            for d in shape:
                size *= d
            buf_sizes[self.buffer_cnt_cfg + i] = size * elem_bytes
        # Outputs
        for i, shape in enumerate(self.output_shapes):
            size = 1
            for d in shape:
                size *= d
            buf_sizes[self.buffer_cnt_cfg + self.buffer_cnt_input + i] = size * elem_bytes

        header = struct.pack(
            f"<I128s3I64I",
            self.total_desc_size,
            name_padded,
            self.buffer_cnt_cfg,
            self.buffer_cnt_input,
            self.buffer_cnt_output,
            *buf_sizes
        )
        return header + self.cfg_data


class DSPEmitter:
    """Emits DSP task descriptors and multi-engine coordination manifests."""

    @staticmethod
    def emit_rms_norm(
        channels: int,
        batch_size: int = 1,
        epsilon: float = 1e-5,
        dtype: str = "float16"
    ) -> DSPOperatorDesc:
        """Emits descriptor for RMSNorm (Llama/Qwen/SmolLM normalization)."""
        # Pack config struct: uint32 channels, float32 epsilon
        cfg_data = struct.pack("<If", channels, epsilon)
        return DSPOperatorDesc(
            op_name="rms_norm_f16f16",
            kernel_lib_name="es_rms_norm_f16f16.pkg.lib",
            buffer_cnt_cfg=1,
            buffer_cnt_input=2,  # input tensor + scale (gamma) weights
            buffer_cnt_output=1,
            cfg_data=cfg_data,
            input_shapes=[(batch_size, channels), (channels,)],
            output_shapes=[(batch_size, channels)],
            dtype=dtype
        )

    @staticmethod
    def emit_softmax(
        channels: int,
        batch_size: int = 1,
        axis: int = -1,
        dtype: str = "float16"
    ) -> DSPOperatorDesc:
        """Emits descriptor for Softmax attention probabilities."""
        cfg_data = struct.pack("<ii", axis, channels)
        return DSPOperatorDesc(
            op_name="onnx_softmax",
            kernel_lib_name="es_onnx_softmax.pkg.lib",
            buffer_cnt_cfg=1,
            buffer_cnt_input=1,
            buffer_cnt_output=1,
            cfg_data=cfg_data,
            input_shapes=[(batch_size, channels)],
            output_shapes=[(batch_size, channels)],
            dtype=dtype
        )

    @staticmethod
    def emit_rope(
        seq_len: int,
        head_dim: int,
        num_heads: int,
        base: float = 10000.0,
        dtype: str = "float16"
    ) -> DSPOperatorDesc:
        """Emits descriptor for Rotary Position Embeddings (RoPE)."""
        cfg_data = struct.pack("<IIIf", seq_len, head_dim, num_heads, base)
        return DSPOperatorDesc(
            op_name="qk_rotary_position_embedding_f16f16",
            kernel_lib_name="es_qk_rotary_position_embedding_f16f16.pkg.lib",
            buffer_cnt_cfg=1,
            buffer_cnt_input=2,  # Q and K tensors
            buffer_cnt_output=2, # Rotated Q and K tensors
            cfg_data=cfg_data,
            input_shapes=[(1, num_heads, seq_len, head_dim), (1, num_heads, seq_len, head_dim)],
            output_shapes=[(1, num_heads, seq_len, head_dim), (1, num_heads, seq_len, head_dim)],
            dtype=dtype
        )

    @staticmethod
    def emit_yolov11_detection(
        num_classes: int = 80,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_boxes: int = 300,
        dtype: str = "float16"
    ) -> DSPOperatorDesc:
        """Emits descriptor for YOLOv11 Anchor-Free BBox decoding and NMS."""
        cfg_data = struct.pack("<IffI", num_classes, conf_threshold, iou_threshold, max_boxes)
        return DSPOperatorDesc(
            op_name="detection_out_f16f16_yolov11",
            kernel_lib_name="es_detection_out_f16f16_yolov11.pkg.lib",
            buffer_cnt_cfg=1,
            buffer_cnt_input=1,  # Raw feature pyramid output
            buffer_cnt_output=2, # Detection boxes + scores
            cfg_data=cfg_data,
            input_shapes=[(1, 84, 8400)],
            output_shapes=[(1, max_boxes, 4), (1, max_boxes, num_classes)],
            dtype=dtype
        )

    @staticmethod
    def emit_manifest_entry(op: DSPOperatorDesc) -> Dict[str, Any]:
        """Creates a manifest JSON entry for hybrid NPU+DSP runtime coordination."""
        return {
            "engine": "DSP",
            "op_name": op.op_name,
            "kernel_lib": op.kernel_lib_name,
            "desc_size": op.total_desc_size,
            "cfg_bytes": len(op.cfg_data),
            "inputs": [{"shape": list(s), "dtype": op.dtype} for s in op.input_shapes],
            "outputs": [{"shape": list(s), "dtype": op.dtype} for s in op.output_shapes],
        }
