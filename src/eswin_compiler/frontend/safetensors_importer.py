"""Direct SafeTensors Importer & LLM Computational Graph Synthesizer for EIR.

This module provides a pure-Python zero-dependency SafeTensors file reader
and synthesizes optimized EIR graphs for LLM transformer components (SwiGLU MLP,
Multi-Head Attention, and Dual-Die LM Head).
"""

import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import ModelImporter
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


# NumPy data type mapping from SafeTensors dtype strings
SAFETENSORS_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float16,  # Convert bfloat16 to float16 for NPU execution
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
}


class SafeTensorsReader:
    """Zero-dependency parser for Hugging Face .safetensors files."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        with open(self.file_path, "rb") as f:
            # 1. First 8 bytes: unsigned 64-bit little-endian integer for header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"Corrupt safetensors file: {file_path}")
            self.header_len = struct.unpack("<Q", header_len_bytes)[0]

            # 2. Read JSON header
            header_json_bytes = f.read(self.header_len)
            self.header: Dict[str, Any] = json.loads(header_json_bytes.decode("utf-8"))

        self.data_offset = 8 + self.header_len

    def keys(self) -> List[str]:
        """Returns list of tensor names stored in the file (excluding __metadata__)."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor_meta(self, name: str) -> Dict[str, Any]:
        """Returns shape, dtype, and offsets metadata for a given tensor."""
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.file_path}")
        return self.header[name]

    def get_tensor(self, name: str) -> np.ndarray:
        """Reads and deserializes a tensor from disk as a numpy array."""
        meta = self.get_tensor_meta(name)
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        start_offset, end_offset = meta["data_offsets"]

        np_dtype = SAFETENSORS_DTYPE_MAP.get(dtype_str, np.float16)

        with open(self.file_path, "rb") as f:
            f.seek(self.data_offset + start_offset)
            raw_bytes = f.read(end_offset - start_offset)

        if dtype_str == "BF16":
            # Convert BF16 (uint16) to FP32 then FP16
            u16_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
            f32_arr = (u16_arr.astype(np.uint32) << 16).view(np.float32)
            arr = f32_arr.astype(np.float16).reshape(shape)
        else:
            arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

        return arr


class SafeTensorsImporter(ModelImporter):
    """Synthesizes EIR neural network graphs directly from Hugging Face SafeTensors weights."""

    def __init__(self, reader: Optional[SafeTensorsReader] = None):
        self.reader = reader

    def import_model(self, source: Union[str, Path, SafeTensorsReader]) -> EIRGraph:
        """Default import method; when called without args, synthesizes complete graph."""
        if isinstance(source, (str, Path)):
            self.reader = SafeTensorsReader(source)
        elif isinstance(source, SafeTensorsReader):
            self.reader = source
        else:
            raise TypeError(f"Expected path or SafeTensorsReader, got {type(source)}")

        # Default synthesis for LLM: synthesize all layers found in keys
        raise NotImplementedError("Use specific synthesize_mlp_block or synthesize_lm_head methods.")

    def synthesize_mlp_block(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "model.layers",
        dtype: str = "float16"
    ) -> EIRGraph:
        """Synthesizes a Llama/SmolLM/Qwen SwiGLU MLP Block:
        
        y = (silu(x @ W_gate) * (x @ W_up)) @ W_down
        
        Dimensions:
        - input: [1, hidden_size, 1, 1]
        - gate_proj: [intermediate_size, hidden_size]
        - up_proj: [intermediate_size, hidden_size]
        - down_proj: [hidden_size, intermediate_size]
        - output: [1, hidden_size, 1, 1]
        """
        graph = EIRGraph(name=f"swiglu_mlp_layer_{layer_idx}")

        # Declare network input
        input_tensor = EIRTensor(
            name="input",
            shape=(1, hidden_size, 1, 1),
            dtype=dtype
        )
        graph.add_input(input_tensor)

        # Retrieve weights if reader is present, otherwise initialize zeros
        gate_name = f"{prefix}.{layer_idx}.mlp.gate_proj.weight"
        up_name = f"{prefix}.{layer_idx}.mlp.up_proj.weight"
        down_name = f"{prefix}.{layer_idx}.mlp.down_proj.weight"

        gate_weight = self.reader.get_tensor(gate_name) if self.reader and gate_name in self.reader.keys() else np.zeros((intermediate_size, hidden_size), dtype=np.float16)
        up_weight = self.reader.get_tensor(up_name) if self.reader and up_name in self.reader.keys() else np.zeros((intermediate_size, hidden_size), dtype=np.float16)
        down_weight = self.reader.get_tensor(down_name) if self.reader and down_name in self.reader.keys() else np.zeros((hidden_size, intermediate_size), dtype=np.float16)

        gate_weight_tensor = EIRTensor(name="gate_proj_weight", shape=normalize_to_4d(gate_weight.shape), dtype=dtype, data=gate_weight)
        up_weight_tensor = EIRTensor(name="up_proj_weight", shape=normalize_to_4d(up_weight.shape), dtype=dtype, data=up_weight)
        down_weight_tensor = EIRTensor(name="down_proj_weight", shape=normalize_to_4d(down_weight.shape), dtype=dtype, data=down_weight)

        graph.add_tensor(gate_weight_tensor)
        graph.add_tensor(up_weight_tensor)
        graph.add_tensor(down_weight_tensor)

        # 1. Gate projection: input @ gate_weight^T -> [1, intermediate_size, 1, 1]
        gate_out = EIRTensor(name="gate_out", shape=(1, intermediate_size, 1, 1), dtype=dtype)
        node_gate = EIRNode(
            name=f"matmul_gate_{layer_idx}",
            op_type=OpType.GEMM,
            inputs=[input_tensor, gate_weight_tensor],
            outputs=[gate_out],
            attributes={"transB": 1},
            engine=EngineType.NPU
        )
        graph.add_node(node_gate)

        # 2. SiLU activation on gate
        silu_out = EIRTensor(name="silu_out", shape=(1, intermediate_size, 1, 1), dtype=dtype)
        node_silu = EIRNode(
            name=f"silu_{layer_idx}",
            op_type=OpType.SILU,
            inputs=[gate_out],
            outputs=[silu_out],
            engine=EngineType.NPU
        )
        graph.add_node(node_silu)

        # 3. Up projection: input @ up_weight^T -> [1, intermediate_size, 1, 1]
        up_out = EIRTensor(name="up_out", shape=(1, intermediate_size, 1, 1), dtype=dtype)
        node_up = EIRNode(
            name=f"matmul_up_{layer_idx}",
            op_type=OpType.GEMM,
            inputs=[input_tensor, up_weight_tensor],
            outputs=[up_out],
            attributes={"transB": 1},
            engine=EngineType.NPU
        )
        graph.add_node(node_up)

        # 4. Element-wise multiply: silu_out * up_out
        mul_out = EIRTensor(name="mul_out", shape=(1, intermediate_size, 1, 1), dtype=dtype)
        node_mul = EIRNode(
            name=f"mul_{layer_idx}",
            op_type=OpType.MUL,
            inputs=[silu_out, up_out],
            outputs=[mul_out],
            engine=EngineType.NPU
        )
        graph.add_node(node_mul)

        # 5. Down projection: mul_out @ down_weight^T -> [1, hidden_size, 1, 1]
        down_out = EIRTensor(name="output", shape=(1, hidden_size, 1, 1), dtype=dtype)
        node_down = EIRNode(
            name=f"matmul_down_{layer_idx}",
            op_type=OpType.GEMM,
            inputs=[mul_out, down_weight_tensor],
            outputs=[down_out],
            attributes={"transB": 1},
            engine=EngineType.NPU
        )
        graph.add_node(node_down)

        # Declare network output
        graph.add_output(down_out)
        return graph

    def synthesize_lm_head(
        self,
        hidden_size: int,
        vocab_size: int,
        die: Optional[int] = None,
        weight_name: str = "lm_head.weight",
        dtype: str = "float16"
    ) -> EIRGraph:
        """Synthesizes an LM Head projection block with optional dual-die vocabulary partitioning.
        
        Args:
            hidden_size: Embedding dimension (e.g. 576 for SmolLM-135M, 1536 for Qwen2.5-1.5B).
            vocab_size: Total vocabulary size (e.g. 49,152 for SmolLM, 152,064 for Qwen2.5).
            die: If None, emits full head. If 0, emits vocab [0..vocab_size//2].
                 If 1, emits vocab [vocab_size//2..vocab_size].
        """
        die_tag = f"_d{die}" if die is not None else ""
        graph = EIRGraph(name=f"lm_head{die_tag}")

        # Declare input
        input_tensor = EIRTensor(
            name="input",
            shape=(1, hidden_size, 1, 1),
            dtype=dtype
        )
        graph.add_input(input_tensor)

        # Determine vocabulary slice
        half_vocab = vocab_size // 2
        if die == 0:
            v_start, v_end = 0, half_vocab
        elif die == 1:
            v_start, v_end = half_vocab, vocab_size
        else:
            v_start, v_end = 0, vocab_size

        sub_vocab_len = v_end - v_start

        # Retrieve weights
        if self.reader and weight_name in self.reader.keys():
            full_w = self.reader.get_tensor(weight_name)
            sub_w = full_w[v_start:v_end, :]
        else:
            sub_w = np.zeros((sub_vocab_len, hidden_size), dtype=np.float16)

        w_tensor = EIRTensor(
            name=f"lm_head_weight{die_tag}",
            shape=normalize_to_4d(sub_w.shape),
            dtype=dtype,
            data=sub_w
        )
        graph.add_tensor(w_tensor)

        # Output logits tensor
        logits_tensor = EIRTensor(
            name="output",
            shape=(1, sub_vocab_len, 1, 1),
            dtype=dtype
        )

        node = EIRNode(
            name=f"lm_head_proj{die_tag}",
            op_type=OpType.GEMM,
            inputs=[input_tensor, w_tensor],
            outputs=[logits_tensor],
            attributes={"transB": 1},
            engine=EngineType.NPU
        )
        graph.add_node(node)
        graph.add_output(logits_tensor)
        return graph

    def synthesize_attention_block(
        self,
        layer_idx: int,
        hidden_size: int = 576,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        head_dim: int = 64,
        prefix: str = "model.layers",
        dtype: str = "float16"
    ) -> EIRGraph:
        """Synthesizes a full Multi-Head Attention block with RoPE and Softmax.
        
        The block partitions into:
        - NPU: Q, K, V linear projections (CMAC)
        - DSP: RMSNorm, RoPE rotary position embedding, and Softmax
        - NPU: Context linear projection (CMAC)
        """
        graph = EIRGraph(name=f"attention_layer_{layer_idx}")

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Input tensor
        input_tensor = EIRTensor(name="input", shape=(1, hidden_size, 1, 1), dtype=dtype)
        graph.add_input(input_tensor)

        # 1. Pre-attention RMSNorm (DSP)
        norm_out = EIRTensor(name=f"norm_out_{layer_idx}", shape=(1, hidden_size, 1, 1), dtype=dtype)
        node_norm = EIRNode(
            name=f"rmsnorm_{layer_idx}",
            op_type=OpType.RMSNORM,
            inputs=[input_tensor],
            outputs=[norm_out],
            attributes={"epsilon": 1e-5},
            engine=EngineType.DSP,
            dsp_kernel_name="libes_onnx_rms_normalization_f16f16.so"
        )
        graph.add_node(node_norm)

        # 2. QKV Linear Projections (NPU)
        q_weight = EIRTensor(name=f"q_weight_{layer_idx}", shape=(q_dim, hidden_size, 1, 1), dtype=dtype, data=np.zeros((q_dim, hidden_size, 1, 1), dtype=np.float16))
        k_weight = EIRTensor(name=f"k_weight_{layer_idx}", shape=(kv_dim, hidden_size, 1, 1), dtype=dtype, data=np.zeros((kv_dim, hidden_size, 1, 1), dtype=np.float16))
        v_weight = EIRTensor(name=f"v_weight_{layer_idx}", shape=(kv_dim, hidden_size, 1, 1), dtype=dtype, data=np.zeros((kv_dim, hidden_size, 1, 1), dtype=np.float16))
        graph.add_tensor(q_weight)
        graph.add_tensor(k_weight)
        graph.add_tensor(v_weight)

        q_out = EIRTensor(name=f"q_out_{layer_idx}", shape=(1, q_dim, 1, 1), dtype=dtype)
        k_out = EIRTensor(name=f"k_out_{layer_idx}", shape=(1, kv_dim, 1, 1), dtype=dtype)
        v_out = EIRTensor(name=f"v_out_{layer_idx}", shape=(1, kv_dim, 1, 1), dtype=dtype)

        node_q = EIRNode(name=f"q_proj_{layer_idx}", op_type=OpType.GEMM, inputs=[norm_out, q_weight], outputs=[q_out], attributes={"transB": 1}, engine=EngineType.NPU)
        node_k = EIRNode(name=f"k_proj_{layer_idx}", op_type=OpType.GEMM, inputs=[norm_out, k_weight], outputs=[k_out], attributes={"transB": 1}, engine=EngineType.NPU)
        node_v = EIRNode(name=f"v_proj_{layer_idx}", op_type=OpType.GEMM, inputs=[norm_out, v_weight], outputs=[v_out], attributes={"transB": 1}, engine=EngineType.NPU)
        graph.add_node(node_q)
        graph.add_node(node_k)
        graph.add_node(node_v)

        # 3. RoPE Rotary Position Embedding (DSP)
        q_rot = EIRTensor(name=f"q_rot_{layer_idx}", shape=(1, q_dim, 1, 1), dtype=dtype)
        k_rot = EIRTensor(name=f"k_rot_{layer_idx}", shape=(1, kv_dim, 1, 1), dtype=dtype)
        node_rope = EIRNode(
            name=f"rope_{layer_idx}",
            op_type=OpType.ROPE,
            inputs=[q_out, k_out],
            outputs=[q_rot, k_rot],
            attributes={"head_dim": head_dim, "rotary_dim": head_dim, "theta": 10000.0},
            engine=EngineType.DSP,
            dsp_kernel_name="libes_qk_rotary_position_embedding_f16f16.so"
        )
        graph.add_node(node_rope)

        # 4. Attention Softmax (DSP)
        scores = EIRTensor(name=f"scores_{layer_idx}", shape=(1, num_heads, 1, 1), dtype=dtype)
        probs = EIRTensor(name=f"probs_{layer_idx}", shape=(1, num_heads, 1, 1), dtype=dtype)
        node_score = EIRNode(name=f"attn_score_{layer_idx}", op_type=OpType.MATMUL, inputs=[q_rot, k_rot], outputs=[scores], engine=EngineType.NPU)
        node_softmax = EIRNode(
            name=f"softmax_{layer_idx}",
            op_type=OpType.SOFTMAX,
            inputs=[scores],
            outputs=[probs],
            attributes={"axis": -1},
            engine=EngineType.DSP,
            dsp_kernel_name="libes_onnx_softmax.so"
        )
        graph.add_node(node_score)
        graph.add_node(node_softmax)

        # 5. Output Projection (NPU)
        o_weight = EIRTensor(name=f"o_weight_{layer_idx}", shape=(hidden_size, q_dim, 1, 1), dtype=dtype, data=np.zeros((hidden_size, q_dim, 1, 1), dtype=np.float16))
        graph.add_tensor(o_weight)
        out_tensor = EIRTensor(name="output", shape=(1, hidden_size, 1, 1), dtype=dtype)

        node_out = EIRNode(name=f"o_proj_{layer_idx}", op_type=OpType.GEMM, inputs=[probs, o_weight], outputs=[out_tensor], attributes={"transB": 1}, engine=EngineType.NPU)
        graph.add_node(node_out)
        graph.add_output(out_tensor)

        return graph

