"""Unit tests for Attention Block synthesis, DSP partitioning, and parameter packing."""

from pathlib import Path
import struct
import sys
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.dsp_task_builder import (
    DSPSubmodelBuilder,
    pack_rmsnorm_params,
    pack_rope_params,
    pack_softmax_params,
)
from frontend.safetensors_importer import SafeTensorsImporter
from ir.graph import EIRGraph
from ir.node import EIRNode
from ir.op_types import EngineType, OpType
from ir.tensor import EIRTensor
from passes.dsp_partitioner import DSPPartitionerPass
from passes.gemm_to_conv1x1 import GemmToConv1x1Pass


class TestAttentionBlockAndDSP(unittest.TestCase):
    """Test suite for full Attention Block compilation and DSP offload."""

    def test_dsp_param_packers(self):
        """Test binary packing of DSP kernel parameter structs."""
        # 1. RMSNorm params (16 bytes: float32 eps, uint32 hidden_size, 2x uint32 pad)
        p_rms = pack_rmsnorm_params(epsilon=1e-5, hidden_size=576)
        self.assertEqual(len(p_rms), 16)
        eps, h_size, pad0, pad1 = struct.unpack("<fIII", p_rms)
        self.assertAlmostEqual(eps, 1e-5, places=6)
        self.assertEqual(h_size, 576)
        self.assertEqual(pad0, 0)
        self.assertEqual(pad1, 0)

        # 2. Softmax params (16 bytes: int32 axis, 3x uint32 pad)
        p_sm = pack_softmax_params(axis=-1)
        self.assertEqual(len(p_sm), 16)
        axis, pad0, pad1, pad2 = struct.unpack("<iIII", p_sm)
        self.assertEqual(axis, -1)
        self.assertEqual(pad0, 0)

        # 3. RoPE params (16 bytes: uint32 head_dim, uint32 rotary_dim, float32 theta, uint32 max_pos)
        p_rope = pack_rope_params(head_dim=64, rotary_dim=64, theta=10000.0, max_position=2048)
        self.assertEqual(len(p_rope), 16)
        h_dim, r_dim, theta, max_pos = struct.unpack("<IIfI", p_rope)
        self.assertEqual(h_dim, 64)
        self.assertEqual(r_dim, 64)
        self.assertAlmostEqual(theta, 10000.0, places=2)
        self.assertEqual(max_pos, 2048)

    def test_dsp_submodel_builder(self):
        """Test building UMD DSP task blobs and synchronization events."""
        x = EIRTensor(name="rms_in", shape=(1, 576, 1, 1), dtype="float16")
        y = EIRTensor(name="rms_out", shape=(1, 576, 1, 1), dtype="float16")
        node_rms = EIRNode(
            name="rmsnorm_0",
            op_type=OpType.RMSNORM,
            inputs=[x],
            outputs=[y],
            attributes={"epsilon": 1e-5},
            engine=EngineType.DSP,
            dsp_kernel_name="libes_onnx_rms_normalization_f16f16.so"
        )

        param_blob, surf_blob, events = DSPSubmodelBuilder.build_dsp_task(
            node=node_rms,
            task_id=2,
            input_address_ids=[10],
            output_address_ids=[11]
        )

        # Verify param blob
        self.assertEqual(param_blob.name, "dsp-params-rmsnorm_0")
        self.assertEqual(param_blob.interface, 1)
        self.assertEqual(param_blob.sub_interface, 5)
        self.assertEqual(param_blob.size, 16)

        # Verify surface blob
        self.assertEqual(surf_blob.name, "umd-surf-rmsnorm_0")
        self.assertEqual(surf_blob.interface, 1)
        self.assertEqual(surf_blob.sub_interface, 4)
        self.assertEqual(surf_blob.size % 16, 0)  # 16-byte aligned

        # Verify sync events (wait + signal pair)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].target, 2)
        self.assertEqual(events[0].op, 0)  # wait
        self.assertEqual(events[1].target, 2)
        self.assertEqual(events[1].op, 1)  # signal

    def test_attention_block_synthesis_and_partitioning(self):
        """Test end-to-end synthesis of SmolLM-135M Attention block and DSP partitioning."""
        importer = SafeTensorsImporter()
        graph = importer.synthesize_attention_block(
            layer_idx=0,
            hidden_size=576,
            num_heads=9,
            num_kv_heads=3,
            head_dim=64
        )
        graph.validate()

        # Check total nodes created
        # Expected:
        # 1. rmsnorm_0 (DSP)
        # 2. q_proj_0 (GEMM NPU)
        # 3. k_proj_0 (GEMM NPU)
        # 4. v_proj_0 (GEMM NPU)
        # 5. rope_0 (ROPE DSP)
        # 6. attn_score_0 (MATMUL NPU)
        # 7. softmax_0 (SOFTMAX DSP)
        # 8. o_proj_0 (GEMM NPU)
        self.assertEqual(len(graph.nodes), 8)

        # Run DSP Partitioner
        dsp_pass = DSPPartitionerPass()
        dsp_pass.run(graph)

        dsp_nodes = [n for n in graph.nodes if n.engine == EngineType.DSP]
        npu_nodes = [n for n in graph.nodes if n.engine == EngineType.NPU]

        self.assertEqual(len(dsp_nodes), 3)  # RMSNorm, RoPE, Softmax
        self.assertEqual(len(npu_nodes), 5)  # Q, K, V, Score, Out

        # Verify exact DSP kernel library assignments
        dsp_map = {n.op_type: n.dsp_kernel_name for n in dsp_nodes}
        self.assertEqual(dsp_map[OpType.RMSNORM], "libes_onnx_rms_normalization_f16f16.so")
        self.assertEqual(dsp_map[OpType.ROPE], "libes_qk_rotary_position_embedding_f16f16.so")
        self.assertEqual(dsp_map[OpType.SOFTMAX], "libes_onnx_softmax.so")

        # Run GEMM -> Conv1x1 pass
        conv_pass = GemmToConv1x1Pass()
        conv_pass.run(graph)

        # Check that Q, K, V, Score, and Out projections became CONV
        conv_nodes = [n for n in graph.nodes if n.op_type == OpType.CONV]
        self.assertEqual(len(conv_nodes), 5)
        for c in conv_nodes:
            self.assertEqual(c.engine, EngineType.NPU)
            self.assertEqual(c.attributes["kernel_shape"], [1, 1])


if __name__ == "__main__":
    unittest.main()
