"""Unit tests for SDP LUT Generation, Activation Fusion, and SRAM Tiling passes."""

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

from backend.lut_generator import ActivationType, generate_sdp_lut
from backend.nvdla_builder import compile_eir_to_model
from ir.graph import EIRGraph
from ir.node import EIRNode
from ir.op_types import EngineType, OpType
from ir.tensor import EIRTensor
from passes.fusion_sdp import SDPFusionPass
from passes.sram_tiler import SRAMTilerPass


class TestSDPFusion(unittest.TestCase):
    """Test suite for SDP activation fusion, LUT table synthesis, and SRAM tiling."""

    def test_sdp_lut_generation_sigmoid(self):
        """Test generating Sigmoid piecewise LUT table."""
        lut_bytes = generate_sdp_lut(ActivationType.SIGMOID, domain=(-8.0, 8.0))
        self.assertEqual(len(lut_bytes), 688)

        # Verify header fields
        precision, cfg, info, le_start, le_end, lo_start, lo_end = struct.unpack_from("<IIIffff", lut_bytes, 0)
        self.assertEqual(precision, 2)       # FP16
        self.assertEqual(cfg, 0x0031)
        self.assertEqual(le_start, -8.0)
        self.assertEqual(le_end, 8.0)

        # Inspect midpoint value in linear_only_table (index 128 of 257 is x = 0.0)
        # Midpoint of sigmoid(0.0) == 0.5
        lo_midpoint = struct.unpack_from("<e", lut_bytes, 44 + 130 + 128 * 2)[0]
        self.assertAlmostEqual(lo_midpoint, 0.5, places=2)

    def test_sdp_lut_generation_silu(self):
        """Test generating SiLU piecewise LUT table."""
        lut_bytes = generate_sdp_lut(ActivationType.SILU, domain=(-8.0, 8.0))
        self.assertEqual(len(lut_bytes), 688)

        # Midpoint of silu(0.0) == 0.0
        lo_midpoint = struct.unpack_from("<e", lut_bytes, 44 + 130 + 128 * 2)[0]
        self.assertAlmostEqual(lo_midpoint, 0.0, places=2)

    def test_bias_fusion(self):
        """Test fusing a downstream Add (bias) node into Conv."""
        graph = EIRGraph(name="test_bias_graph")

        x = EIRTensor(name="input", shape=(1, 16, 8, 8), dtype="float16")
        w = EIRTensor(name="weight", shape=(32, 16, 1, 1), dtype="float16", data=np.zeros((32, 16, 1, 1), dtype=np.float16))
        conv_out = EIRTensor(name="conv_out", shape=(1, 32, 8, 8), dtype="float16")

        node_conv = EIRNode(name="conv", op_type=OpType.CONV, inputs=[x, w], outputs=[conv_out], engine=EngineType.NPU)
        graph.add_input(x)
        graph.add_tensor(w)
        graph.add_node(node_conv)

        # Add bias
        b = EIRTensor(name="bias", shape=(1, 32, 1, 1), dtype="float16", data=np.ones((1, 32, 1, 1), dtype=np.float16))
        add_out = EIRTensor(name="output", shape=(1, 32, 8, 8), dtype="float16")
        node_add = EIRNode(name="add_bias", op_type=OpType.ADD, inputs=[conv_out, b], outputs=[add_out], engine=EngineType.NPU)
        graph.add_tensor(b)
        graph.add_node(node_add)
        graph.add_output(add_out)
        graph.validate()

        # Run SDPFusionPass
        pass_sdp = SDPFusionPass()
        modified = pass_sdp.run(graph)
        self.assertTrue(modified)

        # Verify Add node was fused into Conv
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].op_type, OpType.CONV)
        self.assertTrue(graph.nodes[0].attributes.get("has_bias", False))
        self.assertIn(b, graph.nodes[0].inputs)
        self.assertEqual(graph.nodes[0].outputs[0].name, "output")

    def test_activation_fusion_silu(self):
        """Test fusing a downstream SiLU activation node into Conv."""
        graph = EIRGraph(name="test_silu_graph")

        x = EIRTensor(name="input", shape=(1, 16, 8, 8), dtype="float16")
        w = EIRTensor(name="weight", shape=(32, 16, 1, 1), dtype="float16", data=np.zeros((32, 16, 1, 1), dtype=np.float16))
        conv_out = EIRTensor(name="conv_out", shape=(1, 32, 8, 8), dtype="float16")

        node_conv = EIRNode(name="conv", op_type=OpType.CONV, inputs=[x, w], outputs=[conv_out], engine=EngineType.NPU)
        graph.add_input(x)
        graph.add_tensor(w)
        graph.add_node(node_conv)

        # SiLU node
        act_out = EIRTensor(name="output", shape=(1, 32, 8, 8), dtype="float16")
        node_silu = EIRNode(name="silu", op_type=OpType.SILU, inputs=[conv_out], outputs=[act_out], engine=EngineType.NPU)
        graph.add_node(node_silu)
        graph.add_output(act_out)
        graph.validate()

        # Compile model through pipeline
        model = compile_eir_to_model(graph)

        # Assert LUT blob was generated with 688 non-zero bytes
        lut_blob = [b for b in model.blobs if b.name == "kmd-task-lut_list"][0]
        self.assertEqual(len(lut_blob.data), 688)
        self.assertGreater(sum(1 for b in lut_blob.data if b != 0), 100)

    def test_sram_tiler(self):
        """Test 4MB SPRAM residency planning."""
        graph = EIRGraph(name="test_sram_graph")

        # Input & Output stay in SYSTEM DRAM
        x = EIRTensor(name="input", shape=(1, 16, 4, 4), dtype="float16")
        y = EIRTensor(name="output", shape=(1, 16, 4, 4), dtype="float16")
        graph.add_input(x)
        graph.add_output(y)

        # Small weight fits in 4MB SRAM
        small_w = EIRTensor(
            name="small_weight",
            shape=(16, 16, 1, 1),
            dtype="float16",
            data=np.zeros((16, 16, 1, 1), dtype=np.float16)
        )
        graph.add_tensor(small_w)

        node = EIRNode(name="conv", op_type=OpType.CONV, inputs=[x, small_w], outputs=[y], engine=EngineType.NPU)
        graph.add_node(node)
        graph.validate()

        # Run SRAM Tiler
        tiler = SRAMTilerPass(sram_capacity=4 * 1024 * 1024)
        tiler.run(graph)

        # Assert small weight resides in SRAM
        self.assertEqual(small_w.memory_domain, "SRAM")
        self.assertEqual(x.memory_domain, "SYSTEM")
        self.assertEqual(y.memory_domain, "SYSTEM")


if __name__ == "__main__":
    unittest.main()
