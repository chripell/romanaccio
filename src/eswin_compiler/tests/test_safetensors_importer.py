"""Unit tests for SafeTensorsReader and SafeTensorsImporter."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from frontend.safetensors_importer import SafeTensorsImporter, SafeTensorsReader
from ir.op_types import EngineType, OpType


class TestSafeTensorsImporter(unittest.TestCase):
    """Test suite for SafeTensors parsing and LLM computational graph synthesis."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_st_file = Path(cls.temp_dir.name) / "test_weights.safetensors"

        # Create a synthetic safetensors file
        w_gate = np.arange(128 * 64, dtype=np.float16).reshape((128, 64))
        w_up = (np.arange(128 * 64, dtype=np.float16) * 0.5).reshape((128, 64))
        w_down = (np.arange(64 * 128, dtype=np.float16) * 0.1).reshape((64, 128))
        w_lm = np.ones((512, 64), dtype=np.float16)

        gate_bytes = w_gate.tobytes()
        up_bytes = w_up.tobytes()
        down_bytes = w_down.tobytes()
        lm_bytes = w_lm.tobytes()

        o0 = 0
        o1 = o0 + len(gate_bytes)
        o2 = o1 + len(up_bytes)
        o3 = o2 + len(down_bytes)
        o4 = o3 + len(lm_bytes)

        header = {
            "model.layers.0.mlp.gate_proj.weight": {
                "dtype": "F16",
                "shape": [128, 64],
                "data_offsets": [o0, o1]
            },
            "model.layers.0.mlp.up_proj.weight": {
                "dtype": "F16",
                "shape": [128, 64],
                "data_offsets": [o1, o2]
            },
            "model.layers.0.mlp.down_proj.weight": {
                "dtype": "F16",
                "shape": [64, 128],
                "data_offsets": [o2, o3]
            },
            "lm_head.weight": {
                "dtype": "F16",
                "shape": [512, 64],
                "data_offsets": [o3, o4]
            }
        }
        header_json = json.dumps(header).encode("utf-8")
        header_len = len(header_json)

        with open(cls.test_st_file, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            f.write(gate_bytes)
            f.write(up_bytes)
            f.write(down_bytes)
            f.write(lm_bytes)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_safetensors_reader(self):
        """Test reading weights and metadata from synthetic .safetensors file."""
        reader = SafeTensorsReader(self.test_st_file)
        keys = reader.keys()
        self.assertIn("model.layers.0.mlp.gate_proj.weight", keys)
        self.assertIn("lm_head.weight", keys)

        gate_w = reader.get_tensor("model.layers.0.mlp.gate_proj.weight")
        self.assertEqual(gate_w.shape, (128, 64))
        self.assertEqual(gate_w.dtype, np.float16)
        self.assertEqual(gate_w[0, 1], 1.0)

    def test_synthesize_mlp_block(self):
        """Test synthesizing SwiGLU MLP EIR graph from safetensors."""
        reader = SafeTensorsReader(self.test_st_file)
        importer = SafeTensorsImporter(reader)

        graph = importer.synthesize_mlp_block(
            layer_idx=0,
            hidden_size=64,
            intermediate_size=128
        )
        graph.validate()

        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs[0].name, "input")
        self.assertEqual(graph.inputs[0].shape, (1, 64, 1, 1))

        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].name, "output")
        self.assertEqual(graph.outputs[0].shape, (1, 64, 1, 1))

        # Check nodes: GEMM -> SILU -> GEMM -> MUL -> GEMM
        op_types = [n.op_type for n in graph.nodes]
        self.assertIn(OpType.GEMM, op_types)
        self.assertIn(OpType.SILU, op_types)
        self.assertIn(OpType.MUL, op_types)

    def test_synthesize_lm_head_dual_die(self):
        """Test synthesizing dual-die split LM head."""
        reader = SafeTensorsReader(self.test_st_file)
        importer = SafeTensorsImporter(reader)

        # Die 0 head (vocab 0..255)
        graph_d0 = importer.synthesize_lm_head(hidden_size=64, vocab_size=512, die=0)
        graph_d0.validate()
        self.assertEqual(graph_d0.outputs[0].shape, (1, 256, 1, 1))

        # Die 1 head (vocab 256..511)
        graph_d1 = importer.synthesize_lm_head(hidden_size=64, vocab_size=512, die=1)
        graph_d1.validate()
        self.assertEqual(graph_d1.outputs[0].shape, (1, 256, 1, 1))


if __name__ == "__main__":
    unittest.main()
