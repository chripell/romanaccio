"""Unit tests for Dual-Die Tensor Parallelism and Pipeline Partitioning."""

from pathlib import Path
import sys
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from ir.graph import EIRGraph
from ir.node import EIRNode
from ir.op_types import EngineType, OpType
from ir.tensor import EIRTensor
from passes.dual_die_splitter import DualDieSplitterPass


class TestDualDieSplitter(unittest.TestCase):
    """Test suite for dual-die tensor parallel and pipeline splitting."""

    def test_vocab_tensor_parallel_split(self):
        """Test splitting a wide LM head (e.g. 576 -> 4096) into Die 0 and Die 1 halves."""
        graph = EIRGraph(name="test_lm_head")

        hidden_dim = 64
        vocab_dim = 4096

        # Input: [1, hidden_dim, 1, 1]
        x = EIRTensor(name="input", shape=(1, hidden_dim, 1, 1), dtype="float16")
        graph.add_input(x)

        # Weight: [vocab_dim, hidden_dim, 1, 1]
        w_data = np.arange(vocab_dim * hidden_dim, dtype=np.float16).reshape(vocab_dim, hidden_dim, 1, 1)
        w = EIRTensor(name="lm_head.weight", shape=(vocab_dim, hidden_dim, 1, 1), dtype="float16", data=w_data)
        graph.add_tensor(w)

        # Bias: [vocab_dim, 1, 1, 1]
        b_data = np.arange(vocab_dim, dtype=np.float16).reshape(vocab_dim, 1, 1, 1)
        b = EIRTensor(name="lm_head.bias", shape=(vocab_dim, 1, 1, 1), dtype="float16", data=b_data)
        graph.add_tensor(b)

        # Output: [1, vocab_dim, 1, 1]
        out = EIRTensor(name="logits", shape=(1, vocab_dim, 1, 1), dtype="float16")
        graph.add_output(out)

        node = EIRNode(
            name="conv_lm_head",
            op_type=OpType.CONV,
            inputs=[x, w, b],
            outputs=[out],
            attributes={"kernel_shape": [1, 1], "strides": [1, 1], "pads": [0, 0, 0, 0]},
            engine=EngineType.NPU
        )
        graph.add_node(node)

        # Apply DualDieSplitterPass
        pass_split = DualDieSplitterPass(vocab_split_threshold=2048)
        self.assertTrue(pass_split.run(graph))

        g_d0, g_d1 = pass_split.split_graph_tensor_parallel(graph)

        # Verify Die 0 graph
        self.assertEqual(g_d0.name, "test_lm_head_d0")
        w_d0 = g_d0.nodes[0].inputs[1]
        b_d0 = g_d0.nodes[0].inputs[2]
        out_d0 = g_d0.nodes[0].outputs[0]

        self.assertEqual(w_d0.shape, (2048, hidden_dim, 1, 1))
        self.assertEqual(b_d0.shape, (2048, 1, 1, 1))
        self.assertEqual(out_d0.shape, (1, 2048, 1, 1))
        np.testing.assert_array_equal(w_d0.data, w_data[:2048])
        np.testing.assert_array_equal(b_d0.data, b_data[:2048])

        # Verify Die 1 graph
        self.assertEqual(g_d1.name, "test_lm_head_d1")
        w_d1 = g_d1.nodes[0].inputs[1]
        b_d1 = g_d1.nodes[0].inputs[2]
        out_d1 = g_d1.nodes[0].outputs[0]

        self.assertEqual(w_d1.shape, (2048, hidden_dim, 1, 1))
        self.assertEqual(b_d1.shape, (2048, 1, 1, 1))
        self.assertEqual(out_d1.shape, (1, 2048, 1, 1))
        np.testing.assert_array_equal(w_d1.data, w_data[2048:])
        np.testing.assert_array_equal(b_d1.data, b_data[2048:])

    def test_pipeline_layers_partitioning(self):
        """Test splitting sequential layers across Die 0 and Die 1."""
        layers = [EIRGraph(name=f"layer_{i}") for i in range(10)]
        d0, d1 = DualDieSplitterPass.partition_pipeline_layers(layers)

        self.assertEqual(len(d0), 5)
        self.assertEqual(len(d1), 5)
        self.assertEqual(d0[0].name, "layer_0_d0")
        self.assertEqual(d0[4].name, "layer_4_d0")
        self.assertEqual(d1[0].name, "layer_5_d1")
        self.assertEqual(d1[4].name, "layer_9_d1")


if __name__ == "__main__":
    unittest.main()
