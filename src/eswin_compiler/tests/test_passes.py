"""Unit tests for EIR optimization passes."""

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
from passes.gemm_to_conv1x1 import GemmToConv1x1Pass


class TestOptimizationPasses(unittest.TestCase):
    """Test suite for EIR graph transformations."""

    def test_gemm_to_conv1x1(self):
        """Test lowering a GEMM node into a 1x1 Pointwise Convolution."""
        graph = EIRGraph(name="test_linear")

        # Input: [1, 64, 1, 1]
        x = EIRTensor(name="input", shape=(1, 64, 1, 1), dtype="float16")
        graph.add_input(x)

        # Weight: [128, 64] (transB=1)
        w_data = np.random.randn(128, 64).astype(np.float16)
        w = EIRTensor(name="weight", shape=(1, 1, 128, 64), dtype="float16", data=w_data)
        graph.add_tensor(w)

        # Output: [1, 128, 1, 1]
        y = EIRTensor(name="output", shape=(1, 128, 1, 1), dtype="float16")
        graph.add_output(y)

        node = EIRNode(
            name="gemm_0",
            op_type=OpType.GEMM,
            inputs=[x, w],
            outputs=[y],
            attributes={"transB": 1},
            engine=EngineType.NPU
        )
        graph.add_node(node)
        graph.validate()

        # Run Pass
        pass_gemm = GemmToConv1x1Pass()
        modified = pass_gemm.run(graph)
        self.assertTrue(modified)

        # Verify mutation
        self.assertEqual(node.op_type, OpType.CONV)
        self.assertEqual(node.engine, EngineType.NPU)
        self.assertEqual(node.attributes["kernel_shape"], [1, 1])
        self.assertEqual(node.attributes["strides"], [1, 1])
        self.assertEqual(node.attributes["pads"], [0, 0, 0, 0])

        # Verify weight shape lowered to 4D [C_out, C_in, 1, 1]
        self.assertEqual(w.shape, (128, 64, 1, 1))
        self.assertEqual(w.data.shape, (128, 64, 1, 1))
        self.assertEqual(y.shape, (1, 128, 1, 1))


if __name__ == "__main__":
    unittest.main()
