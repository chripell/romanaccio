"""Unit tests for ONNXImporter."""

import os
from pathlib import Path
import sys
import unittest

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from frontend.onnx_importer import ONNXImporter
from ir.op_types import EngineType, OpType


class TestONNXImporter(unittest.TestCase):
    """Test suite for importing ONNX graphs into EIR."""

    @classmethod
    def setUpClass(cls):
        test_data_dir = os.environ.get("ESWIN_TEST_DATA", "")
        cls.artifacts_dir = Path(test_data_dir) if test_data_dir else Path(__file__).resolve().parent / "data"
        cls.conv_onnx = cls.artifacts_dir / "test_conv.onnx"
        cls.swiglu_onnx = cls.artifacts_dir / "test_swiglu_mlp.onnx"

    def test_import_conv_onnx(self):
        """Test importing simple Conv2D model."""
        if not self.conv_onnx.exists():
            self.skipTest(f"Test model not present at {self.conv_onnx}")

        importer = ONNXImporter()
        graph = importer.import_model(self.conv_onnx)

        # Validate graph
        graph.validate()

        # Check Inputs
        self.assertEqual(len(graph.inputs), 1)
        inp = graph.inputs[0]
        self.assertEqual(inp.name, "input")
        self.assertEqual(inp.shape, (1, 3, 32, 32))

        # Check Outputs
        self.assertEqual(len(graph.outputs), 1)
        out = graph.outputs[0]
        self.assertEqual(out.name, "output")
        self.assertEqual(out.shape, (1, 16, 30, 30))

        # Check Conv Node
        self.assertGreaterEqual(len(graph.nodes), 1)
        conv_nodes = [n for n in graph.nodes if n.op_type == OpType.CONV]
        self.assertEqual(len(conv_nodes), 1)
        node = conv_nodes[0]
        self.assertEqual(node.engine, EngineType.NPU)
        self.assertIn("kernel_shape", node.attributes)
        self.assertEqual(list(node.attributes["kernel_shape"]), [3, 3])

    def test_import_swiglu_onnx(self):
        """Test importing SwiGLU MLP ONNX model if present."""
        if not self.swiglu_onnx.exists():
            self.skipTest("test_swiglu_mlp.onnx not present")

        importer = ONNXImporter()
        graph = importer.import_model(self.swiglu_onnx)
        graph.validate()

        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs[0].shape[0], 1)
        self.assertEqual(len(graph.outputs), 1)

        # Verify topological sort order
        sorted_nodes = graph.topological_sort()
        self.assertEqual(len(sorted_nodes), len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
