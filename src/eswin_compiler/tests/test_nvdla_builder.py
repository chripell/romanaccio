"""Unit tests for NVDLAConvBuilder and end-to-end model compilation."""

from pathlib import Path
import sys
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.flatbuffer_emitter import ModelReader, ModelWriter
from backend.nvdla_builder import NVDLAConvBuilder, compile_eir_to_model
from ir.graph import EIRGraph
from ir.node import EIRNode
from ir.op_types import EngineType, OpType
from ir.tensor import EIRTensor


class TestNVDLABuilder(unittest.TestCase):
    """Test suite for NVDLA command stream assembly and .model emission."""

    def test_build_conv_model(self):
        """Test building a LoadableModel for a 2D Convolution."""
        graph = EIRGraph(name="test_conv_graph")

        x = EIRTensor(name="input", shape=(1, 3, 32, 32), dtype="float16")
        graph.add_input(x)

        w_data = np.random.randn(16, 3, 3, 3).astype(np.float16)
        w = EIRTensor(name="W_0", shape=(16, 3, 3, 3), dtype="float16", data=w_data)
        graph.add_tensor(w)

        y = EIRTensor(name="output", shape=(1, 16, 30, 30), dtype="float16")
        graph.add_output(y)

        node = EIRNode(
            name="conv_0",
            op_type=OpType.CONV,
            inputs=[x, w],
            outputs=[y],
            attributes={"kernel_shape": [3, 3], "strides": [1, 1]},
            engine=EngineType.NPU
        )
        graph.add_node(node)

        # Build loadable model
        model = NVDLAConvBuilder.build_from_node(node, model_name="test_conv")
        self.assertEqual(str(model.version), "0.7.0")
        self.assertEqual(len(model.tasks), 2)
        self.assertEqual(len(model.tensor_descs), 2)
        self.assertEqual(model.tensor_descs[0].name, "input")
        self.assertEqual(model.tensor_descs[1].name, "output")
        self.assertEqual(len(model.relocs), 4)
        self.assertEqual(len(model.memory_list), 18)

        # Test serializing to bytes
        serialized = ModelWriter.serialize(model)
        self.assertEqual(serialized[4:8], b"NVDA")

        # Test re-reading
        reparsed = ModelReader.read_bytes(serialized)
        self.assertEqual(len(reparsed.blobs), len(model.blobs))
        self.assertEqual(reparsed.blobs[0].name, "W_0")

    def test_compile_eir_to_model_gemm(self):
        """Test compiling an EIR graph containing a GEMM layer through lowering."""
        graph = EIRGraph(name="test_gemm_graph")

        x = EIRTensor(name="input", shape=(1, 64, 1, 1), dtype="float16")
        graph.add_input(x)

        w_data = np.ones((128, 64), dtype=np.float16)
        w = EIRTensor(name="weight", shape=(1, 1, 128, 64), dtype="float16", data=w_data)
        graph.add_tensor(w)

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

        # Compile through pipeline
        model = compile_eir_to_model(graph)
        self.assertEqual(len(model.tasks), 2)
        self.assertEqual(model.tensor_descs[0].size, 64 * 2)
        self.assertEqual(model.tensor_descs[1].size, 128 * 2)

        # Serialize
        serialized = ModelWriter.serialize(model)
        self.assertEqual(serialized[4:8], b"NVDA")


if __name__ == "__main__":
    unittest.main()
