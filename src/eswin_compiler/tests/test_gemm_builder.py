"""Unit and integration tests for NVDLA Multi-Pass Linear / GEMM Builder."""

from pathlib import Path
import sys
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.gemm_builder import NVDLAGEMMBuilder, pack_nvdla_gemm_weights
from backend.flatbuffer_emitter import ModelWriter, ModelReader
from ir.graph import EIRGraph
from ir.node import EIRNode
from ir.tensor import EIRTensor
from ir.op_types import EngineType, OpType


class TestNVDLAGEMMBuilder(unittest.TestCase):
    """Verifies 3-pass linear / GEMM compilation and weight tiling."""

    def test_pack_nvdla_gemm_weights(self):
        """Test packing FP16 GEMM weights into NVDLA hardware atomic groups."""
        w = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
        packed = pack_nvdla_gemm_weights(w, cout_sub=8, cin_sub=2)
        # Expected size: 16 channels padded to 16, 8 channels padded to 8 -> 16 * 8 * 2 = 256 bytes
        self.assertEqual(len(packed), 256)

        # Output shape is (2, 4, 8, 2) in FP16
        arr = np.frombuffer(packed, dtype=np.float16).reshape(2, 4, 8, 2)
        # Check first channel: w[0, 0] and w[0, 1]
        self.assertEqual(arr[0, 0, 0, 0], 0.0)
        self.assertEqual(arr[0, 0, 0, 1], 1.0)
        # Check second channel: w[1, 0] and w[1, 1]
        self.assertEqual(arr[0, 0, 1, 0], 8.0)
        self.assertEqual(arr[0, 0, 1, 1], 9.0)

    def test_build_linear_3pass_model(self):
        """Test building a full 3-pass Linear / GEMM model (576 -> 1536)."""
        np.random.seed(42)
        w = (np.random.randn(1536, 576) * 0.02).astype(np.float32)

        model = NVDLAGEMMBuilder.build_linear_3pass(
            w_arr=w,
            act_in_name="in_features",
            act_out_name="out_features",
            model_name="test_linear3p"
        )

        self.assertEqual(model.version.major, 0)
        self.assertEqual(model.version.minor, 7)
        self.assertEqual(len(model.tasks), 2)
        self.assertEqual(len(model.memory_list), 18)
        self.assertEqual(len(model.relocs), 8)
        self.assertEqual(len(model.blobs), 14)
        self.assertEqual(len(model.tensor_descs), 2)

        # Verify weight blobs
        self.assertEqual(model.blobs[0].name, "WsplitM_0_0")
        self.assertEqual(model.blobs[1].name, "WsplitM_1_1")
        self.assertEqual(model.blobs[2].name, "WsplitM_2_2")
        self.assertEqual(model.blobs[0].size, 589824)

        # Verify serialization round-trip
        ser = ModelWriter.serialize(model)
        self.assertGreater(len(ser), 1024 * 1024)

        parsed = ModelReader.read_bytes(ser)
        self.assertEqual(len(parsed.tasks), 2)
        self.assertEqual(len(parsed.blobs), 14)
        self.assertEqual(parsed.tensor_descs[0].name, "in_features")
        self.assertEqual(parsed.tensor_descs[1].name, "out_features")

    def test_build_from_node(self):
        """Test building a GEMM model from an EIRNode."""
        x = EIRTensor(name="gemm_in", shape=(1, 576, 1, 1), dtype="float16")
        w_data = np.zeros((1536, 576), dtype=np.float32)
        w = EIRTensor(name="gemm_w", shape=(1536, 576, 1, 1), dtype="float16", data=w_data)
        y = EIRTensor(name="gemm_out", shape=(1, 1536, 1, 1), dtype="float16")

        node = EIRNode(
            name="proj_linear",
            op_type=OpType.CONV,
            inputs=[x, w],
            outputs=[y],
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            engine=EngineType.NPU
        )

        model = NVDLAGEMMBuilder.build_from_node(node)
        self.assertEqual(len(model.blobs), 14)
        self.assertEqual(model.tensor_descs[0].name, "gemm_in")
        self.assertEqual(model.tensor_descs[1].name, "gemm_out")


if __name__ == "__main__":
    unittest.main()
