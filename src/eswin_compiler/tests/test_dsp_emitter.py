"""Unit tests for DSPEmitter and Cadence Vision Q7 operator descriptors."""

from pathlib import Path
import sys
import unittest

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.dsp_emitter import DSPEmitter, DSPOperatorDesc, DSPKernelType


class TestDSPEmitter(unittest.TestCase):
    """Test suite for DSP task descriptor packing and manifest generation."""

    def test_emit_rms_norm_desc(self):
        """Test packing an RMSNorm descriptor for SmolLM/Llama (576 channels)."""
        desc = DSPEmitter.emit_rms_norm(channels=576, epsilon=1e-5)
        self.assertEqual(desc.op_name, "rms_norm_f16f16")
        self.assertEqual(desc.kernel_lib_name, "es_rms_norm_f16f16.pkg.lib")
        self.assertEqual(desc.buffer_cnt_cfg, 1)
        self.assertEqual(desc.buffer_cnt_input, 2)
        self.assertEqual(desc.buffer_cnt_output, 1)

        raw = desc.pack_operator_desc()
        # Header (400 bytes) + Cfg (8 bytes: uint32 + float32) = 408 bytes
        self.assertEqual(len(raw), 408)
        self.assertEqual(desc.total_desc_size, 408)

    def test_emit_softmax_desc(self):
        """Test packing a Softmax descriptor."""
        desc = DSPEmitter.emit_softmax(channels=2048, axis=-1)
        self.assertEqual(desc.op_name, "onnx_softmax")
        self.assertEqual(desc.kernel_lib_name, "es_onnx_softmax.pkg.lib")
        raw = desc.pack_operator_desc()
        self.assertEqual(len(raw), 408)

    def test_emit_rope_desc(self):
        """Test packing a RoPE descriptor for transformer attention."""
        desc = DSPEmitter.emit_rope(seq_len=64, head_dim=64, num_heads=9)
        self.assertEqual(desc.op_name, "qk_rotary_position_embedding_f16f16")
        self.assertEqual(desc.buffer_cnt_input, 2)
        self.assertEqual(desc.buffer_cnt_output, 2)
        raw = desc.pack_operator_desc()
        self.assertEqual(len(raw), 416)

    def test_emit_yolov11_detection_desc(self):
        """Test packing a YOLOv11 NMS and detection head descriptor."""
        desc = DSPEmitter.emit_yolov11_detection(num_classes=80, conf_threshold=0.25)
        self.assertEqual(desc.op_name, "detection_out_f16f16_yolov11")
        self.assertEqual(desc.kernel_lib_name, "es_detection_out_f16f16_yolov11.pkg.lib")
        entry = DSPEmitter.emit_manifest_entry(desc)
        self.assertEqual(entry["engine"], "DSP")
        self.assertEqual(entry["op_name"], "detection_out_f16f16_yolov11")


if __name__ == "__main__":
    unittest.main()
