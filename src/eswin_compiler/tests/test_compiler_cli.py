"""Unit tests for EswinCompiler driver and end-to-end compilation pipeline."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.flatbuffer_emitter import ModelReader
from compiler import CompilerConfig, EswinCompiler


class TestCompilerPipeline(unittest.TestCase):
    """Test suite for end-to-end model compilation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compile_safetensors_dual_die(self):
        """Test compiling an LLM with dual-die tensor parallel LM head and layer offload."""
        # Create a mock model dir with config.json
        model_dir = self.temp_path / "mock_smollm"
        model_dir.mkdir()

        config = {
            "hidden_size": 128,
            "intermediate_size": 256,
            "vocab_size": 4096,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Output dir
        out_dir = self.temp_path / "output_dual_die"

        cfg = CompilerConfig(target="dual-die", opt_level=2, dtype="float16", batch_size=1)
        compiler = EswinCompiler(config=cfg)

        # Compile first 2 layers
        emitted_files = compiler.compile_safetensors_llm(
            model_source=model_dir,
            output_dir=out_dir,
            model_name="smollm_test",
            num_layers_to_compile=2
        )

        # Check emitted files
        # Expected:
        # - smollm_test_lm_head_d0_npu_b1.model
        # - smollm_test_lm_head_d1_npu_b1.model
        # - smollm_test_mlp_l0_d0_npu_b1.model
        # - smollm_test_mlp_l1_d1_npu_b1.model
        # - smollm_test_attn_l0_d0_npu_b1.model
        # - smollm_test_attn_l1_d1_npu_b1.model
        self.assertEqual(len(emitted_files), 6)

        # Verify each emitted file is a valid .model FlatBuffer
        for f in emitted_files:
            self.assertTrue(f.is_file())
            data = f.read_bytes()
            self.assertEqual(data[4:8], b"NVDA")

            # Parse back with ModelReader
            model = ModelReader.read_bytes(data)
            self.assertGreater(len(model.tasks), 0)
            self.assertGreater(len(model.memory_list), 0)
            self.assertGreater(len(model.address_list), 0)
            self.assertGreater(len(model.blobs), 0)
            self.assertGreater(len(model.tensor_descs), 0)

        # Verify LM Head vocabulary half split
        head_d0 = ModelReader.read_file(out_dir / "smollm_test_lm_head_d0_npu_b1.model")
        head_d1 = ModelReader.read_file(out_dir / "smollm_test_lm_head_d1_npu_b1.model")

        # Vocab = 4096 -> half is 2048
        # Output tensor size = 2048 * sizeof(float16) = 4096 bytes
        self.assertEqual(head_d0.tensor_descs[1].size, 2048 * 2)
        self.assertEqual(head_d1.tensor_descs[1].size, 2048 * 2)


if __name__ == "__main__":
    unittest.main()
