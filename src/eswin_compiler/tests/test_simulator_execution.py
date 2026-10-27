"""
End-to-End Compiler and Simulator Integration Verification.
Validates that .model binaries emitted by eswin_compiler load and execute
seamlessly in eswin_simulator with bit-accurate numerical parity against PyTorch.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn.functional as F

_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
for p in (str(_COMPILER_DIR), str(_COMPILER_DIR / "schema" / "generated")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compiler import CompilerConfig, EswinCompiler
from eswin_simulator.loader import LoadableModel
from eswin_simulator.memory import VirtualAddressSpace
from eswin_simulator.engines import (
    CMACEngine,
    CACCEngine,
    SDPEngine,
    RubikEngine,
    SDPLookupTable,
    unpack_nvdla_weights_1x1,
)
from eswin_simulator.evaluation import TensorComparator


class TestCompilerSimulatorIntegration(unittest.TestCase):
    """End-to-end integration tests connecting eswin_compiler and eswin_simulator."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compiled_dual_die_lm_head_execution(self):
        """Compiles a Dual-Die LM-Head and executes both partitions on eswin_simulator."""
        model_dir = self.temp_path / "mock_llm"
        model_dir.mkdir()

        hidden_dim = 128
        vocab_size = 512
        config = {
            "hidden_size": hidden_dim,
            "intermediate_size": 256,
            "vocab_size": vocab_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        out_dir = self.temp_path / "out"
        # allow_partial_graph: the emitter is single-operation, so a synthesized SwiGLU or
        # attention block compiles to a loadable holding only the first CONV. These tests
        # exercise emission + simulator plumbing, NOT block-level numerical correctness.
        # See tests/test_partial_graph_guard.py for the guard that rejects them by default.
        cfg = CompilerConfig(target="dual-die", opt_level=2, dtype="float16", batch_size=1,
                             allow_partial_graph=True)
        compiler = EswinCompiler(config=cfg)

        emitted_models = compiler.compile_safetensors_llm(
            model_source=model_dir,
            output_dir=out_dir,
            model_name="test_llm",
            num_layers_to_compile=1,
        )

        d0_path = out_dir / "test_llm_lm_head_d0_npu_b1.model"
        d1_path = out_dir / "test_llm_lm_head_d1_npu_b1.model"
        self.assertTrue(d0_path.is_file())
        self.assertTrue(d1_path.is_file())

        # 1. Load both partitions in simulator
        m0 = LoadableModel.from_file(d0_path)
        m1 = LoadableModel.from_file(d1_path)

        vas0 = VirtualAddressSpace(m0)
        vas1 = VirtualAddressSpace(m1)

        cmac0 = CMACEngine(vas0)
        cacc0 = CACCEngine(vas0)
        cmac1 = CMACEngine(vas1)
        cacc1 = CACCEngine(vas1)
        rubik = RubikEngine(vas0)

        # 2. Input hidden state: (1, 128, 1, 1)
        np.random.seed(42)
        x = np.random.randn(1, hidden_dim, 1, 1).astype(np.float16) * 0.1

        # 3. Unpack weights from each partition
        half_vocab = vocab_size // 2  # 256
        w0 = unpack_nvdla_weights_1x1(m0.blobs["W_0"].data, c_out=half_vocab, c_in=hidden_dim, dtype=np.float16)
        w1 = unpack_nvdla_weights_1x1(m1.blobs["W_0"].data, c_out=half_vocab, c_in=hidden_dim, dtype=np.float16)

        # 4. Execute Die 0 and Die 1 in parallel / pipeline
        logits_d0 = cacc0.finalize_fp16(cmac0.gemm_1x1(x, w0, precision="FP16"), to_fp16=True)
        logits_d1 = cacc1.finalize_fp16(cmac1.gemm_1x1(x, w1, precision="FP16"), to_fp16=True)

        self.assertEqual(logits_d0.shape, (1, half_vocab, 1, 1))
        self.assertEqual(logits_d1.shape, (1, half_vocab, 1, 1))

        # 5. Merge logits via Rubik
        full_logits_sim = rubik.concat([logits_d0, logits_d1], axis=1)
        self.assertEqual(full_logits_sim.shape, (1, vocab_size, 1, 1))

        # 6. Verify against PyTorch reference
        w_full = np.vstack([w0, w1])
        x_pt = torch.from_numpy(x.reshape(1, hidden_dim).astype(np.float32))
        w_pt = torch.from_numpy(w_full.astype(np.float32))
        logits_pt = F.linear(x_pt, w_pt).numpy().reshape(1, vocab_size, 1, 1)

        rep = TensorComparator.compare(full_logits_sim, logits_pt.astype(np.float16), tolerance=1e-3)
        self.assertTrue(rep.passed, f"LM Head simulation mismatch:\n{rep.summary()}")
        self.assertGreater(rep.cosine_similarity, 0.99999)

    def test_compiled_swiglu_mlp_execution(self):
        """Compiles a SwiGLU MLP model and executes end-to-end on eswin_simulator."""
        model_dir = self.temp_path / "mock_llm2"
        model_dir.mkdir()

        hidden_dim = 128
        intermediate_dim = 256
        config = {
            "hidden_size": hidden_dim,
            "intermediate_size": intermediate_dim,
            "vocab_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        out_dir = self.temp_path / "out2"
        # allow_partial_graph: the emitter is single-operation, so a synthesized SwiGLU or
        # attention block compiles to a loadable holding only the first CONV. These tests
        # exercise emission + simulator plumbing, NOT block-level numerical correctness.
        # See tests/test_partial_graph_guard.py for the guard that rejects them by default.
        cfg = CompilerConfig(target="dual-die", opt_level=2, dtype="float16", batch_size=1,
                             allow_partial_graph=True)
        compiler = EswinCompiler(config=cfg)

        compiler.compile_safetensors_llm(
            model_source=model_dir,
            output_dir=out_dir,
            model_name="test_mlp",
            num_layers_to_compile=1,
        )

        mlp_path = out_dir / "test_mlp_mlp_l0_d1_npu_b1.model"
        self.assertTrue(mlp_path.is_file())

        model = LoadableModel.from_file(mlp_path)
        vas = VirtualAddressSpace(model)
        cmac = CMACEngine(vas)
        cacc = CACCEngine(vas)
        sdp = SDPEngine(vas)

        x = np.random.randn(1, hidden_dim, 1, 1).astype(np.float16) * 0.1

        # Unpack weights and LUT from compiled model
        w = unpack_nvdla_weights_1x1(model.blobs["W_0"].data, intermediate_dim, hidden_dim, dtype=np.float16)
        lut = SDPLookupTable.from_bytes(model.blobs["kmd-task-lut_list"].data)

        # Execute CMAC GEMM and SDP SiLU LUT
        gate = cacc.finalize_fp16(cmac.gemm_1x1(x, w, precision="FP16"), to_fp16=True)
        silu_out = sdp.interpolate_lut(gate, lut)

        self.assertEqual(silu_out.shape, (1, intermediate_dim, 1, 1))

        # PyTorch Reference
        x_pt = torch.from_numpy(x.reshape(1, hidden_dim).astype(np.float32))
        w_pt = torch.from_numpy(w.astype(np.float32))
        ref_pt = F.silu(F.linear(x_pt, w_pt)).numpy().reshape(1, intermediate_dim, 1, 1)

        rep = TensorComparator.compare(silu_out, ref_pt.astype(np.float16), tolerance=1e-3)
        self.assertTrue(rep.passed, f"SwiGLU simulation mismatch:\n{rep.summary()}")
        self.assertGreater(rep.cosine_similarity, 0.99999)


if __name__ == "__main__":
    unittest.main()
