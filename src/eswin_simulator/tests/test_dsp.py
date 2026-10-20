"""
Unit and differential tests for Cadence Vision Q7 DSP Submodel Engine (Milestone 5).
Validates bit-accurate native fallbacks for RMSNorm, Softmax, RoPE, and TopK against PyTorch.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from eswin_simulator.dsp import (
    DSPExecutor,
    rms_norm,
    softmax,
    apply_rope,
    compute_rope_frequencies,
    topk,
)
from eswin_simulator.memory.address_space import VirtualAddressSpace


class TestDSPRMSNorm:
    """Differential verification of RMSNorm against PyTorch F.rms_norm."""

    @pytest.mark.parametrize("channels", [32, 64, 128, 576, 896, 1024, 2048, 4096])
    def test_rms_norm_channels_parity(self, channels: int):
        torch.manual_seed(42 + channels)
        batch = 2
        seq_len = 8

        x_pt = torch.randn(batch, seq_len, channels, dtype=torch.float32)
        weight_pt = torch.randn(channels, dtype=torch.float32)

        out_sim = rms_norm(x_pt.numpy(), weight_pt.numpy(), eps=1e-5)
        out_pt = F.rms_norm(x_pt, (channels,), weight_pt, eps=1e-5).numpy()

        max_diff = np.max(np.abs(out_sim - out_pt))
        assert max_diff < 1e-5, f"Channels {channels}: max diff {max_diff} exceeds 1e-5"

    def test_rms_norm_fp16_input(self):
        channels = 576
        x_f16 = np.random.randn(1, 4, channels).astype(np.float16)
        w_f16 = np.random.randn(channels).astype(np.float16)

        out_f16 = rms_norm(x_f16, w_f16, eps=1e-5)
        assert out_f16.dtype == np.float16

        # Compare with float32 computation
        out_f32_ref = rms_norm(x_f16.astype(np.float32), w_f16.astype(np.float32), eps=1e-5)
        assert np.allclose(out_f16, out_f32_ref.astype(np.float16), atol=1e-3)

    def test_rms_norm_quantized_int8(self):
        channels = 64
        x = np.random.randn(2, 8, channels).astype(np.float32) * 5.0
        w = np.ones(channels, dtype=np.float32)

        # Scale factor 0.05
        scale = 0.05
        out_int8 = rms_norm(x, w, eps=1e-5, out_dtype=np.int8, out_scale=scale)

        assert out_int8.dtype == np.int8
        assert np.all(out_int8 >= -128) and np.all(out_int8 <= 127)

        # Dequantize and check high correlation with float output
        out_float = rms_norm(x, w, eps=1e-5)
        dequant = out_int8.astype(np.float32) * scale
        abs_err = np.max(np.abs(dequant - out_float))
        assert abs_err <= scale, f"Dequantization error {abs_err} should be within scale {scale}"


class TestDSPSoftmax:
    """Differential verification of Softmax against PyTorch F.softmax."""

    @pytest.mark.parametrize("axis", [-1, 1, 0])
    def test_softmax_axes(self, axis: int):
        torch.manual_seed(123)
        x_pt = torch.randn(4, 16, 64, dtype=torch.float32)

        out_sim = softmax(x_pt.numpy(), axis=axis)
        out_pt = F.softmax(x_pt, dim=axis).numpy()

        max_diff = np.max(np.abs(out_sim - out_pt))
        assert max_diff < 1e-6, f"Axis {axis}: max diff {max_diff} exceeds 1e-6"
        assert np.all(out_sim >= 0.0)
        assert np.allclose(np.sum(out_sim, axis=axis), 1.0, atol=1e-5)

    def test_softmax_large_values_numerical_stability(self):
        # Prevent exp overflow with large inputs (+1000.0)
        x = np.array([1000.0, 1005.0, 995.0, 1002.0], dtype=np.float32)
        out = softmax(x, axis=-1)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        assert np.isclose(np.sum(out), 1.0, atol=1e-6)
        assert np.argmax(out) == 1  # 1005.0 is largest


class TestDSPRoPE:
    """Differential verification of Rotary Position Embeddings."""

    def test_rope_half_mode_huggingface_parity(self):
        torch.manual_seed(999)
        batch = 2
        num_heads = 4
        seq_len = 16
        head_dim = 64

        x_pt = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
        out_sim = apply_rope(x_pt.numpy(), theta_base=10000.0, mode="half")

        # PyTorch reference for HuggingFace half-mode
        dim = head_dim
        half = dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        sinusoid = torch.matmul(torch.arange(seq_len).float().unsqueeze(1), inv_freq.unsqueeze(0))
        sinusoid = torch.cat([sinusoid, sinusoid], dim=-1)
        cos = torch.cos(sinusoid).view(1, 1, seq_len, dim)
        sin = torch.sin(sinusoid).view(1, 1, seq_len, dim)

        x1 = x_pt[..., :half]
        x2 = x_pt[..., half:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        out_ref = (x_pt * cos + x_rot * sin).numpy()

        max_diff = np.max(np.abs(out_sim - out_ref))
        assert max_diff < 1e-5, f"RoPE half mode max diff {max_diff} exceeds 1e-5"

    def test_rope_interleaved_mode_parity(self):
        torch.manual_seed(888)
        batch = 2
        num_heads = 4
        seq_len = 16
        head_dim = 64

        x_pt = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
        out_sim = apply_rope(x_pt.numpy(), theta_base=10000.0, mode="interleaved")

        # PyTorch reference for interleaved mode
        dim = head_dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        sinusoid = torch.matmul(torch.arange(seq_len).float().unsqueeze(1), inv_freq.unsqueeze(0))
        sinusoid = torch.repeat_interleave(sinusoid, 2, dim=-1)
        cos = torch.cos(sinusoid).view(1, 1, seq_len, dim)
        sin = torch.sin(sinusoid).view(1, 1, seq_len, dim)

        x_even = x_pt[..., 0::2]
        x_odd = x_pt[..., 1::2]
        x_rot = torch.empty_like(x_pt)
        x_rot[..., 0::2] = -x_odd
        x_rot[..., 1::2] = x_even
        out_ref = (x_pt * cos + x_rot * sin).numpy()

        max_diff = np.max(np.abs(out_sim - out_ref))
        assert max_diff < 1e-5, f"RoPE interleaved mode max diff {max_diff} exceeds 1e-5"

    def test_rope_norm_preservation(self):
        # Rotation matrices are unitary; L2 norm must be strictly preserved
        x = np.random.randn(2, 4, 16, 64).astype(np.float32)
        out = apply_rope(x, mode="half")

        norm_in = np.linalg.norm(x, axis=-1)
        norm_out = np.linalg.norm(out, axis=-1)
        assert np.allclose(norm_in, norm_out, atol=1e-5)


class TestDSPTopK:
    """Differential verification of TopK against PyTorch torch.topk."""

    @pytest.mark.parametrize("k", [1, 5, 20])
    @pytest.mark.parametrize("largest", [True, False])
    def test_topk_parity(self, k: int, largest: bool):
        torch.manual_seed(777 + k)
        batch = 4
        vocab_size = 1000

        x_pt = torch.randn(batch, vocab_size, dtype=torch.float32)

        val_sim, idx_sim = topk(x_pt.numpy(), k=k, axis=-1, largest=largest, sorted=True)
        val_pt, idx_pt = torch.topk(x_pt, k=k, dim=-1, largest=largest, sorted=True)

        assert np.allclose(val_sim, val_pt.numpy(), atol=1e-6)
        assert np.array_equal(idx_sim, idx_pt.numpy())


class TestDSPExecutor:
    """Integration tests for DSPExecutor engine."""

    def test_executor_lifecycle_and_stats(self):
        vas = VirtualAddressSpace()
        dsp = DSPExecutor(address_space=vas)

        x = np.random.randn(2, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)

        out_rms = dsp.execute_rms_norm(x, w)
        assert out_rms.shape == (2, 64)

        out_sm = dsp.execute_softmax(x)
        assert out_sm.shape == (2, 64)

        x_rope = np.random.randn(1, 2, 8, 64).astype(np.float32)
        out_rope = dsp.execute_rope(x_rope)
        assert out_rope.shape == (1, 2, 8, 64)

        val, idx = dsp.execute_topk(x, k=5)
        assert val.shape == (2, 5)
        assert idx.shape == (2, 5)

        stats = dsp.stats()
        assert stats["total_dsp_ops"] == 4
        assert stats["rms_norm_count"] == 1
        assert stats["softmax_count"] == 1
        assert stats["rope_count"] == 1
        assert stats["topk_count"] == 1
        assert stats["total_operations"] > 0

        dsp.reset()
        reset_stats = dsp.stats()
        assert reset_stats["total_dsp_ops"] == 0
        assert reset_stats["rms_norm_count"] == 0
