"""
Unit and differential tests for Planar Data Processor (PDPEngine) - Milestone 6.
Validates 2D Max Pooling, Average Pooling, and Global Spatial Pooling against PyTorch.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from eswin_simulator.engines.pdp import PDPEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


class TestPDPEngine:
    @pytest.fixture
    def pdp(self):
        vas = VirtualAddressSpace()
        return PDPEngine(address_space=vas)

    @pytest.mark.parametrize("kernel_size", [2, 3])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    def test_max_pool2d_parity(self, pdp, kernel_size, stride, padding):
        torch.manual_seed(42 + kernel_size + stride)
        n, c, h, w = 2, 4, 16, 16
        x_pt = torch.randn(n, c, h, w, dtype=torch.float32)

        out_sim = pdp.max_pool2d(
            x_pt.numpy(),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        out_pt = F.max_pool2d(
            x_pt,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).numpy()

        assert out_sim.shape == out_pt.shape
        assert np.allclose(out_sim, out_pt, atol=1e-5)

    @pytest.mark.parametrize("kernel_size", [2, 3])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_avg_pool2d_parity(self, pdp, kernel_size, stride):
        torch.manual_seed(100 + kernel_size)
        n, c, h, w = 2, 4, 16, 16
        x_pt = torch.randn(n, c, h, w, dtype=torch.float32)

        out_sim = pdp.avg_pool2d(
            x_pt.numpy(),
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        out_pt = F.avg_pool2d(
            x_pt,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        ).numpy()

        assert out_sim.shape == out_pt.shape
        assert np.allclose(out_sim, out_pt, atol=1e-5)

    def test_max_pool2d_int8_saturating(self, pdp):
        np.random.seed(42)
        x_int8 = np.random.randint(-128, 127, size=(1, 8, 14, 14), dtype=np.int8)

        out_sim = pdp.max_pool2d(x_int8, kernel_size=2, stride=2, padding=0)
        assert out_sim.dtype == np.int8
        assert out_sim.shape == (1, 8, 7, 7)

        # PyTorch reference via float
        pt_ref = F.max_pool2d(torch.from_numpy(x_int8.astype(np.float32)), 2, 2)
        assert np.array_equal(out_sim, pt_ref.numpy().astype(np.int8))

    def test_avg_pool2d_int8(self, pdp):
        np.random.seed(99)
        x_int8 = np.random.randint(-100, 100, size=(1, 4, 8, 8), dtype=np.int8)

        out_sim = pdp.avg_pool2d(x_int8, kernel_size=2, stride=2, padding=0)
        assert out_sim.dtype == np.int8
        assert out_sim.shape == (1, 4, 4, 4)
        assert np.all(out_sim >= -128) and np.all(out_sim <= 127)

    def test_global_avg_pool2d(self, pdp):
        x = np.random.randn(2, 16, 8, 8).astype(np.float32)
        out = pdp.global_avg_pool2d(x)
        assert out.shape == (2, 16, 1, 1)

        ref = np.mean(x, axis=(2, 3), keepdims=True)
        assert np.allclose(out, ref, atol=1e-6)

    def test_global_max_pool2d(self, pdp):
        x = np.random.randn(2, 16, 8, 8).astype(np.float32)
        out = pdp.global_max_pool2d(x)
        assert out.shape == (2, 16, 1, 1)

        ref = np.max(x, axis=(2, 3), keepdims=True)
        assert np.allclose(out, ref, atol=1e-6)

    def test_pdp_stats_and_reset(self, pdp):
        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        pdp.max_pool2d(x, kernel_size=2)
        pdp.avg_pool2d(x, kernel_size=2)
        pdp.global_avg_pool2d(x)

        stats = pdp.stats()
        assert stats["total_pools"] == 3
        assert stats["max_pool_count"] == 1
        assert stats["avg_pool_count"] == 2
        assert stats["total_operations"] > 0

        pdp.reset()
        assert pdp.stats()["total_pools"] == 0
