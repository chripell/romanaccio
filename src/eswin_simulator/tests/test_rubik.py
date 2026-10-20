"""
Unit tests for Rubik Data Transformation Engine (RubikEngine) - Milestone 6.
Validates Reshape, Transpose, Slice, Split, Concat, Squeeze, Flatten, and Depth/Space.
"""

import pytest
import numpy as np

from eswin_simulator.engines.rubik import RubikEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


class TestRubikEngine:
    @pytest.fixture
    def rubik(self):
        vas = VirtualAddressSpace()
        return RubikEngine(address_space=vas)

    def test_reshape(self, rubik):
        x = np.arange(24).reshape(2, 3, 4)
        out = rubik.reshape(x, (2, 12))
        assert out.shape == (2, 12)
        assert out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out.flatten(), x.flatten())

    def test_transpose(self, rubik):
        x = np.random.randn(2, 8, 16, 64).astype(np.float32)
        # Multi-head attention transpose: (B, S, H, D) -> (B, H, S, D)
        out = rubik.transpose(x, (0, 2, 1, 3))
        assert out.shape == (2, 16, 8, 64)
        assert out.flags["C_CONTIGUOUS"]
        assert np.allclose(out, np.transpose(x, (0, 2, 1, 3)))

    def test_slice(self, rubik):
        x = np.arange(60).reshape(3, 4, 5)
        # Slice axis 1 from 1 to 3, axis 2 from 0 to 4 step 2
        out = rubik.slice(x, starts=[1, 0], ends=[3, 4], axes=[1, 2], steps=[1, 2])
        assert out.shape == (3, 2, 2)
        assert out.flags["C_CONTIGUOUS"]
        ref = x[:, 1:3, 0:4:2]
        assert np.array_equal(out, ref)

    def test_concat(self, rubik):
        t1 = np.ones((2, 3, 4), dtype=np.float32)
        t2 = np.zeros((2, 5, 4), dtype=np.float32)
        out = rubik.concat([t1, t2], axis=1)
        assert out.shape == (2, 8, 4)
        assert out.flags["C_CONTIGUOUS"]
        assert np.all(out[:, :3, :] == 1.0)
        assert np.all(out[:, 3:, :] == 0.0)

    def test_split(self, rubik):
        x = np.random.randn(2, 960, 1, 1).astype(np.float32)
        # Split into Q (576), K (192), V (192)
        splits = rubik.split(x, [576, 192, 192], axis=1)
        assert len(splits) == 3
        assert splits[0].shape == (2, 576, 1, 1)
        assert splits[1].shape == (2, 192, 1, 1)
        assert splits[2].shape == (2, 192, 1, 1)
        assert np.array_equal(np.concatenate(splits, axis=1), x)

    def test_squeeze_unsqueeze(self, rubik):
        x = np.zeros((1, 10, 1, 20))
        sq = rubik.squeeze(x)
        assert sq.shape == (10, 20)
        un = rubik.unsqueeze(sq, axis=0)
        assert un.shape == (1, 10, 20)

    def test_flatten(self, rubik):
        x = np.zeros((2, 3, 4, 5))
        f = rubik.flatten(x, start_dim=1, end_dim=2)
        assert f.shape == (2, 12, 5)

    def test_space_to_depth_and_depth_to_space(self, rubik):
        # (N, C, H, W)
        x = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        s2d = rubik.space_to_depth(x, block_size=2)
        assert s2d.shape == (1, 8, 2, 2)

        # Invert via depth_to_space
        d2s = rubik.depth_to_space(s2d, block_size=2)
        assert d2s.shape == x.shape
        assert np.array_equal(d2s, x)

    def test_rubik_stats_and_reset(self, rubik):
        x = np.zeros((2, 4, 8), dtype=np.float32)
        rubik.reshape(x, (2, 32))
        rubik.transpose(x, (0, 2, 1))

        stats = rubik.stats()
        assert stats["total_reorders"] == 2
        assert stats["reshape_count"] == 1
        assert stats["transpose_count"] == 1
        assert stats["bytes_reordered"] > 0

        rubik.reset()
        assert rubik.stats()["total_reorders"] == 0
