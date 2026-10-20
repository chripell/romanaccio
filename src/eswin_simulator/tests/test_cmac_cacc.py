"""
Unit and differential verification tests for CMAC (Convolution MAC)
and CACC (Convolution Accumulator) engines against PyTorch and vendor EsSimulator.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.memory.address_space import VirtualAddressSpace
from eswin_simulator.engines.cmac import (
    CMACEngine,
    unpack_nvdla_weights_conv2d,
    unpack_nvdla_weights_1x1,
    unpack_nvdla_weights,
)
from eswin_simulator.engines.cacc import CACCEngine


# Path to local deep/artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parents[4] / "deep" / "artifacts"


@pytest.fixture
def vas() -> VirtualAddressSpace:
    return VirtualAddressSpace()


@pytest.fixture
def cmac(vas: VirtualAddressSpace) -> CMACEngine:
    return CMACEngine(address_space=vas)


@pytest.fixture
def cacc(vas: VirtualAddressSpace) -> CACCEngine:
    return CACCEngine(address_space=vas)


def test_cmac_conv2d_basic_fp16(cmac: CMACEngine, cacc: CACCEngine):
    """Verifies FP16 2D convolution against PyTorch golden output."""
    N, C_in, H, W = 2, 8, 16, 16
    C_out = 16
    KH, KW = 3, 3

    np.random.seed(42)
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)
    bias = np.random.randn(C_out).astype(np.float32)

    # PyTorch golden
    x_t = torch.from_numpy(x)
    w_t = torch.from_numpy(w)
    b_t = torch.from_numpy(bias)
    golden_t = F.conv2d(x_t, w_t, b_t, stride=1, padding=1)
    golden = golden_t.numpy()

    # Simulator execution
    acc = cmac.conv2d(x, w, stride=1, padding=1, precision="FP16")
    out = cacc.finalize_fp16(acc, bias=bias, to_fp16=False)

    diff = np.max(np.abs(out - golden))
    cos = np.dot(out.flatten(), golden.flatten()) / (
        np.linalg.norm(out.flatten()) * np.linalg.norm(golden.flatten())
    )

    assert cos >= 0.999999
    assert diff < 1e-4
    assert cmac.total_convolutions == 1
    assert cmac.total_macs == N * C_out * C_in * KH * KW * H * W


def test_cmac_conv2d_int8_bit_exact(cmac: CMACEngine):
    """Verifies signed INT8 convolution achieves exact bit-level parity with integer PyTorch."""
    N, C_in, H, W = 1, 4, 12, 12
    C_out = 8
    KH, KW = 3, 3

    np.random.seed(123)
    x = np.random.randint(-128, 127, size=(N, C_in, H, W), dtype=np.int8)
    w = np.random.randint(-128, 127, size=(C_out, C_in, KH, KW), dtype=np.int8)

    # PyTorch integer conv2d
    x_t = torch.from_numpy(x).to(torch.int32)
    w_t = torch.from_numpy(w).to(torch.int32)
    golden = F.conv2d(x_t, w_t, stride=(2, 2), padding=1).numpy()

    # Simulator execution
    out = cmac.conv2d(x, w, stride=2, padding=1, precision="INT8")

    assert np.array_equal(out, golden)
    assert out.dtype == np.int32


def test_cmac_conv2d_grouped_and_depthwise(cmac: CMACEngine):
    """Verifies grouped and depthwise convolutions."""
    # Depthwise: groups = C_in = C_out
    C = 8
    x = np.random.randn(1, C, 14, 14).astype(np.float32)
    w = np.random.randn(C, 1, 3, 3).astype(np.float32)

    golden = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), groups=C, padding=1).numpy()
    out = cmac.conv2d(x, w, groups=C, padding=1, precision="FP16")

    diff = np.max(np.abs(out - golden))
    assert diff < 1e-5


def test_cmac_gemm_1x1_and_shapes(cmac: CMACEngine):
    """Verifies 1x1 pointwise conv and linear GEMM across 2D, 3D, and 4D tensor inputs."""
    C_in, C_out = 64, 128

    w = np.random.randn(C_out, C_in).astype(np.float32)

    # 2D input (N, C_in)
    x2d = np.random.randn(4, C_in).astype(np.float32)
    golden2d = x2d @ w.T
    out2d = cmac.gemm_1x1(x2d, w, precision="FP16")
    assert np.max(np.abs(out2d - golden2d)) < 1e-5

    # 3D input (Batch, Seq, C_in)
    x3d = np.random.randn(2, 8, C_in).astype(np.float32)
    golden3d = x3d @ w.T
    out3d = cmac.gemm_1x1(x3d, w, precision="FP16")
    assert np.max(np.abs(out3d - golden3d)) < 1e-5

    # 4D input (N, C_in, 1, 1)
    x4d = np.random.randn(2, C_in, 1, 1).astype(np.float32)
    out4d = cmac.gemm_1x1(x4d, w, precision="FP16")
    assert out4d.shape == (2, C_out, 1, 1)
    assert np.max(np.abs(out4d.squeeze() - (x4d.squeeze() @ w.T))) < 1e-5


def test_cacc_int8_saturating_clip(cacc: CACCEngine):
    """Verifies CACC INT8 channel scaling and strict hardware saturating clipping."""
    # Accumulator with extreme values
    acc = np.array([[-1000, -200, -128, 0, 50, 127, 200, 1000]], dtype=np.int32)
    bias = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    scale = 1.0

    out = cacc.finalize_int8(acc, scale=scale, bias=bias, clip_min=-128, clip_max=127)

    assert out.dtype == np.int8
    expected = np.array([[-128, -128, -128, 0, 50, 127, 127, 127]], dtype=np.int8)
    assert np.array_equal(out, expected)
    assert cacc.total_saturations == 4  # -1000, -200, 200, 1000 saturated


def test_golden_model_conv2d_differential(cmac: CMACEngine, cacc: CACCEngine):
    """Differential verification of test_conv_npu_b1.model against vendor EsSimulator golden."""
    model_path = ARTIFACTS_DIR / "test_conv_npu_b1.model"
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    model = LoadableModel.from_file(str(model_path))

    # Input: all ones (1, 3, 32, 32)
    x = np.ones((1, 3, 32, 32), dtype=np.float16)

    # Unpack weights from blob W_0: C_out=16, C_in=3, KH=3, KW=3
    w_raw = model.blobs["W_0"].data
    w = unpack_nvdla_weights_conv2d(w_raw, c_out=16, c_in=3, kh=3, kw=3, dtype=np.float16)
    assert w.shape == (16, 3, 3, 3)

    # CMAC conv2d + CACC finalize
    acc = cmac.conv2d(x, w, stride=1, padding=0, precision="FP16")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    # Load golden output from EsSimulator verified execution
    golden_path = Path("/tmp/conv_out_B.bin")
    if not golden_path.exists():
        # Fall back to artifacts dir if available
        alt_path = ARTIFACTS_DIR / "golden_conv_fp16.bin"
        if alt_path.exists():
            golden_path = alt_path
        else:
            pytest.skip("Golden output file conv_out_B.bin not found")

    golden = np.fromfile(str(golden_path), dtype=np.float16).reshape(1, 16, 30, 30)

    cos = np.dot(y_sim.astype(np.float32).flatten(), golden.astype(np.float32).flatten()) / (
        np.linalg.norm(y_sim.astype(np.float32).flatten())
        * np.linalg.norm(golden.astype(np.float32).flatten())
    )
    rel_diff = np.max(
        np.abs(y_sim.astype(np.float32) - golden.astype(np.float32))
        / (np.abs(golden.astype(np.float32)) + 1e-7)
    )

    print(f"\n[Golden Conv2D] Cosine Similarity: {cos:.8f}, Max Rel Diff: {rel_diff:.8f}")
    assert cos >= 0.999999
    assert rel_diff < 1e-4


def test_golden_model_linear_1x1_differential(cmac: CMACEngine, cacc: CACCEngine):
    """Differential verification of test_linear_1x1conv_npu_b1.model against vendor EsSimulator golden."""
    model_path = ARTIFACTS_DIR / "test_linear_1x1conv_npu_b1.model"
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    model = LoadableModel.from_file(str(model_path))

    # Input: all ones (1, 576, 1, 1)
    x = np.ones((1, 576, 1, 1), dtype=np.float16)

    # Unpack 3 splits of weights (each 512 x 576 -> total 1536 x 576)
    w_splits = []
    for i in range(3):
        blob_name = f"WsplitM_{i}_{i}"
        w_split = unpack_nvdla_weights_1x1(model.blobs[blob_name].data, c_out=512, c_in=576, dtype=np.float16)
        w_splits.append(w_split)
    w_total = np.vstack(w_splits)
    assert w_total.shape == (1536, 576)

    # CMAC gemm_1x1 + CACC finalize
    acc = cmac.gemm_1x1(x, w_total, precision="FP16")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    # Load actual golden output dumped directly from EsSimulator execution
    golden_path = Path("/tmp/es_linear_actual.bin")
    if not golden_path.exists():
        pytest.skip("Golden output file es_linear_actual.bin not found")

    golden = np.fromfile(str(golden_path), dtype=np.float16).reshape(1, 1536, 1, 1)

    cos = np.dot(y_sim.astype(np.float32).flatten(), golden.astype(np.float32).flatten()) / (
        np.linalg.norm(y_sim.astype(np.float32).flatten())
        * np.linalg.norm(golden.astype(np.float32).flatten())
    )
    rel_diff = np.max(
        np.abs(y_sim.astype(np.float32) - golden.astype(np.float32))
        / (np.abs(golden.astype(np.float32)) + 1e-7)
    )

    print(f"\n[Golden Linear 1x1] Cosine Similarity: {cos:.8f}, Max Rel Diff: {rel_diff:.8f}")
    assert cos >= 0.999999
    assert rel_diff < 1e-3
