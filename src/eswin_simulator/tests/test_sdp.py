"""
Unit and differential verification tests for the SDP (Single Data Point) Engine,
including non-linear piecewise LUT interpolation and end-to-end SwiGLU MLP verification.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.memory.address_space import VirtualAddressSpace
from eswin_simulator.engines.cmac import CMACEngine, unpack_nvdla_weights_1x1
from eswin_simulator.engines.cacc import CACCEngine
from eswin_simulator.engines.sdp import SDPEngine, SDPLookupTable


ARTIFACTS_DIR = Path(__file__).resolve().parents[4] / "deep" / "artifacts"


@pytest.fixture
def vas() -> VirtualAddressSpace:
    return VirtualAddressSpace()


@pytest.fixture
def sdp(vas: VirtualAddressSpace) -> SDPEngine:
    return SDPEngine(address_space=vas)


def test_sdp_bias_add(sdp: SDPEngine):
    """Verifies per-channel bias addition with multi-dimensional broadcasting."""
    # 4D NCHW
    x4d = np.random.randn(2, 16, 8, 8).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    out4d = sdp.bias_add(x4d, bias)
    golden4d = x4d + bias.reshape(1, 16, 1, 1)
    assert np.allclose(out4d, golden4d)

    # 2D (N, C)
    x2d = np.random.randn(4, 32).astype(np.float32)
    bias2d = np.random.randn(32).astype(np.float32)
    out2d = sdp.bias_add(x2d, bias2d)
    assert np.allclose(out2d, x2d + bias2d)


def test_sdp_batch_norm(sdp: SDPEngine):
    """Verifies batch normalization against PyTorch golden."""
    x = np.random.randn(2, 8, 10, 10).astype(np.float32)
    mean = np.random.randn(8).astype(np.float32)
    var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1
    gamma = np.random.randn(8).astype(np.float32)
    beta = np.random.randn(8).astype(np.float32)
    eps = 1e-5

    out = sdp.batch_norm(x, mean, var, gamma=gamma, beta=beta, eps=eps)

    golden = F.batch_norm(
        torch.from_numpy(x),
        torch.from_numpy(mean),
        torch.from_numpy(var),
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        training=False,
        eps=eps,
    ).numpy()

    assert np.allclose(out, golden, atol=1e-5)


def test_sdp_fixed_activations(sdp: SDPEngine):
    """Verifies fixed-function activations: ReLU, LeakyReLU, PReLU, and Clip."""
    x = np.linspace(-5.0, 5.0, 50, dtype=np.float32)

    # ReLU
    out_relu = sdp.relu(x)
    assert np.allclose(out_relu, F.relu(torch.from_numpy(x)).numpy())

    # LeakyReLU
    out_leaky = sdp.leaky_relu(x, negative_slope=0.1)
    assert np.allclose(out_leaky, F.leaky_relu(torch.from_numpy(x), negative_slope=0.1).numpy())

    # PReLU (channel-wise)
    x4d = np.random.randn(1, 4, 8, 8).astype(np.float32)
    weight = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out_prelu = sdp.prelu(x4d, weight)
    golden_prelu = F.prelu(torch.from_numpy(x4d), torch.from_numpy(weight)).numpy()
    assert np.allclose(out_prelu, golden_prelu)

    # Clip
    out_clip = sdp.clip(x, -2.0, 3.0)
    assert np.min(out_clip) == -2.0
    assert np.max(out_clip) == 3.0


def test_sdp_eltwise_operations(sdp: SDPEngine):
    """Verifies elementwise arithmetic and comparison operations."""
    a = np.random.randn(4, 16).astype(np.float32)
    b = np.random.randn(4, 16).astype(np.float32)

    assert np.allclose(sdp.eltwise_add(a, b), a + b)
    assert np.allclose(sdp.eltwise_sub(a, b), a - b)
    assert np.allclose(sdp.eltwise_mul(a, b), a * b)
    assert np.allclose(sdp.eltwise_max(a, b), np.maximum(a, b))
    assert np.allclose(sdp.eltwise_min(a, b), np.minimum(a, b))


def test_sdp_lut_piecewise_interpolation(sdp: SDPEngine):
    """Verifies 257-entry piecewise linear LUT interpolation for SiLU and Sigmoid."""
    # Test SiLU LUT
    lut_silu = SDPLookupTable.create_silu_lut()
    x = np.linspace(-6.0, 6.0, 200, dtype=np.float32)
    y_lut = sdp.interpolate_lut(x, lut_silu, table_type="LO", activation_type="silu")
    y_golden = (x / (1.0 + np.exp(-x))).astype(np.float32)

    max_diff = np.max(np.abs(y_lut - y_golden))
    assert max_diff < 0.005  # Within piecewise linear tolerance

    # Test Sigmoid LUT
    lut_sigmoid = SDPLookupTable.create_sigmoid_lut()
    y_sig_lut = sdp.interpolate_lut(x, lut_sigmoid, table_type="LO", activation_type="sigmoid")
    y_sig_golden = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    assert np.max(np.abs(y_sig_lut - y_sig_golden)) < 0.005


def test_sdp_precision_conversion(sdp: SDPEngine):
    """Verifies FP16 <-> INT8 precision conversion with scaling and clamping."""
    # Float -> INT8
    x_fp32 = np.array([-150.0, -100.0, 0.0, 50.0, 150.0], dtype=np.float32)
    int8_res = sdp.convert_precision(x_fp32, "float32", "int8", scale=1.0)
    assert int8_res.dtype == np.int8
    assert int8_res[0] == -128  # Clamped
    assert int8_res[-1] == 127  # Clamped

    # INT8 -> Float
    x_int8 = np.array([-10, 0, 10], dtype=np.int8)
    fp16_res = sdp.convert_precision(x_int8, "int8", "float16", scale=0.5)
    assert fp16_res.dtype == np.float16
    assert np.allclose(fp16_res, np.array([-5.0, 0.0, 5.0], dtype=np.float16))


def test_golden_swiglu_mlp_differential(vas: VirtualAddressSpace):
    """Differential verification of test_swiglu_mlp_npu_b1.model against vendor EsSimulator golden."""
    model_path = ARTIFACTS_DIR / "test_swiglu_mlp_npu_b1.model"
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    golden_path = Path("/tmp/es_swiglu_golden.bin")
    if not golden_path.exists():
        golden_path = ARTIFACTS_DIR / "scratch" / "golden_swiglu" / "es_swiglu_golden.bin"
    if not golden_path.exists():
        pytest.skip("Golden file es_swiglu_golden.bin not found")

    model = LoadableModel.from_file(str(model_path))
    cmac = CMACEngine(address_space=vas)
    cacc = CACCEngine(address_space=vas)
    sdp = SDPEngine(address_space=vas)

    in_t = model.input_tensors[0]
    out_t = model.output_tensors[0]
    c_in = in_t.c
    hidden_dim = 1536

    # 1. Unpack weights
    w_gate = np.vstack([
        unpack_nvdla_weights_1x1(model.blobs[f"w_gatesplitM_{i}_{i}"].data, 512, c_in, dtype=in_t.dtype)
        for i in range(3)
    ])
    w_up = np.vstack([
        unpack_nvdla_weights_1x1(model.blobs[f"w_upsplitM_{i}_{i+3}"].data, 512, c_in, dtype=in_t.dtype)
        for i in range(3)
    ])
    w_down = np.vstack([
        unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_0_6"].data, 256, hidden_dim, dtype=in_t.dtype),
        unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_1_7"].data, 256, hidden_dim, dtype=in_t.dtype),
        unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_2_8"].data, 64, hidden_dim, dtype=in_t.dtype),
    ])

    # 2. Input: all ones (1, 576, 1, 1)
    x = np.ones((1, 576, 1, 1), dtype=np.float16)

    # 3. Gate & Up projections (GEMM)
    gate_act = cmac.gemm_1x1(x, w_gate, precision="FP16")
    up_act = cmac.gemm_1x1(x, w_up, precision="FP16")

    # 4. SiLU Activation via SDP LUT
    lut_blob = model.blobs.get("kmd-task-lut_list")
    assert lut_blob is not None
    lut_desc = SDPLookupTable.from_bytes(lut_blob.data)
    silu_act = sdp.silu(gate_act, lut=lut_desc, use_lut=True)

    # 5. Elementwise Multiplication: SiLU(gate) * up
    inter_act = sdp.eltwise_mul(silu_act, up_act)

    # 6. Down projection (GEMM)
    down_act = cmac.gemm_1x1(inter_act, w_down, precision="FP16")
    y_sim = cacc.finalize_fp16(down_act, to_fp16=True)

    # 7. Compare with EsSimulator golden output
    golden = np.fromfile(str(golden_path), dtype=np.float16).reshape(1, 576, 1, 1)

    y_f32 = y_sim.astype(np.float32).flatten()
    g_f32 = golden.astype(np.float32).flatten()
    cos = np.dot(y_f32, g_f32) / (np.linalg.norm(y_f32) * np.linalg.norm(g_f32))
    max_abs_diff = np.max(np.abs(y_f32 - g_f32))

    print(f"\n[Golden SwiGLU MLP] Cosine Similarity: {cos:.8f}, Max Abs Diff: {max_abs_diff:.8f}")
    assert cos >= 0.999999
    assert max_abs_diff < 0.001
