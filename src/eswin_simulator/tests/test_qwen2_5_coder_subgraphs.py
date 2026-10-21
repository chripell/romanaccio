"""
test_qwen2_5_coder_subgraphs.py - Differential verification of compiled Qwen2.5-Coder-1.5B NPU loadables.

Verifies:
1. qwen2_5_layer0_qkv_npu_b1.model (Fused QKV 1536 in -> 2048 out)
2. qwen2_5_layer0_attn_out_npu_b1.model (Attn Out 1536 in -> 1536 out)
3. qwen2_5_mlp_opset13_npu_b1.model (SwiGLU MLP 1536 in -> 8960 hidden -> 1536 out)
4. qwen2_5_lm_head_d0_npu_b1.model (Die 0 LM Head 1536 in -> 76032 out)
5. qwen2_5_lm_head_d1_npu_b1.model (Die 1 LM Head 1536 in -> 76032 out)
"""

from pathlib import Path
import numpy as np
import pytest

from eswin_simulator.loader import LoadableModel
from eswin_simulator.memory.address_space import VirtualAddressSpace
from eswin_simulator.engines.cmac import CMACEngine, unpack_nvdla_weights_1x1
from eswin_simulator.engines.cacc import CACCEngine
from eswin_simulator.engines.sdp import SDPEngine, SDPLookupTable
from eswin_simulator.evaluation.comparator import TensorComparator

MODELS_DIR = Path("/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b/compiled_npu")
IPC_DIR = Path("/run/host-wayland/simulator-ipc")


@pytest.fixture
def test_input():
    in_file = IPC_DIR / "qkv_input.bin"
    if not in_file.exists():
        pytest.skip("Test input vector not found")
    return np.fromfile(str(in_file), dtype=np.float16).reshape(1, 1536, 1, 1)


def test_qwen2_5_qkv_npu_subgraph(test_input):
    model_file = MODELS_DIR / "qwen2_5_layer0_qkv_npu_b1.model"
    gold_file = IPC_DIR / "qkv_golden.bin"
    if not (model_file.exists() and gold_file.exists()):
        pytest.skip("QKV model or golden artifact missing")

    model = LoadableModel.from_file(str(model_file))
    vas = VirtualAddressSpace(model)
    cmac = CMACEngine(vas)
    cacc = CACCEngine(vas)

    c_in = 1536
    c_out = 2048

    split_idx = 0
    w_splits = []
    while f"weightsplitM_{split_idx}_{split_idx}" in model.blobs:
        raw = model.blobs[f"weightsplitM_{split_idx}_{split_idx}"].data
        c_out_split = len(raw) // (c_in * 2)
        w_s = unpack_nvdla_weights_1x1(raw, c_out=c_out_split, c_in=c_in, dtype=np.float16)
        w_splits.append(w_s)
        split_idx += 1

    w_total = np.vstack(w_splits)[:c_out, :]
    acc = cmac.gemm_1x1(test_input, w_total, precision="HALF")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    golden = np.fromfile(str(gold_file), dtype=np.float16).reshape(1, c_out, 1, 1)
    rep = TensorComparator.compare(y_sim, golden, tolerance=0.05, cos_threshold=0.999)

    assert rep.passed
    assert rep.cosine_similarity > 0.99999


def test_qwen2_5_attn_out_npu_subgraph(test_input):
    model_file = MODELS_DIR / "qwen2_5_layer0_attn_out_npu_b1.model"
    gold_file = IPC_DIR / "attn_out_golden.bin"
    if not (model_file.exists() and gold_file.exists()):
        pytest.skip("Attn Out model or golden artifact missing")

    model = LoadableModel.from_file(str(model_file))
    vas = VirtualAddressSpace(model)
    cmac = CMACEngine(vas)
    cacc = CACCEngine(vas)

    c_in = 1536
    c_out = 1536

    split_idx = 0
    w_splits = []
    while f"weightsplitM_{split_idx}_{split_idx}" in model.blobs:
        raw = model.blobs[f"weightsplitM_{split_idx}_{split_idx}"].data
        c_out_split = len(raw) // (c_in * 2)
        w_s = unpack_nvdla_weights_1x1(raw, c_out=c_out_split, c_in=c_in, dtype=np.float16)
        w_splits.append(w_s)
        split_idx += 1

    w_total = np.vstack(w_splits)[:c_out, :]
    acc = cmac.gemm_1x1(test_input, w_total, precision="HALF")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    golden = np.fromfile(str(gold_file), dtype=np.float16).reshape(1, c_out, 1, 1)
    rep = TensorComparator.compare(y_sim, golden, tolerance=0.05, cos_threshold=0.999)

    assert rep.passed
    assert rep.cosine_similarity > 0.99999


def test_qwen2_5_mlp_npu_subgraph():
    model_file = MODELS_DIR / "qwen2_5_mlp_opset13_npu_b1.model"
    in_file = IPC_DIR / "qwen_mlp_input.bin"
    gold_file = IPC_DIR / "qwen_mlp_golden.bin"
    if not (model_file.exists() and in_file.exists() and gold_file.exists()):
        pytest.skip("Full MLP model or golden artifacts missing")

    model = LoadableModel.from_file(str(model_file))
    vas = VirtualAddressSpace(model)
    cmac = CMACEngine(vas)
    cacc = CACCEngine(vas)
    sdp = SDPEngine(vas)

    c_in = 1536
    c_out = 1536

    # Gate splits
    gate_blobs = sorted([b for b in model.blobs if b.startswith("w_gatesplitM_")], key=lambda x: int(x.split("_")[2]))
    w_gate = np.vstack([unpack_nvdla_weights_1x1(model.blobs[b].data, len(model.blobs[b].data) // (c_in * 2), c_in, dtype=np.float16) for b in gate_blobs])
    hidden_dim = w_gate.shape[0]

    # Up splits
    up_blobs = sorted([b for b in model.blobs if b.startswith("w_upsplitM_")], key=lambda x: int(x.split("_")[2]))
    w_up = np.vstack([unpack_nvdla_weights_1x1(model.blobs[b].data, len(model.blobs[b].data) // (c_in * 2), c_in, dtype=np.float16) for b in up_blobs])

    # Down splits
    down_blobs = sorted([b for b in model.blobs if b.startswith("w_downsplitM_")], key=lambda x: int(x.split("_")[2]))
    total_down_bytes = sum(len(model.blobs[b].data) for b in down_blobs)
    w_down = np.vstack([
        unpack_nvdla_weights_1x1(model.blobs[b].data, int(round(c_out * (len(model.blobs[b].data) / total_down_bytes))), hidden_dim, dtype=np.float16)
        for b in down_blobs
    ])[:c_out, :]

    x = np.fromfile(str(in_file), dtype=np.float16).reshape(1, c_in, 1, 1)
    golden = np.fromfile(str(gold_file), dtype=np.float16).reshape(1, c_out, 1, 1)

    gate_act = cmac.gemm_1x1(x, w_gate, precision="HALF")
    up_act = cmac.gemm_1x1(x, w_up, precision="HALF")

    lut_blob = model.blobs.get("kmd-task-lut_list")
    lut_desc = SDPLookupTable.from_bytes(lut_blob.data) if lut_blob else None
    silu_act = sdp.silu(gate_act, lut=lut_desc, use_lut=(lut_desc is not None))

    inter_act = sdp.eltwise_mul(silu_act, up_act)
    down_act = cmac.gemm_1x1(inter_act, w_down, precision="HALF")
    y_sim = cacc.finalize_fp16(down_act, to_fp16=True)

    rep = TensorComparator.compare(y_sim, golden, tolerance=0.05, cos_threshold=0.999)
    assert rep.passed
    assert rep.cosine_similarity > 0.99999


def test_qwen2_5_lm_head_d0_npu_subgraph(test_input):
    model_file = MODELS_DIR / "qwen2_5_lm_head_d0_npu_b1.model"
    gold_file = IPC_DIR / "lm_head_d0_golden.bin"
    if not (model_file.exists() and gold_file.exists()):
        pytest.skip("LM Head d0 model or golden artifact missing")

    model = LoadableModel.from_file(str(model_file))
    vas = VirtualAddressSpace(model)
    cmac = CMACEngine(vas)
    cacc = CACCEngine(vas)

    c_in = 1536
    c_out = 76032

    split_idx = 0
    w_splits = []
    while f"weightsplitM_{split_idx}_{split_idx}" in model.blobs:
        raw = model.blobs[f"weightsplitM_{split_idx}_{split_idx}"].data
        c_out_split = len(raw) // (c_in * 2)
        w_s = unpack_nvdla_weights_1x1(raw, c_out=c_out_split, c_in=c_in, dtype=np.float16)
        w_splits.append(w_s)
        split_idx += 1

    w_total = np.vstack(w_splits)[:c_out, :]
    acc = cmac.gemm_1x1(test_input, w_total, precision="HALF")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    golden = np.fromfile(str(gold_file), dtype=np.float16).reshape(1, c_out, 1, 1)
    rep = TensorComparator.compare(y_sim, golden, tolerance=0.05, cos_threshold=0.999)

    assert rep.passed
    assert rep.cosine_similarity > 0.99999


def test_qwen2_5_lm_head_d1_npu_subgraph(test_input):
    model_file = MODELS_DIR / "qwen2_5_lm_head_d1_npu_b1.model"
    gold_file = IPC_DIR / "lm_head_d1_golden.bin"
    if not (model_file.exists() and gold_file.exists()):
        pytest.skip("LM Head d1 model or golden artifact missing")

    model = LoadableModel.from_file(str(model_file))
    vas = VirtualAddressSpace(model)
    cmac = CMACEngine(vas)
    cacc = CACCEngine(vas)

    c_in = 1536
    c_out = 76032

    split_idx = 0
    w_splits = []
    while f"weightsplitM_{split_idx}_{split_idx}" in model.blobs:
        raw = model.blobs[f"weightsplitM_{split_idx}_{split_idx}"].data
        c_out_split = len(raw) // (c_in * 2)
        w_s = unpack_nvdla_weights_1x1(raw, c_out=c_out_split, c_in=c_in, dtype=np.float16)
        w_splits.append(w_s)
        split_idx += 1

    w_total = np.vstack(w_splits)[:c_out, :]
    acc = cmac.gemm_1x1(test_input, w_total, precision="HALF")
    y_sim = cacc.finalize_fp16(acc, to_fp16=True)

    golden = np.fromfile(str(gold_file), dtype=np.float16).reshape(1, c_out, 1, 1)
    rep = TensorComparator.compare(y_sim, golden, tolerance=0.05, cos_threshold=0.999)

    assert rep.passed
    assert rep.cosine_similarity > 0.99999
