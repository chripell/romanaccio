#!/usr/bin/env python3
"""
Milestone 15 (Phase B): Direct NVDLA INT8/INT16 Weight & CvtPrecision_QuantM Package Builder
for LiquidAI/LFM2.5-8B-A1B Sparse MoE (8.47B Total, ~1.42B Active).

Synthesizes in /home/chri/aixigua/deep/models/LFM2.5-8B-A1B/eswin_pkg_int8/:
  - 9 Operator Class B Archetype Loadables:
      lfm2_{conv_in,proj2048,attn_qkv}_t{1,8,64}_classb.model
  - 3 Expert MLP Class B Archetype Loadables (2048 -> 1792 -> 2048 SwiGLU):
      lfm2_expert_mlp_t{1,8,64}_classb.model
  - 8 x 16,000-channel INT8 LM Head Static Class A Loadables:
      lfm2_lm_head_d{0,1}_s{0..3}.model
  - Dual-Die NPU Weight Blobs (~5.04 GiB/die):
      weights_die0.bin (Layers 0..11 operators + Even experts/blocks e % 2 == 0 across Layers 0..23)
      weights_die1.bin (Layers 12..23 operators + Odd experts/blocks e % 2 == 1 across Layers 0..23)
  - CPU Weights & Embedding:
      embedding.bin ([128000, 2048] FP16, 500 MiB)
      cpu_weights.bin + cpu_weights.json (RMSNorms, K=3 depthwise convs, Q/K norms, MoE Top-4 routers)
  - Quantization Metadata:
      quant_chan_scales.bin + quant_scales.json + weights_map.json + manifest.json + tokenizer.json
  - Golden Verification Oracle:
      /home/chri/aixigua/deep/artifacts/lfm2_moe_golden/golden_ref.bin
"""

import os
import sys
import time
import json
import struct
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))
from backend.flatbuffer_emitter import ModelReader, ModelWriter
from backend.dynamic_weight_binding import find_weight_regions

MODEL_DIR       = Path("/home/chri/aixigua/deep/models/LFM2.5-8B-A1B")
PKG_OUT_DIR     = MODEL_DIR / "eswin_pkg_int8"
LFM3_TEMPL_DIR  = Path("/home/chri/aixigua/deep/artifacts/lfm3_probe_templates")
LFM2_TEMPL_DIR  = Path("/home/chri/aixigua/deep/artifacts/lfm2_moe_templates")
LFM3_PKG_DIR    = Path("/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg_int8")
GOLDEN_DIR      = Path("/home/chri/aixigua/deep/artifacts/lfm2_moe_golden")
CALIB_PT_PATH   = GOLDEN_DIR / "lfm2_calib_fp32.pt"
COMPILER_PY     = Path(__file__).resolve().parent.parent / "eswin_compiler" / "compiler.py"
VENV_PY         = "/home/chri/aixigua/venv/bin/python3"

SDP_SIG = bytes.fromhex("000000000000010001010000")
ATTN_LAYERS = {2, 6, 10, 14, 18, 21}

TRUNCS = {
    "conv_in":    [14] * 12,
    "proj2048":   [13] * 4,
    "attn_qkv":   [14] * 6,
    "expert_mlp": [16] * 7 + [15] * 7 + [27] * 8,
    "lm_head":    [16] * 63,
}

LM_TMPL_PATH = LFM2_TEMPL_DIR / "ops_int16" / "lm_head_16k_int16_npu_b1.model"

OP_TMPL_MAP = {
    "conv_in_t1":   LFM2_TEMPL_DIR / "ops_int16" / "conv_in_t1_int16_npu_b1.model",
    "conv_in_t8":   LFM2_TEMPL_DIR / "ops_int16" / "conv_in_t8_int16_npu_b1.model",
    "conv_in_t64":  LFM2_TEMPL_DIR / "ops_int16" / "conv_in_t64_int16_npu_b1.model",
    "proj2048_t1":  LFM2_TEMPL_DIR / "ops_int16" / "proj2048_t1_int16_npu_b1.model",
    "proj2048_t8":  LFM2_TEMPL_DIR / "ops_int16" / "proj2048_t8_int16_npu_b1.model",
    "proj2048_t64": LFM2_TEMPL_DIR / "ops_int16" / "proj2048_t64_int16_npu_b1.model",
    "attn_qkv_t1":  LFM2_TEMPL_DIR / "ops_int16" / "attn_qkv_t1_int16_npu_b1.model",
    "attn_qkv_t8":  LFM2_TEMPL_DIR / "ops_int16" / "attn_qkv_t8_int16_npu_b1.model",
    "attn_qkv_t64": LFM2_TEMPL_DIR / "ops_int16" / "attn_qkv_t64_int16_npu_b1.model",
}

EXP_TMPL_MAP = {
    "expert_mlp_t1":  LFM2_TEMPL_DIR / "expert_mlp_t1_npu_b1.model",
    "expert_mlp_t8":  LFM2_TEMPL_DIR / "expert_mlp_t8_npu_b1.model",
    "expert_mlp_t64": LFM2_TEMPL_DIR / "expert_mlp_t64_npu_b1.model",
}

CALIB_TEXTS = [
    # Diverse prose
    "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture.",
    "Liquid Foundation Models combine gated short convolutions with grouped-query attention and sparse Mixture of Experts routing.",
    "The RISC-V architecture provides an open, extensible instruction set for modern embedded and dual-die neural processing accelerators.",
    "In quantum mechanics, the wave function describes the quantum state of an isolated system of particles and evolves according to the Schrodinger equation.",
    "Operating system kernels manage virtual memory pages using hardware page tables, TLBs, and NUMA-aware allocators across DDR5 channels.",
    "Symmetric integer quantization maps floating-point weights and activations into compact 8-bit and 16-bit integer representations.",
    "Photosynthesis converts light energy into chemical energy stored in carbohydrate molecules within plant chloroplasts.",
    "Distributed consensus algorithms such as Raft replicate state machines safely across unreliable network links.",
    # ChatML Python / coding turns
    "<|im_start|>user\nCan you write a Python function that finds all prime numbers up to n?<|im_end|>\n<|im_start|>assistant\n<think>\nTo find primes up to n, use the Sieve of Eratosthenes.\n</think>\n\n```python\ndef sieve(n: int) -> list[int]:\n    if n < 2:\n        return []\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for p in range(2, int(n ** 0.5) + 1):\n        if is_prime[p]:\n            for m in range(p * p, n + 1, p):\n                is_prime[m] = False\n    return [i for i, ok in enumerate(is_prime) if ok]\n```\n<|im_end|>\n",
    "<|im_start|>user\nWrite a Python function to merge overlapping intervals.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```\n<|im_end|>\n",
    "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nFix the off-by-one bug in this C loop:\n```c\nfor (int i = 0; i <= n; i++) sum += arr[i];\n```\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\nChange `i <= n` to `i < n` so `arr[n]` is not read out of bounds.\n<|im_end|>\n",
    "<|im_start|>user\nExplain the difference between a mutex and a semaphore in C++17.<|im_end|>\n<|im_start|>assistant\n<think>\nA mutex provides mutual exclusion with ownership, whereas a semaphore counts available resources.\n</think>\n\n1. **Ownership**: A `std::mutex` must be unlocked by the same thread that locked it.\n2. **Counting**: A semaphore allows up to `N` concurrent threads.\n<|im_end|>\n",
    # Reasoning & mathematics
    "<|im_start|>user\nCompute the binary search complexity and write an iterative version in Python.<|im_end|>\n<|im_start|>assistant\n<think>\nHalving the search range each step gives O(log n) time and O(1) space.\n</think>\n\n```python\ndef binary_search(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n<|im_end|>\n",
    # Additional prose without BOS
    "When climbing high alpine ridges, mountaineers monitor barometric pressure, wind shear, and freezing levels carefully before committing to an exposed traverse.",
    "Direct memory access controllers transfer large contiguous buffers between device memory and host DRAM without stalling scalar CPU pipelines.",
    "Fourier transforms decompose time-domain waveforms into orthogonal sinusoidal frequency bins with fast O(N log N) butterfly stages.",
    "Hello, world! Today is Monday, and we are benchmarking sparse Mixture of Experts throughput across both dies.",
    "In Python, dictionaries preserve insertion order and support average O(1) hash table lookups for arbitrary hashable keys.",
    "<|im_start|>user\nSummarize the three laws of thermodynamics.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n1. Energy is conserved.\n2. Entropy of an isolated system never decreases.\n3. Entropy approaches a constant as temperature approaches absolute zero.\n<|im_end|>\n",
    "Hardware cache coherence protocols ensure all cores observe a consistent view of shared memory lines across NUMA sockets.",
]


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([-x[..., 32:], x[..., :32]], dim=-1)


def patch_model_truncates(src_model: Path, dst_model: Path, new_truncs: list[int]):
    m = ModelReader.read_file(str(src_model))
    for b in m.blobs:
        if b.name == "kmd-task-op_list":
            op_bytes = bytearray(b.data)
            n_recs = len(op_bytes) // 1984
            hits = [r * 1984 + 82 for r in range(n_recs) if op_bytes[r*1984+82-12:r*1984+82] == SDP_SIG]
            assert len(hits) == len(new_truncs), f"Expected {len(new_truncs)} SDP hits in {src_model.name}, got {len(hits)}"
            for off, val in zip(hits, new_truncs):
                op_bytes[off] = val
            b.data = bytes(op_bytes)
    ModelWriter.write_file(m, str(dst_model))


def build_single_conv_region_dyntok(
    w_t: torch.Tensor,
    X_cal: torch.Tensor,
    sh: int,
    reg,
    margin: float = 2.0,
    q_floor: float = 0.80,
    op_label: str = "",
) -> tuple[bytes, float, float, np.ndarray, np.ndarray, int]:
    """
    Calibrates and tiles a 1x1 Conv2D Class B weight region for dynamic per-token input packing
    (INT8 input + INT16 output via dyntok ratios = amax_tok / in_t), using DropBOS RMS SmoothQuant
    (outliers > 3x median excluded from s_x and in_t, while included in y_norm_max bounds),
    float64 round-to-even arithmetic, and an 80th-percentile row L2-norm output bound floor (u_c).
    """
    M_out, K = w_t.shape
    w_f32 = w_t.float()
    x_f32 = X_cal.float()
    rm = x_f32.abs().amax(dim=1)
    clean_mask = rm <= (torch.median(rm) * 3.0)
    x_clean = x_f32[clean_mask] if int(clean_mask.sum().item()) >= 16 else x_f32
    s_x = x_clean.pow(2).mean(dim=0).sqrt().clamp(min=1e-5)
    s_w = w_f32.abs().amax(dim=0).clamp(min=1e-5)
    smooth = (s_x / s_w).sqrt().clamp(min=1e-3, max=1e3)
    inv_smooth = 1.0 / smooth

    X_sm_clean = x_clean * inv_smooth.unsqueeze(0)
    in_t = float(X_sm_clean.abs().amax(dim=1).clamp(min=1e-8).max().item())
    in_s = in_t / 127.0

    X_sm_all = x_f32 * inv_smooth.unsqueeze(0)
    amax_all = X_sm_all.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    x_q = torch.clamp(torch.round(X_sm_all * (127.0 / amax_all)), -127.0, 127.0).double()

    W_sm = (w_f32 * smooth.unsqueeze(0)).double()
    w_s = W_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    w_q_f64 = torch.clamp(torch.round(W_sm / w_s.unsqueeze(1)), -127.0, 127.0)

    u_c = W_sm.norm(dim=1).clamp(min=1e-8)
    y_norm_max = (F.linear(x_q, w_q_f64).abs().amax(dim=0)) * (in_s * w_s)
    qf = torch.quantile(y_norm_max / u_c, q_floor)
    y_bound_c = torch.maximum(y_norm_max * margin, qf * u_c * margin)

    min_out_s = (in_s * w_s * (1 << sh)) / 32767.0
    out_s_chan = torch.maximum(y_bound_c / 32767.0, min_out_s)

    qm_raw = torch.round((in_s * w_s / out_s_chan) * (1 << sh))
    n_hi = int((qm_raw > 32767.0).sum().item())
    if n_hi > 0:
        print(f"  [WARN] {op_label or 'conv'}: {n_hi} channels had qm_raw > 32767 at sh={sh} (max={float(qm_raw.max().item()):.1f})")
    qm_f = torch.clamp(qm_raw, 1.0, 32767.0)
    out_step_chan = ((in_s * w_s * (1 << sh)) / qm_f).float().numpy()

    w_q = w_q_f64.to(torch.int8)
    K_pad = ((K + 95) // 96) * 96
    w_pad = F.pad(w_q, (0, K_pad - K))
    n_slices = len(reg.contents) // 2
    M_s = M_out // n_slices
    w_tiled = w_pad.view(n_slices, M_s // 8, 8, K_pad // 96, 3, 8, 4).permute(0, 1, 3, 4, 5, 2, 6).contiguous().numpy()
    qm_i16 = qm_f.to(torch.int16).view(n_slices, M_s).numpy()

    buf = bytearray(reg.size)
    slice_w_bytes = (M_s * K_pad)
    slice_qm_bytes = M_s * 2
    for s in range(n_slices):
        buf[reg.offsets[2*s] : reg.offsets[2*s] + slice_w_bytes] = w_tiled[s].tobytes()
        buf[reg.offsets[2*s+1] : reg.offsets[2*s+1] + slice_qm_bytes] = qm_i16[s].tobytes()

    return bytes(buf), in_t, in_s, inv_smooth.float().numpy(), out_step_chan, n_hi


def collect_calibration_activations(sf, tok: AutoTokenizer, out_pt_path: Path) -> dict[str, torch.Tensor]:
    """
    Collects FP32 layer-by-layer calibration activations across diverse prose, ChatML turns,
    <think>...</think> formatting, and Python/C/C++ code sequences, streaming one layer at a
    time directly from safetensors to keep host memory usage minimal.
    """
    t0 = time.time()
    print(f"\n[+] Collecting FP32 calibration activations ({len(CALIB_TEXTS)} sequences) -> {out_pt_path}...")
    emb = sf.get_tensor("model.embed_tokens.weight").float()
    inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))

    seq_states = []
    tot_tokens = 0
    for idx, text in enumerate(CALIB_TEXTS):
        body = tok.encode(text, add_special_tokens=False)
        # Include BOS on the first 12 sequences so BOS-sink and non-BOS tokens are both represented
        ids = ([124894] + body) if (idx < 12) else body
        tot_tokens += len(ids)
        seq_states.append(emb[torch.tensor(ids, dtype=torch.long)].clone())
    del emb
    print(f"  [+] Prepared {len(seq_states)} calibration sequences ({tot_tokens} total tokens)")

    calib_x: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for l in range(24):
            tl0 = time.time()
            pfx = f"model.layers.{l}."
            op_norm = sf.get_tensor(pfx + "operator_norm.weight").float()
            ffn_norm = sf.get_tensor(pfx + "ffn_norm.weight").float()

            x_op_in_list = []
            x_op_out_list = []

            if l not in ATTN_LAYERS:
                w_in = sf.get_tensor(pfx + "conv.in_proj.weight").float()
                w_conv = sf.get_tensor(pfx + "conv.conv.weight").float()
                w_out = sf.get_tensor(pfx + "conv.out_proj.weight").float()
                for s_i, h in enumerate(seq_states):
                    T = h.shape[0]
                    u = rms_norm(h, op_norm)
                    x_op_in_list.append(u)
                    bcx = u @ w_in.t()
                    B, C, xp = bcx.chunk(3, dim=-1)
                    bx = (B * xp).transpose(0, 1).unsqueeze(0)
                    conv_out = F.conv1d(F.pad(bx, (2, 0)), w_conv, groups=2048)[0, :, :T].transpose(0, 1)
                    y_mid = C * conv_out
                    x_op_out_list.append(y_mid)
                    seq_states[s_i] = h + (y_mid @ w_out.t())
                calib_x[f"l{l:02d}_conv_in"] = torch.cat(x_op_in_list, dim=0)
                calib_x[f"l{l:02d}_conv_out"] = torch.cat(x_op_out_list, dim=0)
            else:
                wq = sf.get_tensor(pfx + "self_attn.q_proj.weight").float()
                wk = sf.get_tensor(pfx + "self_attn.k_proj.weight").float()
                wv = sf.get_tensor(pfx + "self_attn.v_proj.weight").float()
                w_qkv = torch.cat([wq, wk, wv], dim=0)
                qn = sf.get_tensor(pfx + "self_attn.q_layernorm.weight").float()
                kn = sf.get_tensor(pfx + "self_attn.k_layernorm.weight").float()
                wo = sf.get_tensor(pfx + "self_attn.out_proj.weight").float()
                for s_i, h in enumerate(seq_states):
                    T = h.shape[0]
                    u = rms_norm(h, op_norm)
                    x_op_in_list.append(u)
                    qkv = u @ w_qkv.t()
                    q_raw = rms_norm(qkv[:, :2048].view(T, 32, 64), qn)
                    k_raw = rms_norm(qkv[:, 2048:2560].view(T, 8, 64), kn)
                    v = qkv[:, 2560:].view(T, 8, 64)
                    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
                    cos_tab = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
                    sin_tab = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)
                    q = q_raw * cos_tab + rotate_half(q_raw) * sin_tab
                    k = k_raw * cos_tab + rotate_half(k_raw) * sin_tab
                    scores = (q.transpose(0, 1) @ k.transpose(0, 1).repeat_interleave(4, dim=0).transpose(-1, -2)) * 0.125
                    probs = F.softmax(scores + torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0), dim=-1)
                    attn_y = (probs @ v.transpose(0, 1).repeat_interleave(4, dim=0)).transpose(0, 1).contiguous().view(T, 2048)
                    x_op_out_list.append(attn_y)
                    seq_states[s_i] = h + (attn_y @ wo.t())
                calib_x[f"l{l:02d}_attn_qkv"] = torch.cat(x_op_in_list, dim=0)
                calib_x[f"l{l:02d}_attn_out"] = torch.cat(x_op_out_list, dim=0)

            # Collect FFN inputs & advance hidden states
            x_ffn_list = [rms_norm(h, ffn_norm) for h in seq_states]
            calib_x[f"l{l:02d}_ffn"] = torch.cat(x_ffn_list, dim=0)
            if l < 2:
                w1 = sf.get_tensor(pfx + "feed_forward.w1.weight").float()
                w3 = sf.get_tensor(pfx + "feed_forward.w3.weight").float()
                w2 = sf.get_tensor(pfx + "feed_forward.w2.weight").float()
                for s_i, (h, v_cur) in enumerate(zip(seq_states, x_ffn_list)):
                    seq_states[s_i] = h + ((F.silu(v_cur @ w1.t()) * (v_cur @ w3.t())) @ w2.t())
            else:
                gw = sf.get_tensor(pfx + "feed_forward.gate.weight").float()
                eb = sf.get_tensor(pfx + "feed_forward.expert_bias").float()
                experts_w = {}
                for s_i, (h, v_cur) in enumerate(zip(seq_states, x_ffn_list)):
                    rw = torch.sigmoid(v_cur @ gw.t())
                    _, top_idx = torch.topk(rw + eb.unsqueeze(0), k=4, dim=-1)
                    top_w = torch.gather(rw, -1, top_idx)
                    top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-6)
                    ffn_out = torch.zeros_like(v_cur)
                    for e in torch.unique(top_idx).tolist():
                        if e not in experts_w:
                            epfx = f"{pfx}feed_forward.experts.{e}."
                            experts_w[e] = (
                                sf.get_tensor(epfx + "w1.weight").float(),
                                sf.get_tensor(epfx + "w3.weight").float(),
                                sf.get_tensor(epfx + "w2.weight").float(),
                            )
                        w1_e, w3_e, w2_e = experts_w[e]
                        e_out = (F.silu(v_cur @ w1_e.t()) * (v_cur @ w3_e.t())) @ w2_e.t()
                        for k_i in range(4):
                            ffn_out += (top_idx[:, k_i] == e).unsqueeze(-1).float() * top_w[:, k_i:k_i+1] * e_out
                    seq_states[s_i] = h + ffn_out
                del experts_w
            print(f"  [+] Calibration L{l:02d} collected in {time.time() - tl0:.2f}s")

        emb_norm = sf.get_tensor("model.embedding_norm.weight").float()
        calib_x["lm_head"] = torch.cat([rms_norm(h, emb_norm) for h in seq_states], dim=0)

    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"calib_x": calib_x, "num_tokens": tot_tokens}, str(out_pt_path))
    print(f"  [+] Saved calibration cache ({len(calib_x)} tensors, {tot_tokens} tokens) in {time.time() - t0:.1f}s")
    return calib_x


def build_expert_mlp_region(
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    X_cal: torch.Tensor,
    smooth1: torch.Tensor,
    inv_smooth1_np: np.ndarray,
    in_t: float,
    in_s: float,
    reg,
) -> tuple[bytes, np.ndarray]:
    """
    Calibrates and tiles an expert SwiGLU (2048 -> 1792 -> 2048) Class B weight region:
      - w1: [1792, 2048] INT8 (7 slices x 256 channels, sh1=16)
      - w3: [1792, 2048] INT8 (7 slices x 256 channels, sh3=15, bilinear gamma scaling)
      - w2: [2048, 1792] INT16 (8 slices x 256 channels, sh2=25, bilinear gamma scaling)
    """
    sh1, sh3, sh2 = 16, 15, 25
    s_c0 = 12.0 / 32767.0
    s_c1 = 20.0 / 32767.0
    s_m1 = 6.0 / 32767.0

    c0_cal = X_cal @ w1.t()
    c1_cal = X_cal @ w3.t()
    m1_cal = F.silu(c0_cal) * c1_cal
    Y_true = m1_cal @ w2.t()

    gamma = max(float(m1_cal.abs().max().item()) / 6.0, float(c1_cal.abs().max().item()) / 20.0, 1e-4)
    s_c1_eff = s_c1 * gamma
    s_m1_eff = s_m1 * gamma

    max_ws1 = 32767.0 * s_c0 / (in_s * (1 << sh1))
    max_ws3 = 32767.0 * s_c1_eff / (in_s * (1 << sh3))

    w1_sm = w1 * smooth1.unsqueeze(0)
    w3_sm = w3 * smooth1.unsqueeze(0)
    ws1 = torch.clamp(w1_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0, max=max_ws1)
    ws3 = torch.clamp(w3_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0, max=max_ws3)
    ws2 = w2.abs().amax(dim=1).clamp(min=1e-12) / 32767.0

    w1_q = torch.clamp(torch.round(w1_sm / ws1.unsqueeze(1)), -127, 127).to(torch.int8)
    w3_q = torch.clamp(torch.round(w3_sm / ws3.unsqueeze(1)), -127, 127).to(torch.int8)
    w2_q = torch.clamp(torch.round(w2 / ws2.unsqueeze(1)), -32767, 32767).to(torch.int16)

    w1_tiled = F.pad(w1_q, (0, 64)).view(7, 32, 8, 22, 3, 8, 4).permute(0, 1, 3, 4, 5, 2, 6).contiguous().numpy()
    w3_tiled = F.pad(w3_q, (0, 64)).view(7, 32, 8, 22, 3, 8, 4).permute(0, 1, 3, 4, 5, 2, 6).contiguous().numpy()
    w2_tiled = F.pad(w2_q, (0, 32)).view(8, 32, 8, 38, 3, 8, 2).permute(0, 1, 3, 4, 5, 2, 6).contiguous().numpy()

    qm1 = torch.clamp(torch.round((in_s * ws1 / s_c0) * (1 << sh1)), 1.0, 32767.0).to(torch.int16).view(7, 256).numpy()
    qm3 = torch.clamp(torch.round((in_s * ws3 / s_c1_eff) * (1 << sh3)), 1.0, 32767.0).to(torch.int16).view(7, 256).numpy()

    out_t_chan = Y_true.abs().amax(dim=0).clamp(min=1e-5) * 1.15
    min_out_s = (s_m1_eff * ws2 * (1 << sh2)) / 32767.0
    out_s_c = torch.max(out_t_chan / 32767.0, min_out_s)
    qm2_f = torch.clamp(torch.round((s_m1_eff * ws2 / out_s_c) * (1 << sh2)), 1.0, 32767.0)
    # Use sh2_hw = sh2 + 2 (27) in NPU SDP to provide 4.0x INT16 headroom for routed MoE tokens
    out_step_chan = ((s_m1_eff * ws2 * (1 << (sh2 + 2))) / qm2_f).float().numpy()
    qm2 = qm2_f.to(torch.int16).view(8, 256).numpy()

    buf = bytearray(reg.size)
    for s in range(7):
        buf[reg.offsets[2*s] : reg.offsets[2*s] + 540672] = w1_tiled[s].tobytes()
        buf[reg.offsets[2*s+1] : reg.offsets[2*s+1] + 512] = qm1[s].tobytes()
        buf[reg.offsets[14+2*s] : reg.offsets[14+2*s] + 540672] = w3_tiled[s].tobytes()
        buf[reg.offsets[15+2*s] : reg.offsets[15+2*s] + 512] = qm3[s].tobytes()
    for s in range(8):
        buf[reg.offsets[28+2*s] : reg.offsets[28+2*s] + 933888] = w2_tiled[s].tobytes()
        buf[reg.offsets[29+2*s] : reg.offsets[29+2*s] + 512] = qm2[s].tobytes()

    return bytes(buf), out_step_chan


def build_lm_head_slices(
    emb_w: torch.Tensor,
    X_cal_lm: torch.Tensor,
    tmpl_lm_path: Path,
    models_out_dir: Path,
) -> tuple[float, float, np.ndarray, list[np.ndarray]]:
    """
    Patches the 8 Class A LM head slices (lfm2_lm_head_d{0,1}_s{0..3}.model) directly
    from lm_head_16k_int16_npu_b1.model with DropBOS RMS SmoothQuant and dyntok INT16 output steps.
    """
    sh = TRUNCS["lm_head"][0]
    x_f32 = X_cal_lm.float()
    rm = x_f32.abs().amax(dim=1)
    clean_mask = rm <= (torch.median(rm) * 3.0)
    x_clean = x_f32[clean_mask] if int(clean_mask.sum().item()) >= 16 else x_f32
    s_x = x_clean.pow(2).mean(dim=0).sqrt().clamp(min=1e-5)
    s_w = emb_w.abs().amax(dim=0).clamp(min=1e-5)
    smooth = (s_x / s_w).sqrt().clamp(min=1e-3, max=1e3)
    inv_smooth = 1.0 / smooth

    X_sm = x_clean * inv_smooth.unsqueeze(0)
    amax_tok = X_sm.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    in_t = float(amax_tok.max().item())
    in_s = in_t / 127.0
    x_q_all = torch.clamp(torch.round(X_sm * (127.0 / amax_tok)), -127.0, 127.0).double()

    W_sm = (emb_w.float() * smooth.unsqueeze(0)).double()
    w_s = W_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    w_q_f64 = torch.clamp(torch.round(W_sm / w_s.unsqueeze(1)), -127.0, 127.0)

    u_c = W_sm.norm(dim=1).clamp(min=1e-8)
    y_norm_max = (F.linear(x_q_all, w_q_f64).abs().amax(dim=0)) * (in_s * w_s)
    qf = torch.quantile(y_norm_max / u_c, 0.90)
    y_bound = torch.maximum(
        torch.maximum(y_norm_max * 2.0, qf * u_c * 2.0),
        torch.full_like(y_norm_max, 120.0),
    )
    min_out_s = (in_s * w_s * (1 << sh)) / 32767.0
    out_s_chan = torch.maximum(y_bound / 32767.0, min_out_s)
    qm_raw = torch.round((in_s * w_s / out_s_chan) * (1 << sh))
    n_hi = int((qm_raw > 32767.0).sum().item())
    assert n_hi == 0, f"LM head qm_raw > 32767 on {n_hi} channels at sh={sh}"
    qm_all_f = torch.clamp(qm_raw, 1.0, 32767.0)
    out_step_all = ((in_s * w_s * (1 << sh)) / qm_all_f).float().numpy()
    qm_all_i16 = qm_all_f.to(torch.int16).numpy()

    w_q_all = w_q_f64.to(torch.int8)
    w_pad_all = F.pad(w_q_all, (0, 64))  # K=2048 -> K_pad=2112

    m = ModelReader.read_file(str(tmpl_lm_path))
    reg = find_weight_regions(m)[0]
    blob_dict = {b.name: b for b in m.blobs}

    op_bytes = bytearray(blob_dict["kmd-task-op_list"].data)
    n_recs = len(op_bytes) // 1984
    hits = [r * 1984 + 82 for r in range(n_recs) if op_bytes[r*1984+82-12:r*1984+82] == SDP_SIG]
    assert len(hits) == 63
    for off in hits:
        op_bytes[off] = sh
    blob_dict["kmd-task-op_list"].data = bytes(op_bytes)

    slice_out_steps = []
    for d in range(2):
        for s in range(4):
            idx_16k = d * 4 + s
            c_start = idx_16k * 16000
            c_end = c_start + 16000
            slice_out_steps.append(out_step_all[c_start:c_end])

            w_sub = w_pad_all[c_start:c_end]
            qm_sub = qm_all_i16[c_start:c_end]

            # First 62 blocks of 256 channels (15,872 channels)
            w_62 = w_sub[:15872].view(62, 32, 8, 22, 3, 8, 4).permute(0, 1, 3, 4, 5, 2, 6).contiguous().numpy()
            qm_62 = qm_sub[:15872].reshape(62, 256)
            for b_i in range(62):
                blob_dict[reg.contents[2*b_i]].data = w_62[b_i].tobytes()
                blob_dict[reg.contents[2*b_i+1]].data = qm_62[b_i].tobytes()

            # Last block of 128 channels
            w_last = w_sub[15872:].view(16, 8, 22, 3, 8, 4).permute(0, 2, 3, 4, 1, 5).contiguous().numpy()
            qm_last = qm_sub[15872:]
            blob_dict[reg.contents[124]].data = w_last.tobytes()
            blob_dict[reg.contents[125]].data = qm_last.tobytes()

            dst_p = models_out_dir / f"lfm2_lm_head_d{d}_s{s}.model"
            ModelWriter.write_file(m, str(dst_p))
            print(f"  [+] Built LM head slice {dst_p.name} ({dst_p.stat().st_size / 1e6:.2f} MB)")

    return in_t, in_s, inv_smooth.float().numpy(), slice_out_steps


def generate_golden_bin(sf, tok: AutoTokenizer, out_bin_path: Path):
    """
    Generates golden_ref.bin in exact lfm3_v2 --verify-golden binary layout for
    PROMPT = '<|startoftext|>The capital of France is' (6 prompt tokens, 3 greedy steps).
    """
    print("\n[+] Generating FP32 golden_ref.bin for hardware --verify-golden gate...")
    in_ids = [124894] + tok.encode("The capital of France is", add_special_tokens=False)
    n_in = len(in_ids)
    n_gen = 3

    W = {k: sf.get_tensor(k).float() for k in sf.keys()}
    inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))

    def run_fp32(seq_ids: list[int]):
        T = len(seq_ids)
        h = W["model.embed_tokens.weight"][torch.tensor(seq_ids, dtype=torch.long)].clone()
        freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
        cos_tab = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
        sin_tab = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)
        layer_last = []
        for l in range(24):
            u = rms_norm(h, W[f"model.layers.{l}.operator_norm.weight"])
            if l not in ATTN_LAYERS:
                bcx = u @ W[f"model.layers.{l}.conv.in_proj.weight"].t()
                B, C, xp = bcx.chunk(3, dim=-1)
                bx = (B * xp).transpose(0, 1).unsqueeze(0)
                conv_out = F.conv1d(F.pad(bx, (2, 0)), W[f"model.layers.{l}.conv.conv.weight"], groups=2048)[0, :, :T].transpose(0, 1)
                h = h + (C * conv_out) @ W[f"model.layers.{l}.conv.out_proj.weight"].t()
            else:
                q_raw = rms_norm((u @ W[f"model.layers.{l}.self_attn.q_proj.weight"].t()).view(T, 32, 64), W[f"model.layers.{l}.self_attn.q_layernorm.weight"])
                k_raw = rms_norm((u @ W[f"model.layers.{l}.self_attn.k_proj.weight"].t()).view(T, 8, 64), W[f"model.layers.{l}.self_attn.k_layernorm.weight"])
                v = (u @ W[f"model.layers.{l}.self_attn.v_proj.weight"].t()).view(T, 8, 64)
                q = q_raw * cos_tab + rotate_half(q_raw) * sin_tab
                k = k_raw * cos_tab + rotate_half(k_raw) * sin_tab
                scores = (q.transpose(0, 1) @ k.transpose(0, 1).repeat_interleave(4, dim=0).transpose(-1, -2)) * 0.125
                probs = F.softmax(scores + torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0), dim=-1)
                attn_y = (probs @ v.transpose(0, 1).repeat_interleave(4, dim=0)).transpose(0, 1).contiguous().view(T, 2048)
                h = h + attn_y @ W[f"model.layers.{l}.self_attn.out_proj.weight"].t()
            v_cur = rms_norm(h, W[f"model.layers.{l}.ffn_norm.weight"])
            if l < 2:
                h = h + ((F.silu(v_cur @ W[f"model.layers.{l}.feed_forward.w1.weight"].t()) * (v_cur @ W[f"model.layers.{l}.feed_forward.w3.weight"].t())) @ W[f"model.layers.{l}.feed_forward.w2.weight"].t())
            else:
                rw = torch.sigmoid(v_cur @ W[f"model.layers.{l}.feed_forward.gate.weight"].t())
                _, top_idx = torch.topk(rw + W[f"model.layers.{l}.feed_forward.expert_bias"].unsqueeze(0), k=4, dim=-1)
                top_w = torch.gather(rw, -1, top_idx)
                top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-6)
                ffn_out = torch.zeros_like(v_cur)
                for e in torch.unique(top_idx).tolist():
                    e_out = (F.silu(v_cur @ W[f"model.layers.{l}.feed_forward.experts.{e}.w1.weight"].t()) * (v_cur @ W[f"model.layers.{l}.feed_forward.experts.{e}.w3.weight"].t())) @ W[f"model.layers.{l}.feed_forward.experts.{e}.w2.weight"].t()
                    for k_i in range(4):
                        ffn_out += (top_idx[:, k_i] == e).unsqueeze(-1).float() * top_w[:, k_i:k_i+1] * e_out
                h = h + ffn_out
            layer_last.append(h[-1].clone())
        emb_norm = rms_norm(h[-1:], W["model.embedding_norm.weight"])[0]
        logits = emb_norm @ W["model.embed_tokens.weight"].t()
        return torch.stack(layer_last), emb_norm, logits

    gold_layers, gold_emb_norm, gold_logits = run_fp32(in_ids)
    curr_ids = list(in_ids)
    gold_gen = []
    for s in range(n_gen):
        if s == 0:
            nxt = int(gold_logits.argmax().item())
        else:
            _, _, lg = run_fp32(curr_ids)
            nxt = int(lg.argmax().item())
        gold_gen.append(nxt)
        curr_ids.append(nxt)

    out_bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin_path, "wb") as f:
        f.write(struct.pack("<ii", n_in, n_gen))
        f.write(np.asarray(in_ids, dtype=np.int32).tobytes())
        f.write(np.asarray(gold_gen, dtype=np.int32).tobytes())
        f.write(gold_layers.numpy().astype(np.float32).tobytes())
        f.write(gold_emb_norm.numpy().astype(np.float32).tobytes())
        f.write(gold_logits.numpy().astype(np.float32).tobytes())
        f.write(gold_logits.numpy().astype(np.float32).tobytes())
    print(f"  [+] Wrote {out_bin_path} (in_ids={in_ids}, gold_gen={gold_gen} -> {[tok.decode([t]) for t in gold_gen]})")


def build_classb_models(tmpl_map: dict[str, Path], models_out_dir: Path):
    cb_work = Path("/tmp/lfm2_cb_work")
    cb_work.mkdir(parents=True, exist_ok=True)
    try:
        for key, src_p in tmpl_map.items():
            arch = key.rsplit("_", 1)[0]
            stem = f"lfm2_{key}"
            patched_p = cb_work / f"{stem}.model"
            patch_model_truncates(src_p, patched_p, TRUNCS[arch])
            subprocess.run(
                [VENV_PY, str(COMPILER_PY), "-m", str(patched_p), "-o", str(cb_work), "--convert-weights"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            cb_out = cb_work / f"{stem}_classb.model"
            dst_cb = models_out_dir / f"{stem}_classb.model"
            shutil.copy2(cb_out, dst_cb)
            print(f"  [+] Built {dst_cb.name} ({dst_cb.stat().st_size} B, truncs={set(TRUNCS[arch])})")
    finally:
        shutil.rmtree(cb_work, ignore_errors=True)


def calibrate_layer_operators(sf, l: int, calib_x: dict, cached_regs: dict, margin: float):
    pfx = f"model.layers.{l}."
    if l not in ATTN_LAYERS:
        k_in, k_out = f"lfm2_l{l:02d}_conv_in", f"lfm2_l{l:02d}_conv_out"
        w_in = sf.get_tensor(pfx + "conv.in_proj.weight").float()
        w_out = sf.get_tensor(pfx + "conv.out_proj.weight").float()
        sh_in, sh_out = TRUNCS["conv_in"][0], TRUNCS["proj2048"][0]
        reg_in, cal_key_in, cal_key_out = cached_regs["conv_in"], f"l{l:02d}_conv_in", f"l{l:02d}_conv_out"
    else:
        k_in, k_out = f"lfm2_l{l:02d}_attn_qkv", f"lfm2_l{l:02d}_attn_out"
        wq = sf.get_tensor(pfx + "self_attn.q_proj.weight").float()
        wk = sf.get_tensor(pfx + "self_attn.k_proj.weight").float()
        wv = sf.get_tensor(pfx + "self_attn.v_proj.weight").float()
        w_in = torch.cat([wq, wk, wv], dim=0)
        w_out = sf.get_tensor(pfx + "self_attn.out_proj.weight").float()
        sh_in, sh_out = TRUNCS["attn_qkv"][0], TRUNCS["proj2048"][0]
        reg_in, cal_key_in, cal_key_out = cached_regs["attn_qkv"], f"l{l:02d}_attn_qkv", f"l{l:02d}_attn_out"

    res_in = build_single_conv_region_dyntok(w_in, calib_x[cal_key_in], sh_in, reg_in, margin=margin, op_label=k_in)
    res_out = build_single_conv_region_dyntok(w_out, calib_x[cal_key_out], sh_out, cached_regs["proj2048"], margin=margin, op_label=k_out)
    return (k_in, res_in), (k_out, res_out)


def patch_operators_only(sf, tok: AutoTokenizer, args):
    t_start = time.time()
    print("=" * 80)
    print(f" LiquidAI / LFM2.5-8B-A1B Milestone 16: Operator-Only Patch (margin={args.op_margin:.2f})")
    print("=" * 80)

    print("\n[1/4] Rebuilding 9 Operator Class B Loadables (conv_in sh=%d, proj2048 sh=%d, attn_qkv sh=%d)..." % (
        TRUNCS["conv_in"][0], TRUNCS["proj2048"][0], TRUNCS["attn_qkv"][0]))
    build_classb_models(OP_TMPL_MAP, PKG_OUT_DIR / "models")

    cached_regs = {}
    for k in ["conv_in_t1", "proj2048_t1", "attn_qkv_t1"]:
        m = ModelReader.read_file(str(OP_TMPL_MAP[k]))
        cached_regs[k.rsplit("_", 1)[0]] = find_weight_regions(m)[0]

    print(f"\n[2/4] Loading FP32 calibration dataset ({CALIB_PT_PATH})...")
    if args.recalibrate or not CALIB_PT_PATH.exists():
        calib_x = collect_calibration_activations(sf, tok, CALIB_PT_PATH)
    else:
        calib_x = torch.load(str(CALIB_PT_PATH))["calib_x"]

    print("\n[3/4] Patching 8 INT16 LM Head slices + 48 operator regions in weights_die{0,1}.bin + quant_chan_scales.bin...")
    w_map_path = PKG_OUT_DIR / "weights_map.json"
    qs_path = PKG_OUT_DIR / "quant_scales.json"
    chan_bin_path = PKG_OUT_DIR / "quant_chan_scales.bin"
    w_map = json.loads(w_map_path.read_text())
    qs = json.loads(qs_path.read_text())
    chan_arr = np.fromfile(str(chan_bin_path), dtype=np.float32)

    emb_f32 = sf.get_tensor("model.embed_tokens.weight").float()
    lm_in_t, lm_in_s, lm_inv_smooth, lm_slice_out_steps = build_lm_head_slices(
        emb_f32, calib_x["lm_head"], LM_TMPL_PATH, PKG_OUT_DIR / "models"
    )
    del emb_f32
    lm_sc0 = qs["subgraphs"]["lfm2_lm_head_d0_s0"]
    lm_in_off, lm_in_cnt = int(lm_sc0["in_smooth_offset"]), int(lm_sc0["in_smooth_count"])
    assert lm_in_cnt == lm_inv_smooth.size
    chan_arr[lm_in_off : lm_in_off + lm_in_cnt] = lm_inv_smooth
    for d in range(2):
        for s in range(4):
            k_lm = f"lfm2_lm_head_d{d}_s{s}"
            sc = qs["subgraphs"][k_lm]
            out_off, out_cnt = int(sc["out_step_offset"]), int(sc["out_step_count"])
            osc_lm = lm_slice_out_steps[d * 4 + s]
            assert out_cnt == osc_lm.size
            chan_arr[out_off : out_off + out_cnt] = osc_lm
            out_s_med = float(np.median(osc_lm))
            sc["input_dtype"] = "int8"
            sc["output_dtype"] = "int16"
            sc["input_threshold"] = float(lm_in_t)
            sc["input_step"] = float(lm_in_s)
            sc["output_threshold"] = float(out_s_med * 32767.0)
            sc["output_step"] = float(out_s_med)

    patch_records = []
    tot_hi = 0
    with open(PKG_OUT_DIR / "weights_die0.bin", "r+b") as df0, open(PKG_OUT_DIR / "weights_die1.bin", "r+b") as df1:
        die_files = {0: df0, 1: df1}
        for l in range(24):
            (k_in, res_in), (k_out, res_out) = calibrate_layer_operators(sf, l, calib_x, cached_regs, args.op_margin)
            for k_name, (blob_bytes, in_t, in_s, inv_sm, osc, n_hi) in [(k_in, res_in), (k_out, res_out)]:
                tot_hi += n_hi
                wm = w_map[k_name]
                die = int(wm["die"])
                off = int(wm["weight_offset"])
                sz = int(wm["weight_size"])
                assert len(blob_bytes) == sz, f"{k_name}: size mismatch {len(blob_bytes)} vs {sz}"
                die_files[die].seek(off)
                die_files[die].write(blob_bytes)
                if args.emit_roma_patch is not None:
                    patch_records.append((die, off, blob_bytes))

                sc = qs["subgraphs"][k_name]
                in_off, in_cnt = int(sc["in_smooth_offset"]), int(sc["in_smooth_count"])
                out_off, out_cnt = int(sc["out_step_offset"]), int(sc["out_step_count"])
                assert in_cnt == inv_sm.size, f"{k_name}: in_cnt {in_cnt} != {inv_sm.size}"
                assert out_cnt == osc.size, f"{k_name}: out_cnt {out_cnt} != {osc.size}"
                chan_arr[in_off : in_off + in_cnt] = inv_sm
                chan_arr[out_off : out_off + out_cnt] = osc

                out_s_med = float(np.median(osc))
                sc["input_dtype"] = "int8"
                sc["output_dtype"] = "int16"
                sc["input_threshold"] = float(in_t)
                sc["input_step"] = float(in_s)
                sc["output_threshold"] = float(out_s_med * 32767.0)
                sc["output_step"] = float(out_s_med)

    chan_arr.tofile(str(chan_bin_path))
    qs_path.write_text(json.dumps(qs, indent=2))

    if args.emit_roma_patch is not None:
        print(f"\n[4/4] Writing binary operator patch -> {args.emit_roma_patch} ({len(patch_records)} regions)...")
        args.emit_roma_patch.parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_roma_patch, "wb") as pf:
            pf.write(b"LFM2OPAT")
            pf.write(struct.pack("<I", len(patch_records)))
            for die, off, blob in patch_records:
                pf.write(struct.pack("<IQQ", die, off, len(blob)))
                pf.write(blob)
        print(f"  [+] Wrote {args.emit_roma_patch} ({args.emit_roma_patch.stat().st_size / (1024*1024):.2f} MiB)")

    print("\n" + "=" * 80)
    print(f" Operator Patch COMPLETE in {time.time() - t_start:.1f}s | total qm_raw > 32767 = {tot_hi}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Build or patch LFM2.5-8B-A1B ENNP package")
    parser.add_argument("--patch-operators-only", action="store_true",
                        help="Rebuild only the 9 operator Class B .model files and patch the 48 operator regions in-place")
    parser.add_argument("--op-margin", type=float, default=2.0,
                        help="Safety margin multiplier on operator INT16 output bounds (default: 2.0)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force regenerating FP32 calibration activations cache")
    parser.add_argument("--emit-roma-patch", type=Path, default=None,
                        help="Optional binary patch file containing the 48 patched operator regions for in-place deployment on roma")
    args = parser.parse_args()

    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt", device="cpu")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

    if args.patch_operators_only:
        patch_operators_only(sf, tok, args)
        return

    t_start = time.time()
    print("=" * 80)
    print(" LiquidAI / LFM2.5-8B-A1B Sparse MoE: Dual-Die ENNP Package Builder")
    print("=" * 80)

    (PKG_OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_DIR / "tokenizer.json", PKG_OUT_DIR / "tokenizer.json")

    # 1. Build embedding.bin ([128000, 2048] FP16)
    emb_bin_path = PKG_OUT_DIR / "embedding.bin"
    print("\n[1/6] Exporting FP16 embedding.bin ([128000, 2048])...")
    emb_f32 = sf.get_tensor("model.embed_tokens.weight").float()
    emb_f32.half().numpy().tofile(str(emb_bin_path))
    print(f"  [+] Wrote {emb_bin_path} ({emb_bin_path.stat().st_size / (1024*1024):.1f} MiB)")

    # 2. Build cpu_weights.bin + cpu_weights.json
    print("\n[2/6] Exporting CPU weights (RMSNorms, K=3 depthwise convs, Q/K norms, MoE routers)...")
    cpu_bin_path = PKG_OUT_DIR / "cpu_weights.bin"
    cpu_json_path = PKG_OUT_DIR / "cpu_weights.json"
    cpu_meta = {"tensors": {}}
    cpu_off = 0
    with open(cpu_bin_path, "wb") as cf:
        def add_cpu_tensor(name: str, t: torch.Tensor):
            nonlocal cpu_off
            arr = np.ascontiguousarray(t.float().numpy(), dtype=np.float32)
            raw = arr.tobytes()
            cf.write(raw)
            cpu_meta["tensors"][name] = {
                "offset": cpu_off,
                "bytes": len(raw),
                "shape": list(arr.shape),
            }
            cpu_off += len(raw)

        add_cpu_tensor("model.embedding_norm.weight", sf.get_tensor("model.embedding_norm.weight"))
        for l in range(24):
            add_cpu_tensor(f"model.layers.{l}.operator_norm.weight", sf.get_tensor(f"model.layers.{l}.operator_norm.weight"))
            add_cpu_tensor(f"model.layers.{l}.ffn_norm.weight", sf.get_tensor(f"model.layers.{l}.ffn_norm.weight"))
            if l not in ATTN_LAYERS:
                cw = sf.get_tensor(f"model.layers.{l}.conv.conv.weight").squeeze(1)  # [2048, 3]
                add_cpu_tensor(f"model.layers.{l}.conv.conv.weight", cw)
            else:
                add_cpu_tensor(f"model.layers.{l}.self_attn.q_layernorm.weight", sf.get_tensor(f"model.layers.{l}.self_attn.q_layernorm.weight"))
                add_cpu_tensor(f"model.layers.{l}.self_attn.k_layernorm.weight", sf.get_tensor(f"model.layers.{l}.self_attn.k_layernorm.weight"))
            if l >= 2:
                add_cpu_tensor(f"model.layers.{l}.feed_forward.gate.weight", sf.get_tensor(f"model.layers.{l}.feed_forward.gate.weight"))
                add_cpu_tensor(f"model.layers.{l}.feed_forward.expert_bias", sf.get_tensor(f"model.layers.{l}.feed_forward.expert_bias"))
    cpu_json_path.write_text(json.dumps(cpu_meta, indent=2))
    print(f"  [+] Wrote {cpu_bin_path} ({cpu_off / (1024*1024):.2f} MiB, {len(cpu_meta['tensors'])} tensors)")

    # 3. Build 12 Class B Archetype Loadables (t1, t8, t64 for conv_in, proj2048, attn_qkv, expert_mlp)
    print("\n[3/6] Building 12 Class B Archetype Loadables (conv_in, proj2048, attn_qkv, expert_mlp x {t1,t8,t64})...")
    tmpl_map = {**OP_TMPL_MAP, **EXP_TMPL_MAP}
    build_classb_models(tmpl_map, PKG_OUT_DIR / "models")

    # Pre-cache template weight regions
    cached_regs = {}
    for k in ["conv_in_t1", "proj2048_t1", "attn_qkv_t1", "expert_mlp_t1"]:
        m = ModelReader.read_file(str(tmpl_map[k]))
        cached_regs[k.rsplit("_", 1)[0]] = find_weight_regions(m)[0]

    # 4. Load FP32 calibration dataset & build 8 dyntok INT16 LM head slices
    print("\n[4/6] Building 8 dyntok INT16 LM Head Slices (lfm2_lm_head_d{0,1}_s{0..3}.model)...")
    if args.recalibrate or not CALIB_PT_PATH.exists():
        calib_x = collect_calibration_activations(sf, tok, CALIB_PT_PATH)
    else:
        calib_x = torch.load(str(CALIB_PT_PATH))["calib_x"]
    lm_in_t, lm_in_s, lm_inv_smooth, lm_slice_out_steps = build_lm_head_slices(
        emb_f32,
        calib_x["lm_head"],
        LM_TMPL_PATH,
        PKG_OUT_DIR / "models",
    )

    # 5. Calibrate, quantize, tile, and pack all 24 layers (48 operators + 712 expert blocks)
    print("\n[5/6] Packing Dual-Die NPU Weight Blobs (weights_die0.bin & weights_die1.bin)...")
    die_files = {
        0: open(PKG_OUT_DIR / "weights_die0.bin", "wb"),
        1: open(PKG_OUT_DIR / "weights_die1.bin", "wb"),
    }
    die_offs = {0: 0, 1: 0}
    w_map = {}
    qs = {"subgraphs": {}}
    chan_floats = []

    def append_chan_vec(arr: np.ndarray) -> tuple[int, int]:
        off = len(chan_floats)
        arr_f32 = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        chan_floats.extend(arr_f32.tolist())
        return off, int(arr_f32.size)

    def record_scale_entry(k_name: str, in_dt: str, out_dt: str, in_t: float, in_s: float,
                           in_off: int, in_cnt: int, out_step_c: np.ndarray):
        out_off, out_cnt = append_chan_vec(out_step_c)
        out_s_med = float(np.median(out_step_c))
        out_div = 32767.0 if out_dt == "int16" else 127.0
        qs["subgraphs"][k_name] = {
            "input_dtype": in_dt,
            "output_dtype": out_dt,
            "input_threshold": float(in_t),
            "input_step": float(in_s),
            "output_threshold": float(out_s_med * out_div),
            "output_step": float(out_s_med),
            "in_smooth_offset": in_off,
            "in_smooth_count": in_cnt,
            "out_step_offset": out_off,
            "out_step_count": out_cnt,
        }

    # Record LM head scale entries (sharing one in_inv_smooth vector)
    lm_in_off, lm_in_cnt = append_chan_vec(lm_inv_smooth)
    for d in range(2):
        for s in range(4):
            k_lm = f"lfm2_lm_head_d{d}_s{s}"
            record_scale_entry(k_lm, "int8", "int16", lm_in_t, lm_in_s, lm_in_off, lm_in_cnt, lm_slice_out_steps[d*4 + s])

    for l in range(24):
        t_l0 = time.time()
        op_die = 0 if l < 12 else 1
        pfx = f"model.layers.{l}."
        (k_in, res_in), (k_out, res_out) = calibrate_layer_operators(sf, l, calib_x, cached_regs, args.op_margin)
        b_in, in_t_i, in_s_i, inv_i, osc_i, _ = res_in
        b_out, in_t_o, in_s_o, inv_o, osc_o, _ = res_out

        in_off_i, in_cnt_i = append_chan_vec(inv_i)
        in_off_o, in_cnt_o = append_chan_vec(inv_o)
        record_scale_entry(k_in,  "int8", "int16", in_t_i, in_s_i, in_off_i, in_cnt_i, osc_i)
        record_scale_entry(k_out, "int8", "int16", in_t_o, in_s_o, in_off_o, in_cnt_o, osc_o)

        for k_name, blob_bytes in [(k_in, b_in), (k_out, b_out)]:
            w_map[k_name] = {
                "layer": l,
                "die": op_die,
                "weight_offset": die_offs[op_die],
                "weight_size": len(blob_bytes),
            }
            die_files[op_die].write(blob_bytes)
            die_offs[op_die] += len(blob_bytes)

        # FFN / MoE Experts (shared input SmoothQuant per layer, EP=2 even->Die0, odd->Die1)
        X_cal_ffn = calib_x[f"l{l:02d}_ffn"]
        x_chan_max = X_cal_ffn.abs().amax(dim=0).clamp(min=1e-5)
        if l < 2:
            w1_full = sf.get_tensor(pfx + "feed_forward.w1.weight").float()
            w3_full = sf.get_tensor(pfx + "feed_forward.w3.weight").float()
            w2_full = sf.get_tensor(pfx + "feed_forward.w2.weight").float()
            w13_in_max = torch.max(w1_full.abs().amax(dim=0), w3_full.abs().amax(dim=0)).clamp(min=1e-5)
            n_exp = 4
        else:
            w1_0 = sf.get_tensor(pfx + "feed_forward.experts.0.w1.weight").float()
            w3_0 = sf.get_tensor(pfx + "feed_forward.experts.0.w3.weight").float()
            w13_in_max = torch.max(w1_0.abs().amax(dim=0), w3_0.abs().amax(dim=0)).clamp(min=1e-5)
            n_exp = 32

        smooth1 = (x_chan_max.pow(0.5) / w13_in_max.pow(0.5)).clamp(min=1e-3, max=1e3)
        inv_smooth1 = 1.0 / smooth1
        inv_smooth1_np = inv_smooth1.float().numpy()
        in_t_ffn = float((X_cal_ffn * inv_smooth1.unsqueeze(0)).abs().max().item() * 1.05)
        in_s_ffn = in_t_ffn / 127.0
        ffn_in_off, ffn_in_cnt = append_chan_vec(inv_smooth1_np)

        for e in range(n_exp):
            exp_die = e % 2
            k_exp = f"lfm2_l{l:02d}_exp{e:02d}"
            if l < 2:
                c0, c1 = e * 1792, (e + 1) * 1792
                w1_e = w1_full[c0:c1]
                w3_e = w3_full[c0:c1]
                w2_e = w2_full[:, c0:c1]
            else:
                epfx = f"{pfx}feed_forward.experts.{e}."
                w1_e = sf.get_tensor(epfx + "w1.weight").float()
                w3_e = sf.get_tensor(epfx + "w3.weight").float()
                w2_e = sf.get_tensor(epfx + "w2.weight").float()

            b_exp, osc_exp = build_expert_mlp_region(
                w1_e, w3_e, w2_e, X_cal_ffn, smooth1, inv_smooth1_np, in_t_ffn, in_s_ffn, cached_regs["expert_mlp"])
            record_scale_entry(k_exp, "int8", "int16", in_t_ffn, in_s_ffn, ffn_in_off, ffn_in_cnt, osc_exp)
            w_map[k_exp] = {
                "layer": l,
                "expert": e,
                "die": exp_die,
                "weight_offset": die_offs[exp_die],
                "weight_size": len(b_exp),
            }
            die_files[exp_die].write(b_exp)
            die_offs[exp_die] += len(b_exp)

        print(f"  [+] Packed Layer {l:02d} ({'attn' if l in ATTN_LAYERS else 'conv':4s}, {n_exp:2d} experts) in {time.time() - t_l0:.2f}s | Die0={die_offs[0]/1e9:.3f} GB, Die1={die_offs[1]/1e9:.3f} GB")

    die_files[0].close()
    die_files[1].close()

    chan_bin_path = PKG_OUT_DIR / "quant_chan_scales.bin"
    np.asarray(chan_floats, dtype=np.float32).tofile(str(chan_bin_path))
    (PKG_OUT_DIR / "weights_map.json").write_text(json.dumps(w_map, indent=2))
    (PKG_OUT_DIR / "quant_scales.json").write_text(json.dumps(qs, indent=2))

    manifest = {
        "model_name": "LFM2.5-8B-A1B",
        "architecture": "Lfm2MoeForCausalLM",
        "precision": "int8",
        "num_layers": 24,
        "hidden_size": 2048,
        "vocab_size": 128000,
        "num_dense_layers": 2,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1792,
        "conv_L_cache": 3,
        "rope_theta": 5000000.0,
        "attn_layers": sorted(list(ATTN_LAYERS)),
        "weights_die0_bytes": die_offs[0],
        "weights_die1_bytes": die_offs[1],
    }
    (PKG_OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 6. Generate golden_ref.bin for hardware --verify-golden
    generate_golden_bin(sf, tok, GOLDEN_DIR / "golden_ref.bin")

    print("\n" + "=" * 80)
    print(f" Package Build COMPLETE in {time.time() - t_start:.1f}s -> {PKG_OUT_DIR}")
    print(f"   weights_die0.bin : {die_offs[0] / (1024**3):.3f} GiB ({die_offs[0]} B)")
    print(f"   weights_die1.bin : {die_offs[1] / (1024**3):.3f} GiB ({die_offs[1]} B)")
    print(f"   quant_chan_scales: {chan_bin_path.stat().st_size / 1024:.1f} KiB ({len(chan_floats)} floats)")
    print("=" * 80)


if __name__ == "__main__":
    main()
