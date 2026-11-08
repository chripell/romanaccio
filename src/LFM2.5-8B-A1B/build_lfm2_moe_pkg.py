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
CALIB_PT_PATH   = Path("/tmp/lfm2_calib_fp32.pt")
COMPILER_PY     = Path(__file__).resolve().parent.parent / "eswin_compiler" / "compiler.py"
VENV_PY         = "/home/chri/aixigua/venv/bin/python3"

SDP_SIG = bytes.fromhex("000000000000010001010000")
ATTN_LAYERS = {2, 6, 10, 14, 18, 21}

TRUNCS = {
    "conv_in":    [20] * 12,
    "proj2048":   [20] * 4,
    "attn_qkv":   [21] * 6,
    "expert_mlp": [16] * 7 + [15] * 7 + [27] * 8,
    "lm_head":    [22] * 63,
}


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
) -> tuple[bytes, float, float, np.ndarray, np.ndarray]:
    """
    Calibrates and tiles a 1x1 Conv2D Class B weight region for dynamic per-token input packing
    (pack_npu_input with dyntok ratios = amax_tok / in_t).
    """
    M_out, K = w_t.shape
    Y_cal = X_cal @ w_t.t()
    x_ch = X_cal.abs().amax(dim=0).clamp(min=1e-5)
    w_ch = w_t.abs().amax(dim=0).clamp(min=1e-5)
    smooth = (x_ch.pow(0.5) / w_ch.pow(0.5)).clamp(min=1e-3, max=1e3)
    inv_smooth = 1.0 / smooth

    X_sm = X_cal * inv_smooth.unsqueeze(0)
    W_sm = w_t * smooth.unsqueeze(0)
    in_t = float(X_sm.abs().max().item())
    in_s = in_t / 127.0

    amax_cal = X_sm.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    ratios_cal = amax_cal / in_t
    out_t_chan = (Y_cal / ratios_cal).abs().amax(dim=0).clamp(min=1e-5) * 1.05

    w_s = W_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    min_out_s = (in_s * w_s * (1 << sh)) / 32767.0
    out_s_chan = torch.max(out_t_chan / 127.0, min_out_s)

    qm_f = torch.clamp(torch.round((in_s * w_s / out_s_chan) * (1 << sh)), 1.0, 32767.0)
    out_step_chan = ((in_s * w_s * (1 << sh)) / qm_f).float().numpy()

    w_q = torch.clamp(torch.round(W_sm / w_s.unsqueeze(1)), -127, 127).to(torch.int8)
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

    return bytes(buf), in_t, in_s, inv_smooth.float().numpy(), out_step_chan


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
    Patches the 8 static Class A LM head slices (lfm2_lm_head_d{0,1}_s{0..3}.model) directly
    from lfm3_lm_head_d0_s0.model with SmoothQuant input balancing and per-channel output steps.
    """
    sh = 22
    Y_true = X_cal_lm @ emb_w.t()
    x_ch = X_cal_lm.abs().amax(dim=0).clamp(min=1e-5)
    w_ch = emb_w.abs().amax(dim=0).clamp(min=1e-5)
    smooth = (x_ch.pow(0.5) / w_ch.pow(0.5)).clamp(min=1e-3, max=1e3)
    inv_smooth = 1.0 / smooth

    X_sm = X_cal_lm * inv_smooth.unsqueeze(0)
    W_sm = emb_w * smooth.unsqueeze(0)
    in_t = float(X_sm.abs().max().item() * 1.05)
    in_s = in_t / 127.0

    w_s = W_sm.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    out_t_chan = Y_true.abs().amax(dim=0).clamp(min=1e-5) * 1.15
    min_out_s = (in_s * w_s * (1 << sh)) / 32767.0
    out_s_chan = torch.max(out_t_chan / 127.0, min_out_s)
    qm_all_f = torch.clamp(torch.round((in_s * w_s / out_s_chan) * (1 << sh)), 1.0, 32767.0)
    out_step_all = ((in_s * w_s * (1 << sh)) / qm_all_f).float().numpy()
    qm_all_i16 = qm_all_f.to(torch.int16).numpy()

    w_q_all = torch.clamp(torch.round(W_sm / w_s.unsqueeze(1)), -127, 127).to(torch.int8)
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


def main():
    t_start = time.time()
    print("=" * 80)
    print(" LiquidAI / LFM2.5-8B-A1B Sparse MoE: Dual-Die ENNP Package Builder")
    print("=" * 80)

    (PKG_OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_DIR / "tokenizer.json", PKG_OUT_DIR / "tokenizer.json")

    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt", device="cpu")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

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
    tmpl_map = {
        "conv_in_t1":     LFM3_TEMPL_DIR / "lfm3_l00_conv_in_npu_b1.model",
        "conv_in_t8":     LFM3_TEMPL_DIR / "conv_in_t8_npu_b1.model",
        "conv_in_t64":    LFM3_TEMPL_DIR / "lfm3_l00_conv_in_t64_npu_b1.model",
        "proj2048_t1":    LFM3_TEMPL_DIR / "lfm3_l00_conv_out_npu_b1.model",
        "proj2048_t8":    LFM3_TEMPL_DIR / "proj2048_t8_npu_b1.model",
        "proj2048_t64":   LFM3_TEMPL_DIR / "lfm3_l00_conv_out_t64_npu_b1.model",
        "attn_qkv_t1":    LFM3_TEMPL_DIR / "lfm3_l02_attn_qkv_npu_b1.model",
        "attn_qkv_t8":    LFM3_TEMPL_DIR / "attn_qkv_t8_npu_b1.model",
        "attn_qkv_t64":   LFM3_TEMPL_DIR / "lfm3_l02_attn_qkv_t64_npu_b1.model",
        "expert_mlp_t1":  LFM2_TEMPL_DIR / "expert_mlp_t1_npu_b1.model",
        "expert_mlp_t8":  LFM2_TEMPL_DIR / "expert_mlp_t8_npu_b1.model",
        "expert_mlp_t64": LFM2_TEMPL_DIR / "expert_mlp_t64_npu_b1.model",
    }
    cb_work = Path("/tmp/lfm2_cb_work")
    cb_work.mkdir(parents=True, exist_ok=True)
    for key, src_p in tmpl_map.items():
        arch = key.rsplit("_", 1)[0]
        stem = f"lfm2_{key}"
        patched_p = cb_work / f"{stem}.model"
        patch_model_truncates(src_p, patched_p, TRUNCS[arch])
        subprocess.run([VENV_PY, str(COMPILER_PY), "-m", str(patched_p), "-o", str(cb_work), "--convert-weights"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cb_out = cb_work / f"{stem}_classb.model"
        dst_cb = PKG_OUT_DIR / "models" / f"{stem}_classb.model"
        shutil.copy2(cb_out, dst_cb)
        print(f"  [+] Built {dst_cb.name} ({dst_cb.stat().st_size} B)")
    shutil.rmtree(cb_work, ignore_errors=True)

    # Pre-cache template weight regions
    cached_regs = {}
    for k in ["conv_in_t1", "proj2048_t1", "attn_qkv_t1", "expert_mlp_t1"]:
        m = ModelReader.read_file(str(tmpl_map[k]))
        cached_regs[k.rsplit("_", 1)[0]] = find_weight_regions(m)[0]

    # 4. Load FP32 calibration dataset & build 8 LM head slices
    print("\n[4/6] Building 8 INT8 LM Head Slices (lfm2_lm_head_d{0,1}_s{0..3}.model)...")
    calib_x = torch.load(str(CALIB_PT_PATH))["calib_x"]
    lm_in_t, lm_in_s, lm_inv_smooth, lm_slice_out_steps = build_lm_head_slices(
        emb_f32,
        calib_x["lm_head"],
        LFM3_PKG_DIR / "models" / "lfm3_lm_head_d0_s0.model",
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
            record_scale_entry(k_lm, "int8", "int8", lm_in_t, lm_in_s, lm_in_off, lm_in_cnt, lm_slice_out_steps[d*4 + s])

    for l in range(24):
        t_l0 = time.time()
        op_die = 0 if l < 12 else 1
        pfx = f"model.layers.{l}."
        if l not in ATTN_LAYERS:
            k_in, k_out = f"lfm2_l{l:02d}_conv_in", f"lfm2_l{l:02d}_conv_out"
            w_in = sf.get_tensor(pfx + "conv.in_proj.weight").float()
            w_out = sf.get_tensor(pfx + "conv.out_proj.weight").float()
            b_in, in_t_i, in_s_i, inv_i, osc_i = build_single_conv_region_dyntok(
                w_in, calib_x[f"l{l:02d}_conv_in"], 20, cached_regs["conv_in"])
            b_out, in_t_o, in_s_o, inv_o, osc_o = build_single_conv_region_dyntok(
                w_out, calib_x[f"l{l:02d}_conv_out"], 20, cached_regs["proj2048"])
        else:
            k_in, k_out = f"lfm2_l{l:02d}_attn_qkv", f"lfm2_l{l:02d}_attn_out"
            wq = sf.get_tensor(pfx + "self_attn.q_proj.weight").float()
            wk = sf.get_tensor(pfx + "self_attn.k_proj.weight").float()
            wv = sf.get_tensor(pfx + "self_attn.v_proj.weight").float()
            wo = sf.get_tensor(pfx + "self_attn.out_proj.weight").float()
            b_in, in_t_i, in_s_i, inv_i, osc_i = build_single_conv_region_dyntok(
                torch.cat([wq, wk, wv], dim=0), calib_x[f"l{l:02d}_attn_qkv"], 21, cached_regs["attn_qkv"])
            b_out, in_t_o, in_s_o, inv_o, osc_o = build_single_conv_region_dyntok(
                wo, calib_x[f"l{l:02d}_attn_out"], 20, cached_regs["proj2048"])

        in_off_i, in_cnt_i = append_chan_vec(inv_i)
        in_off_o, in_cnt_o = append_chan_vec(inv_o)
        record_scale_entry(k_in,  "int8", "int8", in_t_i, in_s_i, in_off_i, in_cnt_i, osc_i)
        record_scale_entry(k_out, "int8", "int8", in_t_o, in_s_o, in_off_o, in_cnt_o, osc_o)

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
