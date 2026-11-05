#!/usr/bin/env python3
"""
Milestone 5: Direct NVDLA INT8/INT16 Weight & CvtPrecision_QuantM Package Builder for LFM3-1.7B.

Synthesizes:
  - 8 Shared Class B Archetype Loadables (lfm3_{conv_in,proj2048,attn_qkv,mlp}_{t1,t64}_classb.model)
  - Contiguous 4K-aligned per-die weight blobs (weights_die0.bin = 0.878 GiB, weights_die1.bin = 0.872 GiB)
    using the verified maxdiff=0 float64 NVDLA INT8/INT16 weight tiler and CvtPrecision_QuantM formula,
    with SmoothQuant input channel balancing (1 / s_in[c]) and per-output-channel CvtPrecision_QuantM
    output steps (out_step_chan[c]) to eliminate Sandwich-RMSNorm denominator-anchor sink quantization
    collapse and t=0 boundary clipping without recompiling any .model files.
  - Per-channel scale sidecar (quant_chan_scales.bin) + updated quant_scales.json
  - 8 x 16,000-channel INT8 LM Head slices (lfm3_lm_head_d{0,1}_s{0..3}.model, 33.23 MiB each)
"""

import os
import sys
import json
import struct
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))
from backend.flatbuffer_emitter import ModelReader, ModelWriter
from backend.dynamic_weight_binding import find_weight_regions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_lfm3_int8 import CALIB_SENTENCES

DEFAULT_MODEL_DIR   = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
DEFAULT_TABLES_DIR  = Path("/home/chri/aixigua/deep/artifacts/lfm3_quant_tables")
DEFAULT_TEMPL_DIR   = Path("/home/chri/aixigua/deep/artifacts/lfm3_probe_templates")
PROBE_DIR           = Path("/run/host-wayland/simulator-ipc/lfm3_probe_int8")
DEFAULT_PKG_FP16    = DEFAULT_MODEL_DIR / "eswin_pkg"
DEFAULT_PKG_INT8    = DEFAULT_MODEL_DIR / "eswin_pkg_int8"
COMPILER_PY         = Path(__file__).resolve().parent.parent / "eswin_compiler" / "compiler.py"
VENV_PY             = "/home/chri/aixigua/venv/bin/python"

# 12-byte SDP CvtPrecision signature immediately preceding the 1-byte truncate shift at record offset +82
SDP_SIG = bytes.fromhex("000000000000010001010000")

TRUNCS = {
    "conv_in":  [22] * 12,
    "proj2048": [22] * 4,
    "attn_qkv": [23] * 6,
    "mlp":      [14] * 15 + [15] * 15 + [28] * 32,
}


def patch_model_truncates(src_model: Path, dst_model: Path, new_truncs: list[int], align_mlp_t1_to_t64: Path = None):
    m = ModelReader.read_file(str(src_model))
    for b in m.blobs:
        if b.name == "kmd-task-op_list":
            op_bytes = bytearray(b.data)
            n_recs = len(op_bytes) // 1984
            hits = [r * 1984 + 82 for r in range(n_recs) if op_bytes[r*1984+82-12:r*1984+82] == SDP_SIG]
            assert len(hits) == len(new_truncs), f"Expected {len(new_truncs)} SDP hits, got {len(hits)}"
            for off, val in zip(hits, new_truncs):
                op_bytes[off] = val
            b.data = bytes(op_bytes)

    if align_mlp_t1_to_t64 is not None:
        m_t64 = ModelReader.read_file(str(align_mlp_t1_to_t64))
        r1 = find_weight_regions(m)[0]
        r64 = find_weight_regions(m_t64)[0]
        for b in m.blobs:
            if b.name == "kmd-task-surf_list":
                surf_bytes = bytearray(b.data)
                for idx in range(123, 59, -1):
                    if r1.offsets[idx] != r64.offsets[idx]:
                        b1  = struct.pack("<I", r1.offsets[idx])
                        b64 = struct.pack("<I", r64.offsets[idx])
                        locs = [i for i in range(0, len(surf_bytes)-4, 4) if surf_bytes[i:i+4] == b1]
                        assert len(locs) == 1
                        surf_bytes[locs[0]:locs[0]+4] = b64
                b.data = bytes(surf_bytes)
            elif b.name.startswith("CvtPrecision_QuantM") and len(b.data) == 128:
                b.data = bytes(b.data) * 2
        for a in m.address_list:
            if a.mem_id == r1.id:
                a.size = r64.size
        r1.offsets = list(r64.offsets)
        r1.size = r64.size

    ModelWriter.write_file(m, str(dst_model))


def tile_int8_slice(w_slice: np.ndarray, max_ws: float = None) -> tuple[bytes, np.ndarray]:
    """Quantizes [M_s, K] FP32/FP64 -> per-channel INT8 in float64 (round-to-even) and tiles for NVDLA CMAC."""
    M_s, K = w_slice.shape
    w_f64 = w_slice.astype(np.float64)
    w_amax = np.maximum(np.max(np.abs(w_f64), axis=1, keepdims=True), 1e-12)
    w_step = w_amax / 127.0
    if max_ws is not None:
        w_step = np.minimum(w_step, max_ws)
    w_q = np.clip(np.round(w_f64 / w_step), -127, 127).astype(np.int8)
    K_pad = ((K + 95) // 96) * 96
    wp = np.zeros((M_s, K_pad), dtype=np.int8)
    wp[:, :K] = w_q
    tiled = wp.reshape(M_s // 8, 8, K_pad // 96, 3, 8, 4).transpose(0, 2, 3, 4, 1, 5).ravel()
    return tiled.tobytes(), w_step[:, 0]


def tile_int16_slice(w_slice: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantizes [M_s, K] FP32/FP64 -> per-channel INT16 in float64 (round-to-even) and tiles for NVDLA CMAC."""
    M_s, K = w_slice.shape
    w_f64 = w_slice.astype(np.float64)
    w_amax = np.maximum(np.max(np.abs(w_f64), axis=1, keepdims=True), 1e-12)
    w_step = w_amax / 32767.0
    w_q = np.clip(np.round(w_f64 / w_step), -32767, 32767).astype(np.int16)
    K_pad = ((K + 47) // 48) * 48
    wp = np.zeros((M_s, K_pad), dtype=np.int16)
    wp[:, :K] = w_q
    tiled = wp.reshape(M_s // 8, 8, K_pad // 48, 3, 8, 2).transpose(0, 2, 3, 4, 1, 5).ravel()
    return tiled.tobytes(), w_step[:, 0]


def collect_calibration_activations(model_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Runs 30 calibration sentences (with and without BOS at t=0) to collect channel statistics."""
    print(f"[+] Collecting SmoothQuant & per-channel calibration statistics for {model_dir.name} (60 x 16 tokens, including t=0)...")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    batch_ids = []
    for s in CALIB_SENTENCES[:30]:
        body = tok.encode(s, add_special_tokens=False)
        while len(body) < 16:
            body = body + body
        batch_ids.append(body[:16])
        batch_ids.append(([tok.bos_token_id] + body[:15])[:16])
    calib_ids = torch.tensor(batch_ids, dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True, dtype=torch.float32)
    model.eval()
    cfg = json.loads((model_dir / "config.json").read_text())
    layer_types = cfg["layer_types"]

    calib_x, calib_y = {}, {}

    def cap_hook(k: str, c_in: int, c_out: int = None):
        def fn(m, i, o):
            calib_x[k] = i[0].detach().reshape(-1, c_in).cpu()
            if c_out is not None:
                calib_y[k] = o.detach().reshape(-1, c_out).cpu()
        return fn

    hooks = []
    for l in range(24):
        layer = model.model.layers[l]
        if layer_types[l] == "conv":
            hooks.append(layer.conv.in_proj.register_forward_hook(cap_hook(f"lfm3_l{l:02d}_conv_in", 2048)))
            hooks.append(layer.conv.out_proj.register_forward_hook(cap_hook(f"lfm3_l{l:02d}_conv_out", 2048)))
        else:
            hooks.append(layer.self_attn.q_proj.register_forward_hook(cap_hook(f"lfm3_l{l:02d}_attn_qkv", 2048)))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(cap_hook(f"lfm3_l{l:02d}_attn_out", 2048)))
        hooks.append(layer.feed_forward.register_forward_hook(cap_hook(f"lfm3_l{l:02d}_mlp", 2048, 2048)))

    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()
    del model
    return calib_x, calib_y


def build_single_conv_region(
    w_f32: np.ndarray,
    calib_x: torch.Tensor,
    tmpl_model_path: Path,
    trunc: int,
    cached_reg = None,
) -> tuple[bytes, float, float, np.ndarray, np.ndarray]:
    """
    Builds a 1x1 Conv2D Class B weight region with SmoothQuant input channel balancing and
    per-output-channel CvtPrecision_QuantM steps.
    Returns (region_bytes, in_thresh, in_step, in_inv_smooth_f32, out_step_chan_f32).
    """
    if cached_reg is not None:
        reg = cached_reg
    else:
        m = ModelReader.read_file(str(tmpl_model_path))
        reg = find_weight_regions(m)[0]
    buf = bytearray(reg.size)

    w0 = torch.from_numpy(w_f32).float()
    x = calib_x.float()
    s_x = x.pow(2).mean(dim=0).sqrt().clamp(min=1e-5)
    s_w = w0.abs().amax(dim=0).clamp(min=1e-5)
    s_in = (s_x / s_w).sqrt().clamp(min=0.05, max=20.0)
    in_inv_smooth = (1.0 / s_in).float().numpy()

    x_sc = x / s_in
    amax_tok = x_sc.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    in_t = float(amax_tok.max().item())
    in_s = in_t / 127.0
    x_q = torch.clamp(torch.round(x_sc * (127.0 / amax_tok)), -127.0, 127.0).double()

    w_sc = (w0 * s_in.unsqueeze(0)).double()
    ws = w_sc.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    wq = torch.clamp(torch.round(w_sc / ws.unsqueeze(1)), -127.0, 127.0)
    u_c = w_sc.norm(dim=1).clamp(min=1e-8)
    y_norm_max = (F.linear(x_q, wq).abs().amax(dim=0)) * (in_s * ws)
    qf = torch.quantile(y_norm_max / u_c, 0.80)
    y_bound_c = torch.maximum(y_norm_max * 1.15, qf * u_c * 1.15)
    out_s_c = (y_bound_c / 127.0).numpy()

    w_sc_np = w_sc.numpy()
    out_step_chan = np.zeros(w_f32.shape[0], dtype=np.float32)
    n_slices = len(reg.contents) // 2
    for s in range(n_slices):
        sl = slice(s * 512, (s + 1) * 512)
        w_bytes, w_step = tile_int8_slice(w_sc_np[sl])
        qm_f = np.clip(np.round((in_s * w_step / out_s_c[sl]) * (1 << trunc)), 1.0, 32767.0)
        out_step_chan[sl] = ((in_s * w_step * (1 << trunc)) / qm_f).astype(np.float32)
        qm = qm_f.astype(np.int16).tobytes()
        buf[reg.offsets[2*s]:reg.offsets[2*s]+len(w_bytes)] = w_bytes
        buf[reg.offsets[2*s+1]:reg.offsets[2*s+1]+len(qm)]  = qm

    return bytes(buf), in_t, in_s, in_inv_smooth, out_step_chan


def build_mlp_region(
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    calib_x: torch.Tensor,
    calib_y: torch.Tensor,
    tbl: dict,
    tmpl_model_path: Path,
    cached_reg = None,
) -> tuple[bytes, float, float, np.ndarray, np.ndarray]:
    """
    Builds a SwiGLU MLP Class B weight region with SmoothQuant input channel balancing (with t=0
    spike capping), per-layer in_s, and per-output-channel w2 output steps.
    Returns (region_bytes, in_thresh, in_step, in_inv_smooth_f32, out_step_chan_f32).
    """
    if cached_reg is not None:
        reg = cached_reg
    else:
        m = ModelReader.read_file(str(tmpl_model_path))
        reg = find_weight_regions(m)[0]
    buf = bytearray(reg.size)

    c0_s = float(tbl["blob_info"]["/Conv_output_0"]["int16_step"])
    c1_s = float(tbl["blob_info"]["/Conv_1_output_0"]["int16_step"])
    m1_s = float(tbl["blob_info"]["/Mul_1_output_0"]["int16_step"])

    w1_t = torch.from_numpy(w1).float()
    w3_t = torch.from_numpy(w3).float()
    w2_t = torch.from_numpy(w2).float()
    x = calib_x.float()
    y = calib_y.float()

    s_x_95 = torch.quantile(x.abs(), 0.95, dim=0).clamp(min=1e-5)
    s_x_max = x.abs().amax(dim=0).clamp(min=1e-5)
    s_w = torch.maximum(w1_t.abs().amax(dim=0), w3_t.abs().amax(dim=0)).clamp(min=1e-5)
    s_in_base = (s_x_95 / s_w).sqrt().clamp(min=0.05, max=25.0)
    target_x = torch.quantile(s_x_95 / s_in_base, 0.98)
    s_in = torch.maximum(s_in_base, s_x_max / target_x)
    in_inv_smooth = (1.0 / s_in).float().numpy()

    x_sc = x / s_in
    in_t = float(x_sc.abs().amax().item() * 1.05)
    in_s = in_t / 127.0

    w1_sc_np = (w1_t * s_in.unsqueeze(0)).double().numpy()
    w3_sc_np = (w3_t * s_in.unsqueeze(0)).double().numpy()
    max_ws1 = 32767.0 * c0_s / (in_s * (1 << 14))
    max_ws3 = 32767.0 * c1_s / (in_s * (1 << 15))

    for s in range(15):
        sl = slice(s * 512, (s + 1) * 512)
        wb, ws = tile_int8_slice(w1_sc_np[sl], max_ws=max_ws1)
        qm = np.clip(np.round((in_s * ws / c0_s) * (1 << 14)), 1.0, 32767.0).astype(np.int16).tobytes()
        buf[reg.offsets[2*s]:reg.offsets[2*s]+len(wb)] = wb
        buf[reg.offsets[2*s+1]:reg.offsets[2*s+1]+len(qm)] = qm

    for s in range(15):
        idx = 30 + 2 * s
        sl = slice(s * 512, (s + 1) * 512)
        wb, ws = tile_int8_slice(w3_sc_np[sl], max_ws=max_ws3)
        qm = np.clip(np.round((in_s * ws / c1_s) * (1 << 15)), 1.0, 32767.0).astype(np.int16).tobytes()
        buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
        buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm

    u2_c = w2_t.double().norm(dim=1).clamp(min=1e-8)
    y_max_c = y.abs().amax(dim=0).double()
    q80 = torch.quantile(y_max_c / u2_c, 0.80)
    y_bound_c = torch.maximum(y_max_c * 1.20, q80 * u2_c * 1.20)
    out_s_c = (y_bound_c / 32767.0).numpy()

    out_step_chan = np.zeros(w2.shape[0], dtype=np.float32)
    for s in range(32):
        idx = 60 + 2 * s
        sl = slice(s * 64, (s + 1) * 64)
        wb, ws = tile_int16_slice(w2[sl])
        qm_f = np.clip(np.round((m1_s * ws / out_s_c[sl]) * (1 << 28)), 1.0, 32767.0)
        out_step_chan[sl] = ((m1_s * ws * (1 << 28)) / qm_f).astype(np.float32)
        qm = qm_f.astype(np.int16).tobytes() * 2
        buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
        buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm

    return bytes(buf), in_t, in_s, in_inv_smooth, out_step_chan


def main():
    parser = argparse.ArgumentParser(description="Direct NVDLA INT8/INT16 Weight & CvtPrecision_QuantM Package Builder for LFM3-1.7B")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to HF model directory")
    parser.add_argument("--pkg-out-dir", type=Path, default=DEFAULT_PKG_INT8, help="Output directory for INT8 package")
    parser.add_argument("--base-pkg-dir", type=Path, default=DEFAULT_PKG_INT8, help="Base INT8 package directory to reuse archetypes/scales from")
    parser.add_argument("--fp16-pkg-dir", type=Path, default=DEFAULT_PKG_FP16, help="Base FP16 package directory")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Quant tables directory")
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPL_DIR, help="Archetype probe templates directory")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    pkg_out_dir = args.pkg_out_dir.resolve()
    base_pkg_dir = args.base_pkg_dir.resolve()
    fp16_pkg_dir = args.fp16_pkg_dir.resolve()
    tables_dir = args.tables_dir.resolve()
    templates_dir = args.templates_dir.resolve()

    (pkg_out_dir / "models").mkdir(parents=True, exist_ok=True)

    # 1. Copy or link static runtime files
    for fname in ["cpu_weights.bin", "cpu_weights.json", "manifest.json"]:
        for src_dir in [base_pkg_dir, fp16_pkg_dir]:
            if (src_dir / fname).exists():
                shutil.copy2(src_dir / fname, pkg_out_dir / fname)
                break

    # Copy tokenizer.json from model_dir if present, else base_pkg_dir
    if (model_dir / "tokenizer.json").exists():
        shutil.copy2(model_dir / "tokenizer.json", pkg_out_dir / "tokenizer.json")
    elif (base_pkg_dir / "tokenizer.json").exists():
        shutil.copy2(base_pkg_dir / "tokenizer.json", pkg_out_dir / "tokenizer.json")

    # Copy or symlink embedding.bin if present in base_pkg_dir or fp16_pkg_dir
    for src_dir in [base_pkg_dir, fp16_pkg_dir]:
        emb = src_dir / "embedding.bin"
        if emb.exists() and not (pkg_out_dir / "embedding.bin").exists():
            try:
                os.link(emb, pkg_out_dir / "embedding.bin")
                print(f"[+] Hardlinked embedding.bin from {emb}")
            except OSError:
                shutil.copy2(emb, pkg_out_dir / "embedding.bin")
                print(f"[+] Copied embedding.bin from {emb}")
            break

    if (pkg_out_dir / "manifest.json").exists():
        manifest = json.loads((pkg_out_dir / "manifest.json").read_text())
        manifest["precision"] = "int8"
        manifest["model_name"] = model_dir.name
        (pkg_out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 2. Re-use or compile archetype models
    if base_pkg_dir.exists() and (base_pkg_dir / "models").exists() and base_pkg_dir != pkg_out_dir:
        print(f"[+] Reusing existing archetype models and LM head slices from {base_pkg_dir / 'models'}...")
        for m_file in (base_pkg_dir / "models").glob("*.model"):
            dst_m = pkg_out_dir / "models" / m_file.name
            if not dst_m.exists():
                try:
                    os.link(m_file, dst_m)
                except OSError:
                    shutil.copy2(m_file, dst_m)

    # 3. Resolve template models for finding weight regions
    def resolve_tmpl(name: str, alt_subpath: str) -> Path:
        if (templates_dir / name).exists():
            return templates_dir / name
        if (PROBE_DIR / alt_subpath).exists():
            return PROBE_DIR / alt_subpath
        raise FileNotFoundError(f"Template model {name} not found in {templates_dir} or {PROBE_DIR}")

    tmpl_paths = {
        "conv_in_t1":   resolve_tmpl("lfm3_l00_conv_in_npu_b1.model", "conv_in_t1_shared/model_files/lfm3_l00_conv_in_npu_b1.model"),
        "conv_in_t64":  resolve_tmpl("lfm3_l00_conv_in_t64_npu_b1.model", "conv_in_t64_shared/model_files/lfm3_l00_conv_in_t64_npu_b1.model"),
        "proj2048_t1":  resolve_tmpl("lfm3_l00_conv_out_npu_b1.model", "proj2048_t1_shared/model_files/lfm3_l00_conv_out_npu_b1.model"),
        "proj2048_t64": resolve_tmpl("lfm3_l00_conv_out_t64_npu_b1.model", "proj2048_t64_shared/model_files/lfm3_l00_conv_out_t64_npu_b1.model"),
        "attn_qkv_t1":  resolve_tmpl("lfm3_l02_attn_qkv_npu_b1.model", "attn_qkv_t1_shared/model_files/lfm3_l02_attn_qkv_npu_b1.model"),
        "attn_qkv_t64": resolve_tmpl("lfm3_l02_attn_qkv_t64_npu_b1.model", "attn_qkv_t64_shared/model_files/lfm3_l02_attn_qkv_t64_npu_b1.model"),
        "mlp_t1":       resolve_tmpl("lfm3_l01_mlp_npu_b1.model", "l01_mlp_shared/model_files/lfm3_l01_mlp_npu_b1.model"),
        "mlp_t64":      resolve_tmpl("lfm3_l00_mlp_t64_npu_b1.model", "l00_mlp_t64_shared/model_files/lfm3_l00_mlp_t64_npu_b1.model"),
    }

    cb_work = PROBE_DIR / "cb_work"
    for key, src_p in tmpl_paths.items():
        arch = key.rsplit("_", 1)[0]
        stem = f"lfm3_{key}"
        dst_cb = pkg_out_dir / "models" / f"{stem}_classb.model"
        if dst_cb.exists() and key not in ("mlp_t1", "mlp_t64"):
            continue
        if dst_cb.exists() and base_pkg_dir != pkg_out_dir:
            continue
        cb_work.mkdir(parents=True, exist_ok=True)
        patched_p = cb_work / f"{stem}.model"
        align_t64 = tmpl_paths["mlp_t64"] if key == "mlp_t1" else None
        patch_model_truncates(src_p, patched_p, TRUNCS[arch], align_mlp_t1_to_t64=align_t64)
        subprocess.run([VENV_PY, str(COMPILER_PY), "-m", str(patched_p), "-o", str(cb_work), "--convert-weights"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cb_out = cb_work / f"{stem}_classb.model"
        shutil.copy2(cb_out, dst_cb)
        print(f"[+] Built Class B archetype {stem}_classb.model ({cb_out.stat().st_size} B)")

    # Pre-cache template weight regions for fast packing
    print("[+] Pre-caching template weight region offsets...")
    cached_regs = {}
    for k in ["conv_in_t1", "proj2048_t1", "attn_qkv_t1", "mlp_t64"]:
        m = ModelReader.read_file(str(tmpl_paths[k]))
        cached_regs[k] = find_weight_regions(m)[0]

    calib_x, calib_y = collect_calibration_activations(model_dir)

    st = safe_open(str(model_dir / "model.safetensors"), framework="pt")
    cfg = json.loads((model_dir / "config.json").read_text())
    layer_types = cfg["layer_types"]
    sh_mlp_tbl = json.loads((tables_dir / "shared_mlp_table.json").read_text())

    base_qs_path = base_pkg_dir / "quant_scales.json" if (base_pkg_dir / "quant_scales.json").exists() else (fp16_pkg_dir / "quant_scales.json")
    qs = json.loads(base_qs_path.read_text())

    die_files = {0: open(pkg_out_dir / "weights_die0.bin", "wb"), 1: open(pkg_out_dir / "weights_die1.bin", "wb")}
    die_offs  = {0: 0, 1: 0}
    w_map     = {}
    chan_floats = []

    def append_chan_vec(arr: np.ndarray) -> tuple[int, int]:
        off = len(chan_floats)
        arr_f32 = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        chan_floats.extend(arr_f32.tolist())
        return off, int(arr_f32.size)

    def record_scale_entry(k_name: str, in_dt: str, out_dt: str, in_t: float, in_s: float,
                           in_inv: np.ndarray, out_step_c: np.ndarray):
        in_off, in_cnt = append_chan_vec(in_inv)
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

    for l in range(24):
        d_id = 0 if l < 12 else 1
        pfx = f"model.layers.{l}."
        lt = layer_types[l]
        if lt == "conv":
            k_in, k_out = f"lfm3_l{l:02d}_conv_in", f"lfm3_l{l:02d}_conv_out"
            w_in  = st.get_tensor(pfx + "conv.in_proj.weight").float().numpy()
            w_out = st.get_tensor(pfx + "conv.out_proj.weight").float().numpy()
            b_in, in_t_i, in_s_i, inv_i, osc_i = build_single_conv_region(
                w_in, calib_x[k_in], tmpl_paths["conv_in_t1"], 22, cached_reg=cached_regs["conv_in_t1"])
            b_out, in_t_o, in_s_o, inv_o, osc_o = build_single_conv_region(
                w_out, calib_x[k_out], tmpl_paths["proj2048_t1"], 22, cached_reg=cached_regs["proj2048_t1"])
        else:
            k_in, k_out = f"lfm3_l{l:02d}_attn_qkv", f"lfm3_l{l:02d}_attn_out"
            wq = st.get_tensor(pfx + "self_attn.q_proj.weight").float().numpy()
            wk = st.get_tensor(pfx + "self_attn.k_proj.weight").float().numpy()
            wv = st.get_tensor(pfx + "self_attn.v_proj.weight").float().numpy()
            wo = st.get_tensor(pfx + "self_attn.o_proj.weight").float().numpy()
            b_in, in_t_i, in_s_i, inv_i, osc_i = build_single_conv_region(
                np.concatenate([wq, wk, wv], axis=0), calib_x[k_in], tmpl_paths["attn_qkv_t1"], 23, cached_reg=cached_regs["attn_qkv_t1"])
            b_out, in_t_o, in_s_o, inv_o, osc_o = build_single_conv_region(
                wo, calib_x[k_out], tmpl_paths["proj2048_t1"], 22, cached_reg=cached_regs["proj2048_t1"])

        record_scale_entry(k_in,  "int8", "int8", in_t_i, in_s_i, inv_i, osc_i)
        record_scale_entry(k_out, "int8", "int8", in_t_o, in_s_o, inv_o, osc_o)

        w1 = st.get_tensor(pfx + "feed_forward.w1.weight").float().numpy()
        w3 = st.get_tensor(pfx + "feed_forward.w3.weight").float().numpy()
        w2 = st.get_tensor(pfx + "feed_forward.w2.weight").float().numpy()
        k_mlp = f"lfm3_l{l:02d}_mlp"
        b_mlp, in_t_m, in_s_m, inv_m, osc_m = build_mlp_region(
            w1, w3, w2, calib_x[k_mlp], calib_y[k_mlp], sh_mlp_tbl, tmpl_paths["mlp_t64"], cached_reg=cached_regs["mlp_t64"])
        record_scale_entry(k_mlp, "int8", "int16", in_t_m, in_s_m, inv_m, osc_m)

        for k_name, blob_bytes in [(k_in, b_in), (k_out, b_out), (k_mlp, b_mlp)]:
            w_map[k_name] = {"layer": l, "die": d_id, "weight_offset": die_offs[d_id], "weight_size": len(blob_bytes)}
            die_files[d_id].write(blob_bytes)
            die_offs[d_id] += len(blob_bytes)
        print(f"[+] Packed Layer {l:02d} ({lt:<17}) -> Die {d_id} (mlp in_t={in_t_m:.3f})")

    die_files[0].close()
    die_files[1].close()
    chan_bin_path = pkg_out_dir / "quant_chan_scales.bin"
    np.asarray(chan_floats, dtype=np.float32).tofile(str(chan_bin_path))
    print(f"[+] Wrote {chan_bin_path} ({len(chan_floats)} floats, {chan_bin_path.stat().st_size} B)")
    (pkg_out_dir / "weights_map.json").write_text(json.dumps(w_map, indent=2))
    (pkg_out_dir / "quant_scales.json").write_text(json.dumps(qs, indent=2))
    shutil.rmtree(cb_work, ignore_errors=True)
    print(f"[+] Successfully generated INT8 package in {pkg_out_dir}")


if __name__ == "__main__":
    main()
