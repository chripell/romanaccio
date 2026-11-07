#!/usr/bin/env python3
"""
Milestone 7: 2-Way Tensor Parallelism (TP=2) Package Builder for LFM3-1.7B.

Synthesizes the Hybrid TP=2 package:
  - 8 Shared Class B Archetype Loadables in models/:
      * lfm3_conv_in_{t1,t64}_classb.model   (full-width 2048 -> 6144, PP=2 on Die 0/1)
      * lfm3_proj2048_{t1,t64}_classb.model  (full-width 2048 -> 2048, PP=2 on Die 0/1)
      * lfm3_attn_qkv_{t1,t64}_classb.model  (full-width 2048 -> 3072, PP=2 on Die 0/1)
      * lfm3_mlp_{t1,t64}_classb.model       (TP=2 2048 -> 3840 -> 2048 per die, intermediate-split)
  - Symmetric 4K-aligned per-die weight blobs (weights_die0.bin = 0.908 GiB, weights_die1.bin = 0.908 GiB)
    where every layer's 64 MB SwiGLU MLP (79.1% of layer weights) is sharded 50/50 across Die 0 and Die 1
    so both LPDDR5 controllers stream MLP weights concurrently at 48.66 GB/s aggregate driver bandwidth,
    while the small pointwise projections run single-die to avoid kernel cross-die submit serialization.
  - Per-channel scale sidecar (quant_chan_scales.bin) + quant_scales.json + manifest.json ("tp_degree": 2)
  - 8 x 16,000-channel INT8 LM Head slices (lfm3_lm_head_d{0,1}_s{0..3}.model, 33.23 MiB each)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))
from backend.flatbuffer_emitter import ModelReader, ModelWriter
from backend.dynamic_weight_binding import find_weight_regions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lfm3_int8_pkg import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TABLES_DIR,
    DEFAULT_TEMPL_DIR,
    PROBE_DIR,
    DEFAULT_PKG_FP16,
    DEFAULT_PKG_INT8,
    COMPILER_PY,
    VENV_PY,
    SDP_SIG,
    tile_int8_slice,
    tile_int16_slice,
    collect_calibration_activations,
)

DEFAULT_PKG_TP2 = DEFAULT_MODEL_DIR / "eswin_pkg_tp2"

TRUNCS_TP2_MLP = [14] * 15 + [15] * 15 + [25] * 16


def patch_tp2_model_truncates(src_model: Path, dst_model: Path, new_truncs: list[int]):
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


def build_tp2_mlp_regions(
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    calib_x: torch.Tensor,
    tbl: dict,
    reg,
) -> tuple[bytes, bytes, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds Die 0 and Die 1 SwiGLU MLP TP=2 Class B weight regions (2048 -> 3840 -> 2048 per die):
      - w1_d (15 slices of 256 channels = 3840)
      - w3_d (15 slices of 256 channels = 3840)
      - w2_d (16 slices of 128 channels = 2048, K = 3840)
    Returns (buf0_bytes, buf1_bytes, in_thresh, in_step, in_inv_smooth_f32[2048], out_step_chan_f32[4096], out_s_c_f64[2048]).
    """
    buf0 = bytearray(reg.size)
    buf1 = bytearray(reg.size)

    c0_s = float(tbl["blob_info"]["/Conv_output_0"]["int16_step"])
    c1_s = float(tbl["blob_info"]["/Conv_1_output_0"]["int16_step"])
    m1_s = float(tbl["blob_info"]["/Mul_1_output_0"]["int16_step"])

    w1_t = torch.from_numpy(w1).float()
    w3_t = torch.from_numpy(w3).float()
    w2_t = torch.from_numpy(w2).float()
    x = calib_x.float()

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

    with torch.no_grad():
        h_swiglu = F.silu(F.linear(x, w1_t)) * F.linear(x, w3_t)
        y_d0 = F.linear(h_swiglu[:, :3840], w2_t[:, :3840])
        y_d1 = F.linear(h_swiglu[:, 3840:], w2_t[:, 3840:])

    out_step_chan_both = np.zeros(4096, dtype=np.float32)

    for d, (buf, y_part) in enumerate(((buf0, y_d0), (buf1, y_d1))):
        k_base = d * 3840
        w1_d = w1_sc_np[k_base:k_base + 3840, :]
        w3_d = w3_sc_np[k_base:k_base + 3840, :]
        w2_d = w2[:, k_base:k_base + 3840]
        w2_d_t = w2_t[:, k_base:k_base + 3840]

        for s in range(15):
            sl = slice(s * 256, (s + 1) * 256)
            wb, ws = tile_int8_slice(w1_d[sl], max_ws=max_ws1)
            qm = np.clip(np.round((in_s * ws / c0_s) * (1 << 14)), 1.0, 32767.0).astype(np.int16).tobytes()
            buf[reg.offsets[2*s]:reg.offsets[2*s]+len(wb)] = wb
            buf[reg.offsets[2*s+1]:reg.offsets[2*s+1]+len(qm)] = qm

        for s in range(15):
            idx = 30 + 2 * s
            sl = slice(s * 256, (s + 1) * 256)
            wb, ws = tile_int8_slice(w3_d[sl], max_ws=max_ws3)
            qm = np.clip(np.round((in_s * ws / c1_s) * (1 << 15)), 1.0, 32767.0).astype(np.int16).tobytes()
            buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
            buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm

        u2_c = w2_d_t.double().norm(dim=1).clamp(min=1e-8)
        y_max_c = y_part.abs().amax(dim=0).double()
        q98 = torch.quantile(y_max_c / u2_c, 0.98)
        y_bound_c = torch.maximum(y_max_c * 1.65, q98 * u2_c * 1.65)
        out_s_c = (y_bound_c / 32767.0).numpy()

        osc_d = out_step_chan_both[d * 2048:(d + 1) * 2048]
        for s in range(16):
            idx = 60 + 2 * s
            sl = slice(s * 128, (s + 1) * 128)
            wb, ws = tile_int16_slice(w2_d[sl])
            qm_f = np.clip(np.round((m1_s * ws / out_s_c[sl]) * (1 << 25)), 1.0, 32767.0)
            osc_d[sl] = ((m1_s * ws * (1 << 25)) / qm_f).astype(np.float32)
            qm = qm_f.astype(np.int16).tobytes()
            buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
            buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm

    return bytes(buf0), bytes(buf1), in_t, in_s, in_inv_smooth, out_step_chan_both, out_s_c


def main():
    parser = argparse.ArgumentParser(description="Milestone 7: 2-Way Tensor Parallel (TP=2) INT8/INT16 Package Builder for LFM3-1.7B")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to HF model directory")
    parser.add_argument("--pkg-out-dir", type=Path, default=DEFAULT_PKG_TP2, help="Output directory for TP=2 INT8 package")
    parser.add_argument("--base-pkg-dir", type=Path, default=DEFAULT_PKG_INT8, help="Base PP=2 INT8 package directory")
    parser.add_argument("--fp16-pkg-dir", type=Path, default=DEFAULT_PKG_FP16, help="Base FP16 package directory")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Quant tables directory")
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPL_DIR, help="Archetype probe templates directory")
    parser.add_argument("--dump-out-s-c", type=Path, default=None, help="Optional path to save [24, 2048] float64 MLP out_s_c array")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    pkg_out_dir = args.pkg_out_dir.resolve()
    base_pkg_dir = args.base_pkg_dir.resolve()
    fp16_pkg_dir = args.fp16_pkg_dir.resolve()
    tables_dir = args.tables_dir.resolve()
    templates_dir = args.templates_dir.resolve()

    (pkg_out_dir / "models").mkdir(parents=True, exist_ok=True)

    # 1. Copy static runtime files
    for fname in ["cpu_weights.bin", "cpu_weights.json", "manifest.json"]:
        for src_dir in [base_pkg_dir, fp16_pkg_dir]:
            if (src_dir / fname).exists():
                shutil.copy2(src_dir / fname, pkg_out_dir / fname)
                break

    if (model_dir / "tokenizer.json").exists():
        shutil.copy2(model_dir / "tokenizer.json", pkg_out_dir / "tokenizer.json")
    elif (base_pkg_dir / "tokenizer.json").exists():
        shutil.copy2(base_pkg_dir / "tokenizer.json", pkg_out_dir / "tokenizer.json")

    for src_dir in [base_pkg_dir, fp16_pkg_dir]:
        emb = src_dir / "embedding.bin"
        if emb.exists() and not (pkg_out_dir / "embedding.bin").exists():
            try:
                os.link(emb, pkg_out_dir / "embedding.bin")
            except OSError:
                shutil.copy2(emb, pkg_out_dir / "embedding.bin")
            break

    if (pkg_out_dir / "manifest.json").exists():
        manifest = json.loads((pkg_out_dir / "manifest.json").read_text())
        manifest["precision"] = "int8"
        manifest["tp_degree"] = 2
        manifest["model_name"] = model_dir.name
        (pkg_out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 2. Reuse full-width pointwise Class B archetypes + 8 LM head slices from base_pkg_dir
    for pat in ["lfm3_lm_head_*.model", "lfm3_conv_in_*_classb.model",
                "lfm3_proj2048_*_classb.model", "lfm3_attn_qkv_*_classb.model"]:
        for m_file in sorted((base_pkg_dir / "models").glob(pat)):
            dst_m = pkg_out_dir / "models" / m_file.name
            if dst_m.exists():
                dst_m.unlink()
            try:
                os.link(m_file, dst_m)
            except OSError:
                shutil.copy2(m_file, dst_m)
    print(f"[+] Linked pointwise Class B archetypes and 8 LM head slices from {base_pkg_dir / 'models'}")

    # 3. Build the TP=2 MLP Class B archetype loadables (2048 -> 3840 -> 2048 per die)
    tmpl_paths = {
        "mlp_t1":  templates_dir / "mlp_tp2_t1_npu_b1.model",
        "mlp_t64": templates_dir / "mlp_tp2_t64_npu_b1.model",
    }
    if (templates_dir / "mlp_tp2_t8_npu_b1.model").exists():
        tmpl_paths["mlp_t8"] = templates_dir / "mlp_tp2_t8_npu_b1.model"

    cb_work = PROBE_DIR / "cb_work_tp2"
    cb_work.mkdir(parents=True, exist_ok=True)
    for key, src_p in tmpl_paths.items():
        if not src_p.exists():
            raise FileNotFoundError(f"Missing TP=2 template model: {src_p}")
        stem = f"lfm3_{key}"
        dst_cb = pkg_out_dir / "models" / f"{stem}_classb.model"
        patched_p = cb_work / f"{stem}.model"
        patch_tp2_model_truncates(src_p, patched_p, TRUNCS_TP2_MLP)
        subprocess.run([VENV_PY, str(COMPILER_PY), "-m", str(patched_p), "-o", str(cb_work), "--convert-weights"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cb_out = cb_work / f"{stem}_classb.model"
        if dst_cb.exists():
            dst_cb.unlink()
        shutil.copy2(cb_out, dst_cb)
        print(f"[+] Built TP=2 Class B archetype {stem}_classb.model ({cb_out.stat().st_size} B)")

    m_mlp_tmpl = ModelReader.read_file(str(tmpl_paths["mlp_t1"]))
    mlp_reg = find_weight_regions(m_mlp_tmpl)[0]

    calib_x, _ = collect_calibration_activations(model_dir)

    st = safe_open(str(model_dir / "model.safetensors"), framework="pt")
    cfg = json.loads((model_dir / "config.json").read_text())
    layer_types = cfg["layer_types"]
    sh_mlp_tbl = json.loads((tables_dir / "shared_mlp_table.json").read_text())

    base_w_map = json.loads((base_pkg_dir / "weights_map.json").read_text())
    base_qs = json.loads((base_pkg_dir / "quant_scales.json").read_text())
    base_chan = np.fromfile(str(base_pkg_dir / "quant_chan_scales.bin"), dtype=np.float32)
    base_die_files = {
        0: open(base_pkg_dir / "weights_die0.bin", "rb"),
        1: open(base_pkg_dir / "weights_die1.bin", "rb"),
    }

    qs = json.loads(json.dumps(base_qs))
    qs["tp_degree"] = 2

    die_files = {0: open(pkg_out_dir / "weights_die0.bin", "wb"), 1: open(pkg_out_dir / "weights_die1.bin", "wb")}
    die_offs  = {0: 0, 1: 0}
    w_map     = {"tp_degree": 2}
    chan_floats = []
    all_out_s_c = np.zeros((24, 2048), dtype=np.float64)

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
        pfx = f"model.layers.{l}."
        lt = layer_types[l]
        d_pp = 0 if l < 12 else 1
        if lt == "conv":
            k_in, k_out = f"lfm3_l{l:02d}_conv_in", f"lfm3_l{l:02d}_conv_out"
        else:
            k_in, k_out = f"lfm3_l{l:02d}_attn_qkv", f"lfm3_l{l:02d}_attn_out"

        # Reuse exact PP=2 weight regions and per-channel scales for k_in and k_out
        for k_op in (k_in, k_out):
            bm = base_w_map[k_op]
            base_die_files[d_pp].seek(bm["weight_offset"])
            op_bytes = base_die_files[d_pp].read(bm["weight_size"])
            w_map[k_op] = {
                "layer": l,
                "die": d_pp,
                "tp_degree": 1,
                "weight_offset": die_offs[d_pp],
                "weight_size": len(op_bytes),
            }
            die_files[d_pp].write(op_bytes)
            die_offs[d_pp] += len(op_bytes)

            b_sc = base_qs["subgraphs"][k_op]
            inv_op = base_chan[b_sc["in_smooth_offset"]:b_sc["in_smooth_offset"] + b_sc["in_smooth_count"]]
            osc_op = base_chan[b_sc["out_step_offset"]:b_sc["out_step_offset"] + b_sc["out_step_count"]]
            record_scale_entry(k_op, b_sc["input_dtype"], b_sc["output_dtype"],
                               b_sc["input_threshold"], b_sc["input_step"], inv_op, osc_op)

        # Build TP=2 MLP regions across both Die 0 and Die 1
        w1 = st.get_tensor(pfx + "feed_forward.w1.weight").float().numpy()
        w3 = st.get_tensor(pfx + "feed_forward.w3.weight").float().numpy()
        w2 = st.get_tensor(pfx + "feed_forward.w2.weight").float().numpy()
        k_mlp = f"lfm3_l{l:02d}_mlp"
        b_mlp0, b_mlp1, in_t_m, in_s_m, inv_m, osc_m, out_s_c = build_tp2_mlp_regions(
            w1, w3, w2, calib_x[k_mlp], sh_mlp_tbl, mlp_reg)
        all_out_s_c[l] = out_s_c
        record_scale_entry(k_mlp, "int8", "int16", in_t_m, in_s_m, inv_m, osc_m)

        w_map[k_mlp] = {
            "layer": l,
            "die": -1,
            "tp_degree": 2,
            "weight_offset": die_offs[0],
            "weight_offset_d1": die_offs[1],
            "weight_size": len(b_mlp0),
        }
        die_files[0].write(b_mlp0)
        die_files[1].write(b_mlp1)
        die_offs[0] += len(b_mlp0)
        die_offs[1] += len(b_mlp1)
        print(f"[+] Packed Hybrid TP=2 Layer {l:02d} ({lt:<17}) -> Die0={die_offs[0]/(1024*1024):.1f} MiB, Die1={die_offs[1]/(1024*1024):.1f} MiB")

    base_die_files[0].close()
    base_die_files[1].close()
    die_files[0].close()
    die_files[1].close()

    if args.dump_out_s_c is not None:
        np.save(str(args.dump_out_s_c), all_out_s_c)
        print(f"[+] Saved {args.dump_out_s_c} ({all_out_s_c.shape})")

    chan_bin_path = pkg_out_dir / "quant_chan_scales.bin"
    np.asarray(chan_floats, dtype=np.float32).tofile(str(chan_bin_path))
    print(f"[+] Wrote {chan_bin_path} ({len(chan_floats)} floats, {chan_bin_path.stat().st_size} B)")
    (pkg_out_dir / "weights_map.json").write_text(json.dumps(w_map, indent=2))
    (pkg_out_dir / "quant_scales.json").write_text(json.dumps(qs, indent=2))
    shutil.rmtree(cb_work, ignore_errors=True)
    print(f"[+] Successfully generated Hybrid TP=2 INT8 package in {pkg_out_dir} (Die0={die_offs[0]/(1024**3):.3f} GiB, Die1={die_offs[1]/(1024**3):.3f} GiB)")


if __name__ == "__main__":
    main()
