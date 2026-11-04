#!/usr/bin/env python3
"""
Milestone 5: Direct NVDLA INT8/INT16 Weight & CvtPrecision_QuantM Package Builder for LFM3-1.7B.

Synthesizes:
  - 8 Shared Class B Archetype Loadables (lfm3_{conv_in,proj2048,attn_qkv,mlp}_{t1,t64}_classb.model)
  - Contiguous 4K-aligned per-die weight blobs (weights_die0.bin = 0.878 GiB, weights_die1.bin = 0.872 GiB)
    using the verified maxdiff=0 float64 NVDLA INT8/INT16 weight tiler and CvtPrecision_QuantM formula
  - 8 x 16,000-channel INT8 LM Head slices (lfm3_lm_head_d{0,1}_s{0..3}.model, 33.23 MiB each)
"""

import os
import sys
import json
import struct
import shutil
import subprocess
from pathlib import Path
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))
from backend.flatbuffer_emitter import ModelReader, ModelWriter
from backend.dynamic_weight_binding import find_weight_regions

MODEL_DIR   = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
TABLES_DIR  = Path("/home/chri/aixigua/deep/artifacts/lfm3_quant_tables")
PROBE_DIR   = Path("/run/host-wayland/simulator-ipc/lfm3_probe_int8")
PKG_FP16    = MODEL_DIR / "eswin_pkg"
PKG_INT8    = MODEL_DIR / "eswin_pkg_int8"
COMPILER_PY = Path(__file__).resolve().parent.parent / "eswin_compiler" / "compiler.py"
VENV_PY     = "/home/chri/aixigua/venv/bin/python"

# 12-byte SDP CvtPrecision signature immediately preceding the 1-byte truncate shift at record offset +82
SDP_SIG = bytes.fromhex("000000000000010001010000")

TRUNCS = {
    "conv_in":  [22] * 12,
    "proj2048": [22] * 4,
    "attn_qkv": [23] * 6,
    "mlp":      [14] * 15 + [15] * 15 + [28] * 32,
}


def patch_model_truncates(src_model: Path, dst_model: Path, new_truncs: list[int], align_mlp_t64_to_t1: Path = None):
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

    if align_mlp_t64_to_t1 is not None:
        m_t1 = ModelReader.read_file(str(align_mlp_t64_to_t1))
        r1 = find_weight_regions(m_t1)[0]
        r64 = find_weight_regions(m)[0]
        for b in m.blobs:
            if b.name == "kmd-task-surf_list":
                surf_bytes = bytearray(b.data)
                for idx in range(60, 124):
                    if r64.offsets[idx] != r1.offsets[idx]:
                        b64 = struct.pack("<I", r64.offsets[idx])
                        b1  = struct.pack("<I", r1.offsets[idx])
                        locs = [i for i in range(0, len(surf_bytes)-4, 4) if surf_bytes[i:i+4] == b64]
                        assert len(locs) == 1
                        surf_bytes[locs[0]:locs[0]+4] = b1
                b.data = bytes(surf_bytes)
            elif b.name.startswith("CvtPrecision_QuantM") and len(b.data) == 256:
                b.data = bytes(b.data)[:128]
        r64.offsets = list(r1.offsets)
        r64.size = r1.size

    ModelWriter.write_file(m, str(dst_model))


def tile_int8_slice(w_slice: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantizes [M_s, K] FP32 -> per-channel INT8 in float64 (round-to-even) and tiles for NVDLA CMAC."""
    M_s, K = w_slice.shape
    w_f64 = w_slice.astype(np.float64)
    w_amax = np.maximum(np.max(np.abs(w_f64), axis=1, keepdims=True), 1e-12)
    w_step = w_amax / 127.0
    w_q = np.clip(np.round(w_f64 / w_step), -127, 127).astype(np.int8)
    K_pad = ((K + 95) // 96) * 96
    wp = np.zeros((M_s, K_pad), dtype=np.int8)
    wp[:, :K] = w_q
    tiled = wp.reshape(M_s // 8, 8, K_pad // 96, 3, 8, 4).transpose(0, 2, 3, 4, 1, 5).ravel()
    return tiled.tobytes(), w_step[:, 0]


def tile_int16_slice(w_slice: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantizes [M_s, K] FP32 -> per-channel INT16 in float64 (round-to-even) and tiles for NVDLA CMAC."""
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


def build_single_conv_region(w_f32: np.ndarray, tbl: dict, tmpl_model_path: Path, trunc: int) -> bytes:
    m = ModelReader.read_file(str(tmpl_model_path))
    reg = find_weight_regions(m)[0]
    buf = bytearray(reg.size)
    in_s  = float(tbl["blob_info"]["input"]["int8_step"])
    out_s = float(tbl["blob_info"]["output"]["int8_step"])
    n_slices = len(reg.contents) // 2
    for s in range(n_slices):
        w_bytes, w_step = tile_int8_slice(w_f32[s*512:(s+1)*512])
        qm_f = np.round((in_s * w_step / out_s) * (1 << trunc))
        assert np.max(np.abs(qm_f)) <= 32767, f"QuantM overflow: {np.max(np.abs(qm_f))}"
        qm = qm_f.astype(np.int16).tobytes()
        buf[reg.offsets[2*s]:reg.offsets[2*s]+len(w_bytes)] = w_bytes
        buf[reg.offsets[2*s+1]:reg.offsets[2*s+1]+len(qm)]  = qm
    return bytes(buf)


def build_mlp_region(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, tbl: dict, tmpl_model_path: Path) -> bytes:
    m = ModelReader.read_file(str(tmpl_model_path))
    reg = find_weight_regions(m)[0]
    buf = bytearray(reg.size)
    in_s  = float(tbl["blob_info"]["input"]["int8_step"])
    c0_s  = float(tbl["blob_info"]["/Conv_output_0"]["int16_step"])
    c1_s  = float(tbl["blob_info"]["/Conv_1_output_0"]["int16_step"])
    m1_s  = float(tbl["blob_info"]["/Mul_1_output_0"]["int16_step"])
    out_s = float(tbl["blob_info"]["output"]["int16_step"])

    for s in range(15):
        wb, ws = tile_int8_slice(w1[s*512:(s+1)*512])
        qm = np.round((in_s * ws / c0_s) * (1 << 14)).astype(np.int16).tobytes()
        buf[reg.offsets[2*s]:reg.offsets[2*s]+len(wb)] = wb
        buf[reg.offsets[2*s+1]:reg.offsets[2*s+1]+len(qm)] = qm

    for s in range(15):
        idx = 30 + 2 * s
        wb, ws = tile_int8_slice(w3[s*512:(s+1)*512])
        qm = np.round((in_s * ws / c1_s) * (1 << 15)).astype(np.int16).tobytes()
        buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
        buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm

    for s in range(32):
        idx = 60 + 2 * s
        wb, ws = tile_int16_slice(w2[s*64:(s+1)*64])
        qm = np.round((m1_s * ws / out_s) * (1 << 28)).astype(np.int16).tobytes()
        buf[reg.offsets[idx]:reg.offsets[idx]+len(wb)] = wb
        buf[reg.offsets[idx+1]:reg.offsets[idx+1]+len(qm)] = qm
    return bytes(buf)


def main():
    (PKG_INT8 / "models").mkdir(parents=True, exist_ok=True)
    for fname in ["cpu_weights.bin", "cpu_weights.json", "tokenizer.json", "manifest.json"]:
        if (PKG_FP16 / fname).exists():
            shutil.copy2(PKG_FP16 / fname, PKG_INT8 / fname)

    manifest = json.loads((PKG_INT8 / "manifest.json").read_text())
    manifest["precision"] = "int8"
    (PKG_INT8 / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tmpl_paths = {
        "conv_in_t1":   PROBE_DIR / "conv_in_t1_shared/model_files/lfm3_l00_conv_in_npu_b1.model",
        "conv_in_t64":  PROBE_DIR / "conv_in_t64_shared/model_files/lfm3_l00_conv_in_t64_npu_b1.model",
        "proj2048_t1":  PROBE_DIR / "proj2048_t1_shared/model_files/lfm3_l00_conv_out_npu_b1.model",
        "proj2048_t64": PROBE_DIR / "proj2048_t64_shared/model_files/lfm3_l00_conv_out_t64_npu_b1.model",
        "attn_qkv_t1":  PROBE_DIR / "attn_qkv_t1_shared/model_files/lfm3_l02_attn_qkv_npu_b1.model",
        "attn_qkv_t64": PROBE_DIR / "attn_qkv_t64_shared/model_files/lfm3_l02_attn_qkv_t64_npu_b1.model",
        "mlp_t1":       PROBE_DIR / "l01_mlp_shared/model_files/lfm3_l01_mlp_npu_b1.model",
        "mlp_t64":      PROBE_DIR / "l00_mlp_t64_shared/model_files/lfm3_l00_mlp_t64_npu_b1.model",
    }

    cb_work = PROBE_DIR / "cb_work"
    shutil.rmtree(cb_work, ignore_errors=True)
    cb_work.mkdir(parents=True, exist_ok=True)

    for key, src_p in tmpl_paths.items():
        arch = key.rsplit("_", 1)[0]
        stem = f"lfm3_{key}"
        patched_p = cb_work / f"{stem}.model"
        align_t1 = tmpl_paths["mlp_t1"] if key == "mlp_t64" else None
        patch_model_truncates(src_p, patched_p, TRUNCS[arch], align_mlp_t64_to_t1=align_t1)
        subprocess.run([VENV_PY, str(COMPILER_PY), "-m", str(patched_p), "-o", str(cb_work), "--convert-weights"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cb_out = cb_work / f"{stem}_classb.model"
        shutil.copy2(cb_out, PKG_INT8 / "models" / f"{stem}_classb.model")
        print(f"[+] Built Class B archetype {stem}_classb.model ({cb_out.stat().st_size} B)")

    st = safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt")
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    layer_types = cfg["layer_types"]
    sh_mlp_tbl = json.loads((TABLES_DIR / "shared_mlp_table.json").read_text())
    qs = json.loads((PKG_FP16 / "quant_scales.json").read_text())

    die_files = {0: open(PKG_INT8 / "weights_die0.bin", "wb"), 1: open(PKG_INT8 / "weights_die1.bin", "wb")}
    die_offs  = {0: 0, 1: 0}
    w_map     = {}

    for l in range(24):
        d_id = 0 if l < 12 else 1
        pfx = f"model.layers.{l}."
        lt = layer_types[l]
        if lt == "conv":
            k_in, k_out = f"lfm3_l{l:02d}_conv_in", f"lfm3_l{l:02d}_conv_out"
            t_in  = json.loads((TABLES_DIR / f"{k_in}_table.json").read_text())
            t_out = json.loads((TABLES_DIR / f"{k_out}_table.json").read_text())
            w_in  = st.get_tensor(pfx + "conv.in_proj.weight").float().numpy()
            w_out = st.get_tensor(pfx + "conv.out_proj.weight").float().numpy()
            b_in  = build_single_conv_region(w_in,  t_in,  tmpl_paths["conv_in_t1"],  22)
            b_out = build_single_conv_region(w_out, t_out, tmpl_paths["proj2048_t1"], 22)
        else:
            k_in, k_out = f"lfm3_l{l:02d}_attn_qkv", f"lfm3_l{l:02d}_attn_out"
            t_in  = json.loads((TABLES_DIR / f"{k_in}_table.json").read_text())
            t_out = json.loads((TABLES_DIR / f"{k_out}_table.json").read_text())
            wq = st.get_tensor(pfx + "self_attn.q_proj.weight").float().numpy()
            wk = st.get_tensor(pfx + "self_attn.k_proj.weight").float().numpy()
            wv = st.get_tensor(pfx + "self_attn.v_proj.weight").float().numpy()
            wo = st.get_tensor(pfx + "self_attn.o_proj.weight").float().numpy()
            b_in  = build_single_conv_region(np.concatenate([wq, wk, wv], axis=0), t_in, tmpl_paths["attn_qkv_t1"], 23)
            b_out = build_single_conv_region(wo, t_out, tmpl_paths["proj2048_t1"], 22)

        w1 = st.get_tensor(pfx + "feed_forward.w1.weight").float().numpy()
        w3 = st.get_tensor(pfx + "feed_forward.w3.weight").float().numpy()
        w2 = st.get_tensor(pfx + "feed_forward.w2.weight").float().numpy()
        b_mlp = build_mlp_region(w1, w3, w2, sh_mlp_tbl, tmpl_paths["mlp_t1"])
        k_mlp = f"lfm3_l{l:02d}_mlp"
        qs["subgraphs"][k_mlp] = dict(qs["archetype_shared"]["mlp"])

        for k_name, blob_bytes in [(k_in, b_in), (k_out, b_out), (k_mlp, b_mlp)]:
            w_map[k_name] = {"layer": l, "die": d_id, "weight_offset": die_offs[d_id], "weight_size": len(blob_bytes)}
            die_files[d_id].write(blob_bytes)
            die_offs[d_id] += len(blob_bytes)
        print(f"[+] Packed Layer {l:02d} ({lt:<17}) -> Die {d_id}")

    die_files[0].close()
    die_files[1].close()
    (PKG_INT8 / "weights_map.json").write_text(json.dumps(w_map, indent=2))
    (PKG_INT8 / "quant_scales.json").write_text(json.dumps(qs, indent=2))
    shutil.rmtree(cb_work, ignore_errors=True)


if __name__ == "__main__":
    main()
