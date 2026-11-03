#!/usr/bin/env python3
"""
Milestone 3.5: Batched (T=64) Prefill Subgraph Exporter, Class B Archetype Compiler,
and Shared Per-Die Weight Blob Packer for LFM3-1.7B.

Key Silicon-Verified Invariants (FINDINGS.md §36 / §37):
1. Across all 24 layers of LFM3-1.7B in FP16, EsAAC emits 100% byte-identical KMD/UMD
   instruction streams across layers of the same subgraph archetype (conv_in, proj2048
   [conv_out & attn_out], attn_qkv, mlp).
2. Furthermore, the NVDLA-tiled weight blobs (weights_mem1.bin) inside each layer's
   T=1 and T=64 .model files are 100% bit-identical (verified by MD5 and on-board
   probe_weight_bitmatch on roma /dev/npu0).
3. Therefore, this script produces:
   a) 8 Shared Class B Archetype Loadables (4 for T=1 decode, 4 for T=64 prefill) in
      eswin_pkg/models/ (only 8 layer loadables + 4 LM head loadables = 12 models/die,
      50 models below the 62-model kernel Oops ceiling, with zero model swapping!).
   b) 2 Contiguous 4K-Aligned Per-Die Weight Blobs (eswin_pkg/weights_die0.bin and
      eswin_pkg/weights_die1.bin) + eswin_pkg/weights_map.json for zero-copy DmaBufferSlice
      binding shared by both T=1 decode and T=64 prefill.
   c) All 72 Static *_t64.model files in eswin_pkg/models/ (bit-exact to EsAAC output).
   d) Updated eswin_pkg/manifest.json.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))

from export_lfm3 import PointwiseConvBlock, Lfm3SwiGLUBlock, SAFETENSORS_PATH, CONFIG_PATH, PKG_DIR, ONNX_OUT_DIR
from backend.flatbuffer_emitter import ModelReader
from backend.dynamic_weight_binding import extract_weight_blob, find_weight_regions

MODELS_DIR = PKG_DIR / "models"
IPC_BASE = Path("/run/host-wayland/simulator-ipc")
IPC_WORK_HOST = IPC_BASE / "lfm3_prefill_work"
IPC_WORK_CONT = "/ipc/lfm3_prefill_work"
COMPILER_PY = Path(__file__).resolve().parent.parent / "eswin_compiler" / "compiler.py"
VENV_PY = "/home/chri/aixigua/venv/bin/python"

WEIGHT_BLOB_PREFIXES = ("w1splitM_", "w2splitM_", "w3splitM_", "weightsplitM_", "WsplitM_")


def is_weight_blob(name: str) -> bool:
    return name.startswith(WEIGHT_BLOB_PREFIXES)


def compile_t64_archetypes(T: int = 64) -> dict[str, Path]:
    """Exports and compiles the 4 canonical T=64 archetype ONNX models via EsAAC."""
    IPC_WORK_HOST.mkdir(parents=True, exist_ok=True)
    os.chmod(IPC_WORK_HOST, 0o777)

    dummy_x = torch.zeros(1, 2048, T, 1, dtype=torch.float32)
    archetype_specs = []

    with safe_open(SAFETENSORS_PATH, framework="pt") as st:
        # 1. conv_in (2048 -> 6144) from Layer 0
        w_in = st.get_tensor("model.layers.0.conv.in_proj.weight")
        archetype_specs.append(("conv_in", PointwiseConvBlock(w_in), "lfm3_l00_conv_in_t64"))

        # 2. proj2048 (2048 -> 2048, shared by conv_out and attn_out) from Layer 0 conv_out
        w_out = st.get_tensor("model.layers.0.conv.out_proj.weight")
        archetype_specs.append(("proj2048", PointwiseConvBlock(w_out), "lfm3_l00_conv_out_t64"))

        # 3. attn_qkv (2048 -> 3072) from Layer 2
        qkv_w = torch.cat([
            st.get_tensor("model.layers.2.self_attn.q_proj.weight"),
            st.get_tensor("model.layers.2.self_attn.k_proj.weight"),
            st.get_tensor("model.layers.2.self_attn.v_proj.weight"),
        ], dim=0)
        archetype_specs.append(("attn_qkv", PointwiseConvBlock(qkv_w), "lfm3_l02_attn_qkv_t64"))

        # 4. mlp (2048 -> 7680 -> 2048) from Layer 0
        w1 = st.get_tensor("model.layers.0.feed_forward.w1.weight")
        w3 = st.get_tensor("model.layers.0.feed_forward.w3.weight")
        w2 = st.get_tensor("model.layers.0.feed_forward.w2.weight")
        archetype_specs.append(("mlp", Lfm3SwiGLUBlock(w1, w3, w2), "lfm3_l00_mlp_t64"))

    compiled_templates = {}
    for arch_key, module, stem in archetype_specs:
        dst_template = MODELS_DIR / f"{stem}.model"
        onnx_path = ONNX_OUT_DIR / f"{stem}.onnx"
        module.eval()
        torch.onnx.export(
            module, dummy_x, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=13, dynamo=False, do_constant_folding=True
        )

        # Check if already compiled in probe_t64 or MODELS_DIR
        probe_cand = IPC_BASE / "lfm3_work" / "probe_t64" / "model_files" / f"{stem.replace('lfm3_', '')}_npu_b1.model"
        if dst_template.exists() and dst_template.stat().st_size > 10000:
            print(f"  [=] Reusing compiled T={T} template: {dst_template.name}")
            compiled_templates[arch_key] = dst_template
            continue
        elif probe_cand.exists() and probe_cand.stat().st_size > 10000:
            shutil.copyfile(probe_cand, dst_template)
            print(f"  [+] Copied pre-compiled T={T} template: {dst_template.name} ({dst_template.stat().st_size/(1024*1024):.2f} MiB)")
            compiled_templates[arch_key] = dst_template
            continue

        job_dir = IPC_WORK_HOST / stem
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(job_dir, 0o777)

        staged_onnx = job_dir / f"{stem}.onnx"
        shutil.copyfile(onnx_path, staged_onnx)
        os.chmod(staged_onnx, 0o666)

        t0 = time.time()
        cmd = [
            "ssh", "simulator",
            f"cd {IPC_WORK_CONT}/{stem} && /home/eswin/EsAAC --input-model={IPC_WORK_CONT}/{stem}/{stem}.onnx --gen-fp16-model=yes"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        out_cands = list(job_dir.rglob(f"{stem}_npu_b1.model"))
        assert out_cands, f"Failed to compile {stem}:\n{res.stdout}"
        shutil.copyfile(out_cands[0], dst_template)
        subprocess.run(["ssh", "simulator", f"rm -rf {IPC_WORK_CONT}/{stem}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(job_dir, ignore_errors=True)
        print(f"  [+] Compiled T={T} template {dst_template.name} in {time.time()-t0:.1f}s")
        compiled_templates[arch_key] = dst_template

    return compiled_templates


def build_classb_archetypes(compiled_t64_templates: dict[str, Path]) -> dict[str, dict]:
    """Generates the 8 shared Class B archetype .model files (4 for T=1, 4 for T=64)."""
    t1_sources = {
        "conv_in":  MODELS_DIR / "lfm3_l00_conv_in.model",
        "proj2048": MODELS_DIR / "lfm3_l00_conv_out.model",
        "attn_qkv": MODELS_DIR / "lfm3_l02_attn_qkv.model",
        "mlp":      MODELS_DIR / "lfm3_l00_mlp.model",
    }

    tmp_cb = IPC_WORK_HOST / "cb_build"
    if tmp_cb.exists():
        shutil.rmtree(tmp_cb, ignore_errors=True)
    tmp_cb.mkdir(parents=True, exist_ok=True)

    archetypes_info = {}
    for arch_key in ("conv_in", "proj2048", "attn_qkv", "mlp"):
        for tag, src_map in (("t1", t1_sources), ("t64", compiled_t64_templates)):
            src_model = src_map[arch_key]
            stem = f"lfm3_{arch_key}_{tag}"
            tmp_in = tmp_cb / f"{stem}.model"
            shutil.copyfile(src_model, tmp_in)
            subprocess.run(
                [VENV_PY, str(COMPILER_PY), "-m", str(tmp_in), "-o", str(tmp_cb), "--convert-weights"],
                check=True, capture_output=True, text=True
            )
            cb_src = tmp_cb / f"{stem}_classb.model"
            w_src = tmp_cb / f"{stem}_weights_mem1.bin"
            cb_dst = MODELS_DIR / f"{stem}_classb.model"
            shutil.copyfile(cb_src, cb_dst)

            archetypes_info[f"{arch_key}_{tag}"] = {
                "file": f"models/{cb_dst.name}",
                "size_bytes": cb_dst.stat().st_size,
                "weight_blob_size": w_src.stat().st_size,
                "md5": hashlib.md5(cb_dst.read_bytes()).hexdigest(),
            }
            print(f"  [+] Class B archetype {cb_dst.name:<34}: {cb_dst.stat().st_size:>8} B | weight_size={w_src.stat().st_size:>10} B")

    shutil.rmtree(tmp_cb, ignore_errors=True)
    return archetypes_info


def compute_template_blob_offsets(template_path: Path) -> tuple[bytearray, dict[str, tuple[int, int]]]:
    """Pre-computes the exact (offset, length) of every weight blob inside a template .model container."""
    raw = bytearray(template_path.read_bytes())
    parsed = ModelReader.read_file(str(template_path))
    offsets = {}
    for b in parsed.blobs:
        if is_weight_blob(b.name):
            pos = raw.find(b.data)
            assert pos != -1, f"Blob {b.name} not found in {template_path}"
            offsets[b.name] = (pos, len(b.data))
    return raw, offsets


def extract_ordered_weight_region(parsed_model) -> bytes:
    """Extracts the contiguous mem_id=1 weight region from a parsed static .model in exact blob order."""
    regions = find_weight_regions(parsed_model, [1])
    assert len(regions) == 1, f"Expected 1 weight region (id=1), got {len(regions)}"
    raw_weights = extract_weight_blob(parsed_model, regions[0])
    assert len(raw_weights) % 4096 == 0, f"Weight region size {len(raw_weights)} is not 4K-aligned"
    return raw_weights


def main():
    print("=================================================================")
    print(" LFM3-1.7B Milestone 3.5: T=64 Prefill & Shared Class B Builder")
    print("=================================================================\n")

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    layer_types = cfg["layer_types"]
    num_layers = cfg["num_hidden_layers"]

    print("[1/4] Exporting & Compiling Canonical T=64 Subgraph Archetypes...")
    t64_templates = compile_t64_archetypes(T=64)

    print("\n[2/4] Building 8 Shared Class B Archetype Loadables (T=1 & T=64)...")
    archetypes_info = build_classb_archetypes(t64_templates)

    print("\n[3/4] Synthesizing All 72 Static *_t64.model Loadables & Packing Per-Die Weight Blobs...")
    # Pre-load the 4 T=64 templates and their weight blob offsets
    template_cache = {
        arch_key: compute_template_blob_offsets(tpath)
        for arch_key, tpath in t64_templates.items()
    }

    die_weight_files = {
        0: open(PKG_DIR / "weights_die0.bin", "wb"),
        1: open(PKG_DIR / "weights_die1.bin", "wb"),
    }
    die_offsets = {0: 0, 1: 0}
    weights_map = {}
    prefill_models = {0: [], 1: []}

    t0 = time.time()
    for l_idx in range(num_layers):
        die_id = 0 if l_idx < 12 else 1
        l_type = layer_types[l_idx]

        if l_type == "conv":
            layer_ops = [
                ("conv_in",  "conv_in",  f"lfm3_l{l_idx:02d}_conv_in",  2048, 6144),
                ("conv_out", "proj2048", f"lfm3_l{l_idx:02d}_conv_out", 2048, 2048),
                ("mlp",      "mlp",      f"lfm3_l{l_idx:02d}_mlp",      2048, 2048),
            ]
        else:
            layer_ops = [
                ("attn_qkv", "attn_qkv", f"lfm3_l{l_idx:02d}_attn_qkv", 2048, 3072),
                ("attn_out", "proj2048", f"lfm3_l{l_idx:02d}_attn_out", 2048, 2048),
                ("mlp",      "mlp",      f"lfm3_l{l_idx:02d}_mlp",      2048, 2048),
            ]

        for op_role, arch_key, t1_stem, in_c, out_c in layer_ops:
            t1_path = MODELS_DIR / f"{t1_stem}.model"
            t64_stem = f"{t1_stem}_t64"
            t64_path = MODELS_DIR / f"{t64_stem}.model"

            parsed_t1 = ModelReader.read_file(str(t1_path))
            t1_blob_map = {b.name: b.data for b in parsed_t1.blobs}

            # 1. Synthesize static *_t64.model from template + layer's T=1 weight blobs
            tmpl_bytes, tmpl_offsets = template_cache[arch_key]
            out_bytes = bytearray(tmpl_bytes)
            for bname, (pos, length) in tmpl_offsets.items():
                w_data = t1_blob_map[bname]
                assert len(w_data) == length, f"Length mismatch on {bname} in {t1_stem}"
                out_bytes[pos:pos + length] = w_data
            t64_path.write_bytes(out_bytes)
            prefill_models[die_id].append(f"models/{t64_path.name}")

            # 2. Extract 4K-aligned weight region and append to weights_die{0,1}.bin
            w_region = extract_ordered_weight_region(parsed_t1)
            w_offset = die_offsets[die_id]
            die_weight_files[die_id].write(w_region)
            die_offsets[die_id] += len(w_region)

            weights_map[t1_stem] = {
                "layer": l_idx,
                "die": die_id,
                "op": op_role,
                "archetype": arch_key,
                "classb_t1_model": f"models/lfm3_{arch_key}_t1_classb.model",
                "classb_t64_model": f"models/lfm3_{arch_key}_t64_classb.model",
                "static_t1_model": f"models/{t1_stem}.model",
                "static_t64_model": f"models/{t64_stem}.model",
                "weight_blob_file": f"weights_die{die_id}.bin",
                "weight_offset": w_offset,
                "weight_size": len(w_region),
                "in_c": in_c,
                "out_c": out_c,
                "in_bytes_t1": in_c * 2,
                "out_bytes_t1": out_c * 2,
                "in_bytes_t64": in_c * 64 * 2,
                "out_bytes_t64": out_c * 64 * 2,
            }

        print(f"  [+] Layer {l_idx:02d} ({l_type:<17}): 3 static *_t64.model written & weights packed (Die {die_id})")

    die_weight_files[0].close()
    die_weight_files[1].close()
    dt = time.time() - t0
    print(f"\n  [+] Built 72 static *_t64.model files and packed per-die weights in {dt:.1f}s:")
    print(f"      - weights_die0.bin: {die_offsets[0] / (1024**3):.3f} GiB ({die_offsets[0]:,} bytes, 4K-aligned)")
    print(f"      - weights_die1.bin: {die_offsets[1] / (1024**3):.3f} GiB ({die_offsets[1]:,} bytes, 4K-aligned)")

    weights_map_path = PKG_DIR / "weights_map.json"
    with open(weights_map_path, "w") as f:
        json.dump(weights_map, f, indent=2)

    print("\n[4/4] Updating manifest.json with Prefill & Route D (Shared-Weight Class B) Metadata...")
    manifest_path = PKG_DIR / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest["prefill"] = {
        "chunk_tokens": 64,
        "input_layout": "NC/16HW16 (shape [1, C//16, 64, 16] FP16, stride_1=32B, stride_2=2048B)",
        "output_layout": "NCHW (shape [1, C_out, 64, 1] FP16, stride_1=2B, stride_2=128B)",
        "residency_route": "Route D (Shared-Weight Class B Archetypes: 4 T=1 + 4 T=64 + 4 LM-head = 12 resident models/die < 62)",
        "classb_archetypes": archetypes_info,
        "die_weight_blobs": {
            "0": {"file": "weights_die0.bin", "size_bytes": die_offsets[0]},
            "1": {"file": "weights_die1.bin", "size_bytes": die_offsets[1]},
        },
        "weights_map_file": "weights_map.json",
    }
    manifest["dies"]["0"]["prefill_models_t64"] = prefill_models[0]
    manifest["dies"]["1"]["prefill_models_t64"] = prefill_models[1]
    manifest["dies"]["0"]["classb_resident_model_count"] = 12
    manifest["dies"]["1"]["classb_resident_model_count"] = 12

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if IPC_WORK_HOST.exists():
        shutil.rmtree(IPC_WORK_HOST, ignore_errors=True)

    print(f"  [+] Updated {manifest_path}")
    print("\n=================================================================")
    print(" MILESTONE 3.5 ARTIFACT GENERATION COMPLETE.")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
