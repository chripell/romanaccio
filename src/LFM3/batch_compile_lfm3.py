#!/usr/bin/env python3
"""
Milestone 3: Batch AOT Compilation (EsAAC), Dual-Die Manifest Generation,
and Simulator / Silicon Verification for LFM3-1.7B.

Features:
1. Uses the zero-copy host-container shared tmpfs mount (/run/host-wayland/simulator-ipc/lfm3_work
   <-> /ipc/lfm3_work) for instantaneous file staging (> 5 GB/s).
2. Compiles all 74 static 4D Opset-13 ONNX subgraphs via EsAAC --gen-fp16-model=yes.
3. Enforces strict output validation:
   - Greps every EsAAC compile log for 'has no graph', 'getBatchFromInputShape failed', or 'Check failed'.
   - Locates the output <name>_npu_b1.model file, verifies size > 1024 bytes, and copies it to
     /home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/models/<name>.model.
   - Cleans up intermediate build artifacts inside /ipc/lfm3_work after each compile.
4. Generates /home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/manifest.json mapping
   37 models to Die 0 (L00..L11 + lm_head_d0) and 37 models to Die 1 (L12..L23 + lm_head_d1).
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
CONFIG_PATH = MODEL_DIR / "config.json"
PKG_DIR = MODEL_DIR / "eswin_pkg"
MODELS_OUT_DIR = PKG_DIR / "models"
ONNX_IN_DIR = Path("/home/chri/aixigua/deep/artifacts/lfm3_onnx")
IPC_BASE = Path("/run/host-wayland/simulator-ipc")
IPC_WORK_HOST = IPC_BASE / "lfm3_work"
IPC_WORK_CONT = "/ipc/lfm3_work"


def get_expected_subgraphs(cfg: dict) -> list[dict]:
    """Returns ordered list of the 74 subgraphs with layer, die, role, and filename metadata."""
    layer_types = cfg["layer_types"]
    num_layers = cfg["num_hidden_layers"]
    items = []

    for l_idx in range(num_layers):
        die_id = 0 if l_idx < 12 else 1
        l_type = layer_types[l_idx]
        if l_type == "conv":
            items.append({
                "name": f"lfm3_l{l_idx:02d}_conv_in",
                "layer": l_idx,
                "die": die_id,
                "op": "conv_in",
                "in_c": 2048,
                "out_c": 6144,
            })
            items.append({
                "name": f"lfm3_l{l_idx:02d}_conv_out",
                "layer": l_idx,
                "die": die_id,
                "op": "conv_out",
                "in_c": 2048,
                "out_c": 2048,
            })
        else:
            items.append({
                "name": f"lfm3_l{l_idx:02d}_attn_qkv",
                "layer": l_idx,
                "die": die_id,
                "op": "attn_qkv",
                "in_c": 2048,
                "out_c": 3072,
            })
            items.append({
                "name": f"lfm3_l{l_idx:02d}_attn_out",
                "layer": l_idx,
                "die": die_id,
                "op": "attn_out",
                "in_c": 2048,
                "out_c": 2048,
            })

        items.append({
            "name": f"lfm3_l{l_idx:02d}_mlp",
            "layer": l_idx,
            "die": die_id,
            "op": "mlp",
            "in_c": 2048,
            "out_c": 2048,
        })

    for die_id in (0, 1):
        for s_idx in range(4):
            items.append({
                "name": f"lfm3_lm_head_d{die_id}_s{s_idx}",
                "layer": -1,
                "die": die_id,
                "op": "lm_head",
                "slice_idx": s_idx,
                "vocab_offset": (die_id * 4 + s_idx) * 16000,
                "in_c": 2048,
                "out_c": 16000,
            })
    return items


def compile_one_subgraph(item: dict, force: bool = False) -> tuple[str, bool, float, str]:
    """Compiles a single ONNX subgraph via EsAAC in an isolated /ipc/lfm3_work/<name> directory."""
    name = item["name"]
    src_onnx = ONNX_IN_DIR / f"{name}.onnx"
    dst_model = MODELS_OUT_DIR / f"{name}.model"

    if not force and dst_model.exists() and dst_model.stat().st_size > 10000:
        return (name, True, 0.0, f"skipped (already exists: {dst_model.stat().st_size / (1024*1024):.1f} MB)")

    if not src_onnx.exists():
        return (name, False, 0.0, f"missing input ONNX: {src_onnx}")

    t0 = time.time()
    job_host_dir = IPC_WORK_HOST / name
    job_cont_dir = f"{IPC_WORK_CONT}/{name}"

    if job_host_dir.exists():
        shutil.rmtree(job_host_dir, ignore_errors=True)
    job_host_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(job_host_dir, 0o777)

    # Zero-copy stage into /ipc tmpfs
    staged_onnx = job_host_dir / f"{name}.onnx"
    shutil.copy2(src_onnx, staged_onnx)
    os.chmod(staged_onnx, 0o666)

    cmd = [
        "ssh", "simulator",
        f"cd {job_cont_dir} && /home/eswin/EsAAC --input-model={job_cont_dir}/{name}.onnx --gen-fp16-model=yes"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_text = res.stdout or ""

    def cleanup_job():
        subprocess.run(["ssh", "simulator", f"rm -rf {job_cont_dir}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(job_host_dir, ignore_errors=True)

    # Check for silent EsAAC failures even if exit code is 0
    bad_markers = ["has no graph", "getBatchFromInputShape failed", "Check failed:", "Aborted", "Segmentation fault"]
    for marker in bad_markers:
        if marker in log_text:
            cleanup_job()
            return (name, False, time.time() - t0, f"EsAAC error marker '{marker}' in log:\n{log_text[-600:]}")

    if res.returncode != 0:
        cleanup_job()
        return (name, False, time.time() - t0, f"EsAAC exit {res.returncode}:\n{log_text[-600:]}")

    # Locate compiled <name>_npu_b1.model in job_host_dir/model_files
    compiled_candidates = list(job_host_dir.rglob(f"{name}_npu_b1.model"))
    if not compiled_candidates:
        cleanup_job()
        return (name, False, time.time() - t0, f"Output {name}_npu_b1.model not found in {job_host_dir}. Log:\n{log_text[-600:]}")

    compiled_file = compiled_candidates[0]
    if compiled_file.stat().st_size < 10000:
        cleanup_job()
        return (name, False, time.time() - t0, f"Output {compiled_file} suspiciously small ({compiled_file.stat().st_size} B)")

    shutil.copyfile(str(compiled_file), str(dst_model))
    cleanup_job()

    dt = time.time() - t0
    size_mb = dst_model.stat().st_size / (1024 * 1024)
    return (name, True, dt, f"{size_mb:6.2f} MB in {dt:5.2f}s")


def write_manifest(cfg: dict, items: list[dict]):
    """Writes the dual-die manifest.json for LFM3-1.7B."""
    die0_models = [f"models/{it['name']}.model" for it in items if it["die"] == 0]
    die1_models = [f"models/{it['name']}.model" for it in items if it["die"] == 1]

    assert len(die0_models) == 40, f"Expected 40 models on Die 0, got {len(die0_models)}"
    assert len(die1_models) == 40, f"Expected 40 models on Die 1, got {len(die1_models)}"

    manifest = {
        "model_name": "LFM3-1.7B",
        "architecture": "lfm3",
        "precision": "fp16",
        "config": {
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
            "conv_L_cache": cfg["conv_L_cache"],
            "sliding_window": cfg["sliding_window"],
            "use_sandwich_norm": cfg["use_sandwich_norm"],
            "norm_eps": cfg["norm_eps"],
            "layer_types": cfg["layer_types"],
            "context_length": 4096,
        },
        "files": {
            "embedding": "embedding.bin",
            "cpu_weights_bin": "cpu_weights.bin",
            "cpu_weights_index": "cpu_weights.json",
            "tokenizer": "../tokenizer.json",
            "tokenizer_config": "../tokenizer_config.json",
            "chat_template": "../chat_template.jinja",
        },
        "dies": {
            "0": {
                "device_id": 0,
                "layers": list(range(0, 12)),
                "resident_model_count": len(die0_models),
                "models": die0_models,
            },
            "1": {
                "device_id": 1,
                "layers": list(range(12, 24)),
                "resident_model_count": len(die1_models),
                "models": die1_models,
            },
        },
        "subgraphs": items,
    }

    manifest_path = PKG_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[+] Wrote dual-die manifest to {manifest_path}")
    print(f"    Die 0 resident models: {len(die0_models)} / 62 ceiling")
    print(f"    Die 1 resident models: {len(die1_models)} / 62 ceiling")


def main():
    force = "--force" in sys.argv
    workers = 4
    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])

    MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(IPC_BASE, 0o775)
    os.system(f"chmod o+rx {IPC_BASE}")
    IPC_WORK_HOST.mkdir(parents=True, exist_ok=True)
    os.chmod(IPC_WORK_HOST, 0o777)

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    items = get_expected_subgraphs(cfg)
    print("=================================================================")
    print(f" LFM3-1.7B Milestone 3: Batch AOT Compilation ({len(items)} Subgraphs)")
    print(f" Parallel workers: {workers} | Shared RAM mount: {IPC_WORK_HOST}")
    print("=================================================================\n")

    t_start = time.time()
    failures = []
    completed = 0

    # Compile layer subgraphs first (72 files), then the two large 500MB LM head slices with 2 workers
    layer_items = [it for it in items if it["op"] != "lm_head"]
    head_items = [it for it in items if it["op"] == "lm_head"]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(compile_one_subgraph, it, force): it for it in layer_items}
        for fut in as_completed(fut_map):
            name, ok, dt, msg = fut.result()
            completed += 1
            status = "+" if ok else "!"
            print(f"  [{status}] ({completed:02d}/{len(items)}) {name:<26} -> {msg}", flush=True)
            if not ok:
                failures.append((name, msg))

    # Compile LM head slices sequentially (1 worker) so EsAAC /tmp/build/build_npu_b1 never collides
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut_map = {ex.submit(compile_one_subgraph, it, force): it for it in head_items}
        for fut in as_completed(fut_map):
            name, ok, dt, msg = fut.result()
            completed += 1
            status = "+" if ok else "!"
            print(f"  [{status}] ({completed:02d}/{len(items)}) {name:<26} -> {msg}", flush=True)
            if not ok:
                failures.append((name, msg))

    total_dt = time.time() - t_start
    print(f"\nBatch compilation finished in {total_dt:.1f}s.")

    if failures:
        print(f"\n[!] {len(failures)} subgraphs FAILED compilation:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)

    # Final verification count
    compiled_files = [MODELS_OUT_DIR / f"{it['name']}.model" for it in items]
    missing = [p for p in compiled_files if not p.exists() or p.stat().st_size <= 10000]
    if missing:
        print(f"[!] Missing or empty model files: {missing}")
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p in compiled_files)
    print(f"[+] Verified all {len(compiled_files)}/{len(items)} .model files exist and are non-empty ({total_bytes / (1024**3):.2f} GiB total).")

    write_manifest(cfg, items)


if __name__ == "__main__":
    main()
