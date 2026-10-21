#!/usr/bin/env python3
"""
batch_compile_all_layers.py - Automated AOT Compiler for all Qwen2.5-Coder layers via EsAAC.
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path


def compile_layer(layer_idx: int, exported_dir: Path, compiled_dir: Path, ipc_dir: Path, force: bool = False):
    model_files_ipc = ipc_dir / "model_files"
    model_files_ipc.mkdir(parents=True, exist_ok=True)

    qkv_only = True
    if qkv_only:
        targets = [("qkv", f"qwen2_5_layer{layer_idx}_qkv")]
    else:
        targets = [
            ("qkv", f"qwen2_5_layer{layer_idx}_qkv"),
            ("attn_out", f"qwen2_5_layer{layer_idx}_attn_out"),
            ("mlp", f"qwen2_5_layer{layer_idx}_mlp"),
        ]

    for op_name, prefix in targets:
        model_name = f"{prefix}_npu_b1.model"
        dst_model = compiled_dir / model_name
        if not force and dst_model.exists() and dst_model.stat().st_size > 1000:
            print(f"[=] Layer {layer_idx} {op_name} already compiled: {model_name}")
            continue

        onnx_name = f"{prefix}.onnx"
        src_onnx = exported_dir / onnx_name
        if not src_onnx.exists():
            print(f"[!] Warning: {src_onnx} does not exist, skipping.")
            continue

        print(f"\n[*] Compiling Layer {layer_idx} {op_name} ({onnx_name}) [FP16 HALF]...")
        t0 = time.time()

        # Copy ONNX to shared IPC tmpfs
        ipc_onnx = ipc_dir / onnx_name
        shutil.copy2(src_onnx, ipc_onnx)

        # Invoke EsAAC in simulator container with --gen-fp16-model=yes
        cmd = [
            "ssh", "simulator",
            f"/home/eswin/EsAAC --input-model /ipc/{onnx_name} --output-dir /ipc/model_files --batchsize=1 --gen-fp16-model=yes"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            print(f"[!] Error compiling {onnx_name}:\n{res.stdout}")
            if ipc_onnx.exists():
                ipc_onnx.unlink()
            continue

        # Check for output model in ipc/model_files
        ipc_model = model_files_ipc / model_name
        if ipc_model.exists():
            shutil.copy2(ipc_model, dst_model)
            dt = time.time() - t0
            print(f"[+] Compiled {dst_model.name} ({dst_model.stat().st_size / (1024*1024):.2f} MB) in {dt:.2f}s")
            # Cleanup in container
            subprocess.run(["ssh", "simulator", f"rm -f /ipc/{onnx_name} /ipc/model_files/{prefix}*"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"[!] Compiled model not found at {ipc_model}")
            if ipc_onnx.exists():
                ipc_onnx.unlink()


def main():
    exported_dir = Path("/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b/exported")
    compiled_dir = Path("/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b/compiled_npu")
    ipc_dir = Path("/run/host-wayland/simulator-ipc")

    start_l = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_l = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    force = "--force" in sys.argv or True

    print(f"[*] Starting Batch AOT Compilation for Layers {start_l} to {end_l - 1} (FP16 HALF mode)...")
    for L in range(start_l, end_l):
        print(f"\n{'='*60}\n=== Processing Layer {L}/27 ===\n{'='*60}")
        compile_layer(L, exported_dir, compiled_dir, ipc_dir, force=force)

    print("\n[+] Batch AOT compilation complete!")


if __name__ == "__main__":
    main()
