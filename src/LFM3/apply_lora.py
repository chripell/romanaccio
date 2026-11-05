#!/usr/bin/env python3
"""
PEFT LoRA Weight Fuser for LFM3-1.7B.

Applies low-rank adaptation weights:
  W_adapted = W_base + (alpha / r) * (B @ A)
directly into base model weights before quantization/packaging.

Zero runtime overhead, zero .model changes required.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def fuse_lora(base_dir: str, lora_dir: str, out_dir: str):
    base_dir = Path(base_dir)
    lora_dir = Path(lora_dir)
    out_dir = Path(out_dir)

    base_safetensors = base_dir / "model.safetensors"
    lora_safetensors = lora_dir / "adapter_model.safetensors"
    lora_config_file = lora_dir / "adapter_config.json"

    if not base_safetensors.exists():
        raise FileNotFoundError(f"Base model not found at {base_safetensors}")
    if not lora_safetensors.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {lora_safetensors}")

    with open(lora_config_file) as f:
        lora_cfg = json.load(f)

    r = lora_cfg.get("r", 3)
    lora_alpha = lora_cfg.get("lora_alpha", 3)
    scaling = float(lora_alpha) / float(r)
    target_modules = lora_cfg.get("target_modules", ["out_proj", "w2", "o_proj"])

    print(f"[+] Loaded LoRA Config: rank={r}, alpha={lora_alpha}, scaling={scaling:.4f}")
    print(f"[+] Target modules: {target_modules}")

    # Load LoRA tensors
    lora_tensors = {}
    with safe_open(lora_safetensors, framework="pt", device="cpu") as f:
        for k in f.keys():
            lora_tensors[k] = f.get_tensor(k)
    print(f"[+] Loaded {len(lora_tensors)} LoRA tensors.")

    # Read and fuse into base weights
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tensors = {}
    fused_count = 0

    with safe_open(base_safetensors, framework="pt", device="cpu") as f:
        base_keys = list(f.keys())
        print(f"[+] Inspecting {len(base_keys)} base model tensors...")

        for k in base_keys:
            w_base = f.get_tensor(k)
            orig_dtype = w_base.dtype

            # Check if this tensor matches a LoRA target
            # LoRA keys: base_model.model.model.layers.{L}.{module}.lora_{A,B}.weight
            matched = False
            for mod in target_modules:
                if f".{mod}.weight" in k:
                    # Construct expected LoRA keys
                    # e.g. k = 'model.layers.0.conv.out_proj.weight'
                    # prefix = 'model.layers.0.conv.out_proj'
                    prefix = k[:-len(".weight")]
                    lora_a_key = f"base_model.model.{prefix}.lora_A.weight"
                    lora_b_key = f"base_model.model.{prefix}.lora_B.weight"

                    if lora_a_key in lora_tensors and lora_b_key in lora_tensors:
                        lora_A = lora_tensors[lora_a_key].to(torch.float32)
                        lora_B = lora_tensors[lora_b_key].to(torch.float32)

                        delta_w = (lora_B @ lora_A) * scaling
                        if delta_w.shape != w_base.shape:
                            raise ValueError(f"Shape mismatch on {k}: base={w_base.shape}, delta={delta_w.shape}")

                        w_fused = (w_base.to(torch.float32) + delta_w).to(orig_dtype)
                        merged_tensors[k] = w_fused
                        fused_count += 1
                        matched = True
                        break

            if not matched:
                merged_tensors[k] = w_base

    print(f"[+] Successfully fused LoRA into {fused_count} target matrices!")
    out_safetensors = out_dir / "model.safetensors"
    print(f"[+] Writing fused weights to {out_safetensors}...")
    save_file(merged_tensors, str(out_safetensors))
    print(f"[+] Saved {out_safetensors} ({out_safetensors.stat().st_size / (1024*1024):.2f} MB)")

    # Copy metadata files (config.json, tokenizer*, etc.)
    for meta_file in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        src_path = base_dir / meta_file
        dst_path = out_dir / meta_file
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"[+] Copied {meta_file} to {out_dir}")

    print(f"[+] LoRA Fusion Complete! Fused model ready in: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fuse PEFT LoRA into LFM3 base model")
    parser.add_argument("--base-dir", type=str, default="/home/chri/aixigua/deep/models/LFM3-1.7B",
                        help="Base model directory containing model.safetensors")
    parser.add_argument("--lora-dir", type=str, default="/home/chri/aixigua/deep/models/LFM3-1.7B-Heretic-LoRA",
                        help="LoRA directory containing adapter_model.safetensors and adapter_config.json")
    parser.add_argument("--out-dir", type=str, default="/home/chri/aixigua/deep/models/LFM3-1.7B-Heretic",
                        help="Output directory for merged model")
    args = parser.parse_args()

    fuse_lora(args.base_dir, args.lora_dir, args.out_dir)

if __name__ == "__main__":
    main()
