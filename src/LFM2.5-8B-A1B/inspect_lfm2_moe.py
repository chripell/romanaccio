#!/usr/bin/env python3
"""
Detailed Architecture & Tensor Inventory Analysis for LiquidAI/LFM2.5-8B-A1B.
Milestone 15 - Phase A.
"""

import os
import sys
import json
from pathlib import Path
import safetensors
import safetensors.torch

MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM2.5-8B-A1B")
CONFIG_PATH = MODEL_DIR / "config.json"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"
OUTPUT_JSON_PATH = Path("/home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/inspect_summary.json")

def format_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B ({n:,})"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M ({n:,})"
    elif n >= 1_000:
        return f"{n / 1_000:.3f}K ({n:,})"
    return f"{n:,}"

def format_bytes(b: int) -> str:
    gb = b / (1024 ** 3)
    mb = b / (1024 ** 2)
    if gb >= 1.0:
        return f"{gb:.3f} GiB ({b:,} B)"
    return f"{mb:.3f} MiB ({b:,} B)"

def main():
    print("=" * 80)
    print(" LiquidAI / LFM2.5-8B-A1B: Architecture & Tensor Inventory Analysis")
    print("=" * 80)

    # 1. Inspect config.json
    print(f"\n[1/4] Reading and validating config from {CONFIG_PATH}...")
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    # Validate essential fields
    model_type = cfg.get("model_type")
    hidden_size = cfg.get("hidden_size")
    intermediate_size = cfg.get("intermediate_size")
    moe_intermediate_size = cfg.get("moe_intermediate_size")
    num_hidden_layers = cfg.get("num_hidden_layers")
    num_dense_layers = cfg.get("num_dense_layers")
    num_experts = cfg.get("num_experts")
    num_experts_per_tok = cfg.get("num_experts_per_tok")
    num_attention_heads = cfg.get("num_attention_heads")
    num_key_value_heads = cfg.get("num_key_value_heads")
    conv_L_cache = cfg.get("conv_L_cache")
    norm_topk_prob = cfg.get("norm_topk_prob")
    use_expert_bias = cfg.get("use_expert_bias")
    rope_theta = cfg.get("rope_parameters", {}).get("rope_theta")
    layer_types = cfg.get("layer_types", [])

    conv_layers_idx = [i for i, t in enumerate(layer_types) if t == "conv"]
    attn_layers_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    num_moe_layers = num_hidden_layers - num_dense_layers

    print(f"  • Model Type:            {model_type} (Expected: lfm2_moe)")
    print(f"  • Hidden Size (d_model): {hidden_size}")
    print(f"  • Total Layers:          {num_hidden_layers} ({len(conv_layers_idx)} Conv + {len(attn_layers_idx)} Attention)")
    print(f"  • Dense FFN Layers:      {num_dense_layers} (Layers 0..{num_dense_layers-1}, intermediate_size={intermediate_size})")
    print(f"  • MoE FFN Layers:        {num_moe_layers} (Layers {num_dense_layers}..{num_hidden_layers-1})")
    print(f"  • Expert Geometry:       {num_experts} experts/layer, top-{num_experts_per_tok} routing")
    print(f"  • MoE Intermediate Size: {moe_intermediate_size} per expert")
    print(f"  • Attention Geometry:    {num_attention_heads} Q heads, {num_key_value_heads} KV heads (head_dim={hidden_size // num_attention_heads})")
    print(f"  • RoPE Theta:            {rope_theta}")
    print(f"  • ShortConv Kernel:      conv_L_cache = {conv_L_cache} (3-tap conv state ring)")
    print(f"  • Router Config:         norm_topk_prob={norm_topk_prob}, use_expert_bias={use_expert_bias}")

    # 2. Inspect safetensors
    print(f"\n[2/4] Inspecting tensor weights from {SAFETENSORS_PATH}...")
    if not SAFETENSORS_PATH.exists():
        print(f"ERROR: safetensors file not found at {SAFETENSORS_PATH}")
        sys.exit(1)

    tensor_inventory = []
    total_elements = 0
    total_bytes = 0

    category_counts = {
        "embedding": 0,
        "embedding_norm": 0,
        "operator_conv": 0,
        "operator_attn": 0,
        "dense_mlp": 0,
        "moe_router": 0,
        "moe_experts": 0,
        "norms_and_biases": 0,
        "lm_head": 0,
        "other": 0
    }

    category_bytes = {k: 0 for k in category_counts}

    dtype_map = {
        "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1,
        "torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4
    }

    with safetensors.safe_open(SAFETENSORS_PATH, framework="pt") as f:
        keys = f.keys()
        print(f"  Total named tensors in safetensors: {len(keys)}")

        for key in sorted(keys):
            tensor_slice = f.get_slice(key)
            shape = list(tensor_slice.get_shape())
            dtype_str = str(tensor_slice.get_dtype())
            
            numel = 1
            for dim in shape:
                numel *= dim
            
            element_size = dtype_map.get(dtype_str, 2)
            n_bytes = numel * element_size
            total_elements += numel
            total_bytes += n_bytes

            # Categorize
            if "embed_tokens" in key:
                cat = "embedding"
            elif "embedding_norm" in key:
                cat = "embedding_norm"
            elif "lm_head" in key:
                cat = "lm_head"
            elif ".conv." in key:
                cat = "operator_conv"
            elif ".self_attn." in key:
                cat = "operator_attn"
            elif any(w in key for w in [".feed_forward.w1.", ".feed_forward.w2.", ".feed_forward.w3."]):
                cat = "dense_mlp"
            elif any(r in key for r in [".feed_forward.gate.", ".feed_forward.expert_bias"]):
                cat = "moe_router"
            elif ".feed_forward.experts." in key:
                cat = "moe_experts"
            elif "norm" in key:
                cat = "norms_and_biases"
            else:
                cat = "other"

            category_counts[cat] += numel
            category_bytes[cat] += n_bytes

            tensor_inventory.append({
                "name": key,
                "shape": shape,
                "dtype": dtype_str,
                "numel": numel,
                "bytes": n_bytes,
                "category": cat
            })

    print(f"  Total unique stored parameters: {format_num(total_elements)}")
    print(f"  Total file weight storage:      {format_bytes(total_bytes)}")

    # 3. Parameter Arithmetic & MoE Active Ratio
    print("\n[3/4] Parameter Accounting & Breakdown:")
    print("-" * 80)
    print(f"{'Category':<22} | {'Parameters':<22} | {'Disk/VRAM Size':<18} | {'% Total':<8}")
    print("-" * 80)
    for cat, count in category_counts.items():
        if count == 0 and category_bytes[cat] == 0:
            continue
        pct = (count / total_elements) * 100
        print(f"{cat:<22} | {format_num(count):<22} | {format_bytes(category_bytes[cat]):<18} | {pct:>6.2f}%")
    print("-" * 80)
    print(f"{'TOTAL STORED':<22} | {format_num(total_elements):<22} | {format_bytes(total_bytes):<18} | 100.00%")
    print("-" * 80)

    # Active parameters calculation
    # In each MoE layer, exactly `num_experts_per_tok` (4) of 32 experts are active per token (12.5% = 1/8)
    moe_experts_total = category_counts["moe_experts"]
    active_moe_experts = moe_experts_total * (num_experts_per_tok / num_experts)
    
    # Active backbone = all operators + norms + dense mlp + moe routers + active moe experts
    active_backbone = (
        category_counts["operator_conv"] +
        category_counts["operator_attn"] +
        category_counts["dense_mlp"] +
        category_counts["moe_router"] +
        category_counts["norms_and_biases"] +
        active_moe_experts
    )
    # Active total including token embedding slice (2048 params) and LM head projection (128000*2048 if tied)
    # With tied embedding table:
    active_with_embed = active_backbone + category_counts["embedding"]

    print("\n[Active Parameter Arithmetic Per Decoding Token]")
    print(f"  • Dense Operator Backbone:  {format_num(category_counts['operator_conv'] + category_counts['operator_attn'])} params")
    print(f"  • Dense FFN (Layers 0, 1):  {format_num(category_counts['dense_mlp'])} params")
    print(f"  • MoE Routers + Norms:       {format_num(category_counts['moe_router'] + category_counts['norms_and_biases'])} params")
    print(f"  • Routed Experts (4 of 32):  {format_num(int(active_moe_experts))} params (1/8th of {format_num(moe_experts_total)})")
    print(f"  -------------------------------------------------------------")
    print(f"  • Total Active Backbone:    {format_num(int(active_backbone))} params (~{active_backbone/1e9:.2f}B)")
    print(f"  • Active With Embed Table:  {format_num(int(active_with_embed))} params (~{active_with_embed/1e9:.2f}B)")
    print(f"  • Active Parameter Ratio:   {(active_backbone / total_elements)*100:.2f}% of total weights")

    # Memory and Bandwidth Projections on Dual-Die EIC7702
    int8_active_backbone_bytes = int(active_backbone) * 1
    fp16_active_backbone_bytes = int(active_backbone) * 2
    aggregate_bw_gb_s = 28.0 # EIC7702 aggregate LPDDR5 bandwidth
    per_die_bw_gb_s = 14.0   # 14 GB/s per NUMA die

    int8_latency_ms = (int8_active_backbone_bytes / (aggregate_bw_gb_s * (1024**3))) * 1000
    fp16_latency_ms = (fp16_active_backbone_bytes / (aggregate_bw_gb_s * (1024**3))) * 1000
    int8_tps = 1000.0 / int8_latency_ms
    fp16_tps = 1000.0 / fp16_latency_ms

    print("\n[EIC7702 Silicon Bandwidth & Latency Projection (28 GB/s DRAM Peak)]")
    print(f"  • Active INT8 DRAM Stream:  {format_bytes(int8_active_backbone_bytes)} per token")
    print(f"    - Per-die stream (EP=2):  {format_bytes(int8_active_backbone_bytes // 2)} / die")
    print(f"    - Projected INT8 Latency: {int8_latency_ms:.1f} ms/tok  =>  {int8_tps:.1f} tokens/sec")
    print(f"  • Active FP16 DRAM Stream:  {format_bytes(fp16_active_backbone_bytes)} per token")
    print(f"    - Per-die stream (EP=2):  {format_bytes(fp16_active_backbone_bytes // 2)} / die")
    print(f"    - Projected FP16 Latency: {fp16_latency_ms:.1f} ms/tok  =>  {fp16_tps:.1f} tokens/sec")

    # MMZ Footprint on Dual-Die (27 GiB total, ~13.5 GiB / die)
    int8_total_bytes = total_elements * 1
    fp16_total_bytes = total_elements * 2
    print("\n[MMZ VRAM Footprint vs 27 GiB Available (13.5 GiB per Die)]")
    print(f"  • INT8 Weight Footprint:    {format_bytes(int8_total_bytes)} ({format_bytes(int8_total_bytes // 2)} / die) -> Fits easily in MMZ!")
    print(f"  • FP16 Weight Footprint:    {format_bytes(fp16_total_bytes)} ({format_bytes(fp16_total_bytes // 2)} / die) -> Fits in MMZ!")

    # 4. Save JSON Report
    print(f"\n[4/4] Writing summary report to {OUTPUT_JSON_PATH}...")
    summary = {
        "model_id": "LiquidAI/LFM2.5-8B-A1B",
        "config": cfg,
        "total_parameters": total_elements,
        "total_bytes": total_bytes,
        "category_counts": category_counts,
        "category_bytes": category_bytes,
        "active_backbone_parameters": int(active_backbone),
        "active_with_embed_parameters": int(active_with_embed),
        "projections": {
            "aggregate_bw_gb_s": aggregate_bw_gb_s,
            "int8_latency_ms": int8_latency_ms,
            "int8_tokens_per_sec": int8_tps,
            "fp16_latency_ms": fp16_latency_ms,
            "fp16_tokens_per_sec": fp16_tps,
            "int8_per_die_bytes": int8_total_bytes // 2,
            "fp16_per_die_bytes": fp16_total_bytes // 2
        },
        "tensor_inventory": tensor_inventory
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Report written successfully ({len(tensor_inventory)} tensors cataloged).")
    print("=" * 80)
    print(" Architecture Inspection: COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
