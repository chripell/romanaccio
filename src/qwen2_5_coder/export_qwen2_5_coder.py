#!/usr/bin/env python3
"""
export_qwen2_5_coder.py - Model extractor and ONNX sub-graph exporter for Qwen2.5-Coder-1.5B.

Exports:
1. embedding.bin: Raw FP16 embedding table (151936 x 1536) for direct CPU mmap.
2. qwen2_5_layer0_qkv.onnx: Fused QKV 1x1 Conv projection (1536 in -> 2048 out) [Q:1536, K:256, V:256].
3. qwen2_5_layer0_attn_out.onnx: Attention output 1x1 Conv projection (1536 in -> 1536 out).
4. qwen2_5_layer0_mlp.onnx: Full SwiGLU MLP Block (1536 in -> 8960 hidden -> 1536 out).
5. qwen2_5_lm_head_d0.onnx: Die 0 LM Head 1x1 Conv projection (1536 in -> 76032 out).
6. qwen2_5_lm_head_d1.onnx: Die 1 LM Head 1x1 Conv projection (1536 in -> 76032 out).

All ONNX models are exported with opset_version=13 and dynamo=False for 100% compatibility
with the ESWIN EsAAC compiler and NPU SPRAM architecture.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class Qwen25QKVBlock(nn.Module):
    """Fused QKV 1x1 Pointwise Convolution (1536 in -> 2048 out) with Q, K, V bias."""
    def __init__(self, q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor,
                 q_b: torch.Tensor, k_b: torch.Tensor, v_b: torch.Tensor):
        super().__init__()
        # Concatenate weights along output channels: Q (1536), K (256), V (256) -> 2048
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        self.weight = nn.Parameter(fused_w.view(fused_w.shape[0], fused_w.shape[1], 1, 1))
        # Concatenate biases along output channels: Q (1536), K (256), V (256) -> 2048
        fused_b = torch.cat([q_b, k_b, v_b], dim=0)
        self.bias = nn.Parameter(fused_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias)


class Qwen25AttnOutBlock(nn.Module):
    """Attention Output 1x1 Pointwise Convolution (1536 in -> 1536 out)."""
    def __init__(self, o_w: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(o_w.view(o_w.shape[0], o_w.shape[1], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight)


class Qwen25MLPBlock(nn.Module):
    """Full SwiGLU MLP Block (1536 in -> 8960 hidden -> 1536 out)."""
    def __init__(self, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor):
        super().__init__()
        self.w_gate = nn.Parameter(gate_w.view(gate_w.shape[0], gate_w.shape[1], 1, 1))
        self.w_up = nn.Parameter(up_w.view(up_w.shape[0], up_w.shape[1], 1, 1))
        self.w_down = nn.Parameter(down_w.view(down_w.shape[0], down_w.shape[1], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.conv2d(x, self.w_gate)
        up = F.conv2d(x, self.w_up)
        act = F.silu(gate) * up
        return F.conv2d(act, self.w_down)


class DualDieLMHead(nn.Module):
    """LM Head slice 1x1 Conv for 76032 vocabulary tokens."""
    def __init__(self, weight_slice: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight_slice.view(weight_slice.shape[0], weight_slice.shape[1], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight)


def export_layer(f, output_path: Path, layer_idx: int, dummy_x: torch.Tensor, force: bool = False):
    """Exports QKV, Attn Out, and SwiGLU MLP subgraphs for a single layer."""
    # 1. QKV (with Q, K, V bias)
    qkv_onnx = output_path / f"qwen2_5_layer{layer_idx}_qkv.onnx"
    if force or not qkv_onnx.exists():
        print(f"[*] Exporting Layer {layer_idx} QKV with bias (1536 -> 2048)...")
        q_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight").to(torch.float32)
        k_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight").to(torch.float32)
        v_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight").to(torch.float32)
        q_b = f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.bias").to(torch.float32)
        k_b = f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.bias").to(torch.float32)
        v_b = f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.bias").to(torch.float32)
        qkv_block = Qwen25QKVBlock(q_w, k_w, v_w, q_b, k_b, v_b)
        qkv_block.eval()
        torch.onnx.export(
            qkv_block, dummy_x, str(qkv_onnx),
            input_names=["input"], output_names=["output"],
            opset_version=13, dynamo=False, do_constant_folding=True
        )
        print(f"[+] Exported {qkv_onnx} ({qkv_onnx.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[=] Layer {layer_idx} QKV already exported.")

    # 2. Attn Out
    attn_out_onnx = output_path / f"qwen2_5_layer{layer_idx}_attn_out.onnx"
    if not attn_out_onnx.exists():
        print(f"[*] Exporting Layer {layer_idx} Attn Out (1536 -> 1536)...")
        o_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight").to(torch.float32)
        attn_out_block = Qwen25AttnOutBlock(o_w)
        attn_out_block.eval()
        torch.onnx.export(
            attn_out_block, dummy_x, str(attn_out_onnx),
            input_names=["input"], output_names=["output"],
            opset_version=13, dynamo=False, do_constant_folding=True
        )
        print(f"[+] Exported {attn_out_onnx} ({attn_out_onnx.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[=] Layer {layer_idx} Attn Out already exported.")

    # 3. SwiGLU MLP
    mlp_onnx = output_path / f"qwen2_5_layer{layer_idx}_mlp.onnx"
    if not mlp_onnx.exists():
        print(f"[*] Exporting Layer {layer_idx} SwiGLU MLP (1536 -> 8960 -> 1536)...")
        gate_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight").to(torch.float32)
        up_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight").to(torch.float32)
        down_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight").to(torch.float32)
        mlp_block = Qwen25MLPBlock(gate_w, up_w, down_w)
        mlp_block.eval()
        torch.onnx.export(
            mlp_block, dummy_x, str(mlp_onnx),
            input_names=["input"], output_names=["output"],
            opset_version=13, dynamo=False, do_constant_folding=True
        )
        print(f"[+] Exported {mlp_onnx} ({mlp_onnx.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[=] Layer {layer_idx} SwiGLU MLP already exported.")


def export_qwen2_5_coder(model_dir: str, out_dir: str, start_layer: int = 0, num_layers: int = 1, force: bool = False):
    model_path = Path(model_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safetensors_file = model_path / "model.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

    print(f"[*] Opening {safetensors_file}...")
    with safe_open(str(safetensors_file), framework="pt") as f:
        # 1. Export embedding.bin
        embed_w = f.get_tensor("model.embed_tokens.weight").to(torch.float16)
        vocab_size, hidden_dim = embed_w.shape
        print(f"[*] Embedding table shape: ({vocab_size}, {hidden_dim}) [FP16]")

        emb_file = output_path / "embedding.bin"
        if not emb_file.exists():
            embed_np = embed_w.numpy()
            embed_np.tofile(str(emb_file))
            print(f"[+] Saved {emb_file} ({emb_file.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"[=] {emb_file} already exists, skipping.")

        dummy_x = torch.zeros(1, hidden_dim, 1, 1, dtype=torch.float32)

        # 2. Export requested layers
        for l_idx in range(start_layer, start_layer + num_layers):
            export_layer(f, output_path, l_idx, dummy_x, force=force)

        # 3. Export Dual-Die LM Head ONNX Models (if not already exported)
        half_padded = 76032
        lm_d0_onnx = output_path / "qwen2_5_lm_head_d0.onnx"
        lm_d1_onnx = output_path / "qwen2_5_lm_head_d1.onnx"

        if not lm_d0_onnx.exists():
            print(f"\n[*] Exporting Dual-Die LM Head Die 0 ({half_padded} tokens)...")
            w_d0_raw = embed_w[:75968, :].to(torch.float32)
            w_d0 = torch.zeros(half_padded, hidden_dim, dtype=torch.float32)
            w_d0[:75968, :] = w_d0_raw

            lm_d0 = DualDieLMHead(w_d0)
            lm_d0.eval()
            torch.onnx.export(
                lm_d0, dummy_x, str(lm_d0_onnx),
                input_names=["input"], output_names=["output"],
                opset_version=13, dynamo=False, do_constant_folding=True
            )
            print(f"[+] Exported {lm_d0_onnx} ({lm_d0_onnx.stat().st_size / (1024*1024):.2f} MB)")

        if not lm_d1_onnx.exists():
            print(f"\n[*] Exporting Dual-Die LM Head Die 1 ({half_padded} tokens)...")
            w_d1_raw = embed_w[75968:, :].to(torch.float32)
            w_d1 = torch.zeros(half_padded, hidden_dim, dtype=torch.float32)
            w_d1[:len(w_d1_raw), :] = w_d1_raw

            lm_d1 = DualDieLMHead(w_d1)
            lm_d1.eval()
            torch.onnx.export(
                lm_d1, dummy_x, str(lm_d1_onnx),
                input_names=["input"], output_names=["output"],
                opset_version=13, dynamo=False, do_constant_folding=True
            )
            print(f"[+] Exported {lm_d1_onnx} ({lm_d1_onnx.stat().st_size / (1024*1024):.2f} MB)")

    print("\n[+] All requested Qwen2.5-Coder artifacts exported successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Qwen2.5-Coder subgraphs to ONNX")
    parser.add_argument("--model-dir", default="/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b")
    parser.add_argument("--out-dir", default="/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b/exported")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force re-exporting existing layers")
    args = parser.parse_args()

    n_layers = 28 if args.all_layers else args.num_layers
    export_qwen2_5_coder(args.model_dir, args.out_dir, start_layer=args.start_layer, num_layers=n_layers, force=args.force)
