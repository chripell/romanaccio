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
    """Fused QKV 1x1 Pointwise Convolution (1536 in -> 2048 out)."""
    def __init__(self, q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor):
        super().__init__()
        # Concatenate along output channels: Q (1536), K (256), V (256) -> 2048
        fused = torch.cat([q_w, k_w, v_w], dim=0)
        self.weight = nn.Parameter(fused.view(fused.shape[0], fused.shape[1], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight)


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


def export_qwen2_5_coder(model_dir: str, out_dir: str, layer_idx: int = 0):
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

        # 2. Export Layer QKV Proj
        print(f"\n[*] Exporting Layer {layer_idx} Fused QKV Projection (1536 -> 2048)...")
        q_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight").to(torch.float32)
        k_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight").to(torch.float32)
        v_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight").to(torch.float32)

        qkv_block = Qwen25QKVBlock(q_w, k_w, v_w)
        qkv_block.eval()

        qkv_onnx = output_path / f"qwen2_5_layer{layer_idx}_qkv.onnx"
        torch.onnx.export(
            qkv_block,
            dummy_x,
            str(qkv_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[+] Exported {qkv_onnx} ({qkv_onnx.stat().st_size / (1024*1024):.2f} MB)")

        # 3. Export Layer Attn Out Proj
        print(f"\n[*] Exporting Layer {layer_idx} Attention Output Projection (1536 -> 1536)...")
        o_w = f.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight").to(torch.float32)

        attn_out_block = Qwen25AttnOutBlock(o_w)
        attn_out_block.eval()

        attn_out_onnx = output_path / f"qwen2_5_layer{layer_idx}_attn_out.onnx"
        torch.onnx.export(
            attn_out_block,
            dummy_x,
            str(attn_out_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[+] Exported {attn_out_onnx} ({attn_out_onnx.stat().st_size / (1024*1024):.2f} MB)")

        # 4. Export Layer SwiGLU MLP Block
        print(f"\n[*] Exporting Layer {layer_idx} SwiGLU MLP Block (1536 -> 8960 -> 1536)...")
        gate_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight").to(torch.float32)
        up_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight").to(torch.float32)
        down_w = f.get_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight").to(torch.float32)

        mlp_block = Qwen25MLPBlock(gate_w, up_w, down_w)
        mlp_block.eval()

        mlp_onnx = output_path / f"qwen2_5_layer{layer_idx}_mlp.onnx"
        torch.onnx.export(
            mlp_block,
            dummy_x,
            str(mlp_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[+] Exported {mlp_onnx} ({mlp_onnx.stat().st_size / (1024*1024):.2f} MB)")

        # 5. Export Dual-Die LM Head ONNX Models (76032 vocab slice each)
        half_padded = 76032
        print(f"\n[*] Exporting Dual-Die LM Head ONNX models (each slice = {half_padded} vocab tokens)...")

        # Die 0: 0..75967 (pad remainder with 0s to 76032)
        w_d0_raw = embed_w[:75968, :].to(torch.float32)
        w_d0 = torch.zeros(half_padded, hidden_dim, dtype=torch.float32)
        w_d0[:75968, :] = w_d0_raw

        lm_d0 = DualDieLMHead(w_d0)
        lm_d0.eval()

        lm_d0_onnx = output_path / "qwen2_5_lm_head_d0.onnx"
        torch.onnx.export(
            lm_d0,
            dummy_x,
            str(lm_d0_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[+] Exported {lm_d0_onnx} ({lm_d0_onnx.stat().st_size / (1024*1024):.2f} MB)")

        # Die 1: 75968..151935 (pad remainder with 0s to 76032)
        w_d1_raw = embed_w[75968:, :].to(torch.float32)
        w_d1 = torch.zeros(half_padded, hidden_dim, dtype=torch.float32)
        w_d1[:len(w_d1_raw), :] = w_d1_raw

        lm_d1 = DualDieLMHead(w_d1)
        lm_d1.eval()

        lm_d1_onnx = output_path / "qwen2_5_lm_head_d1.onnx"
        torch.onnx.export(
            lm_d1,
            dummy_x,
            str(lm_d1_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[+] Exported {lm_d1_onnx} ({lm_d1_onnx.stat().st_size / (1024*1024):.2f} MB)")

    print("\n[+] All requested Qwen2.5-Coder artifacts exported successfully!")


if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b"
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/chri/aixigua/deep/models/qwen2_5_coder_1.5b/exported"
    l_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    export_qwen2_5_coder(src_dir, dst_dir, layer_idx=l_idx)
