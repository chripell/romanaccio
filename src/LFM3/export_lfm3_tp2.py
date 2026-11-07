#!/usr/bin/env python3
"""
Milestone 7: Export Half-Width TP=2 ONNX Archetypes for LFM3-1.7B (`T=1` & `T=64`).

Exports 4D static NCHW `[1, C, 1, W]` opset-13 ONNX subgraphs for `W in {1, 64}`:
  - `mlp_tp2_t{1,64}.onnx`      (`2048 -> 3840 -> 2048` SwiGLU MLP per die; 15 w1/w3 slices + 16 w2 slices)
  - `conv_in_tp2_t{1,64}.onnx`  (`2048 -> 3072` per die)
  - `attn_qkv_tp2_t{1,64}.onnx` (`2048 -> 1536` per die)
  - `proj2048_tp2_t{1,64}.onnx` (`2048 -> 1024` per die)

Compile with EsAAC inside the `simulator` container using `shared_*_table.json` from
`calibrate_lfm3_int8.py`, then package with `build_lfm3_tp2_pkg.py`.
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper
from safetensors import safe_open
import torch


def make_conv1x1_onnx(in_c: int, out_c: int, w_np: np.ndarray, seq_len: int,
                      in_name: str = "x", out_name: str = "y", conv_name: str = "proj") -> onnx.ModelProto:
    w_4d = w_np.astype(np.float32).reshape(out_c, in_c, 1, 1)
    inp = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, in_c, 1, seq_len])
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, out_c, 1, seq_len])
    w_init = helper.make_tensor(f"{conv_name}.weight", TensorProto.FLOAT, [out_c, in_c, 1, 1], w_4d.tobytes(), raw=True)
    node = helper.make_node("Conv", [in_name, f"{conv_name}.weight"], [out_name], name=conv_name, kernel_shape=[1, 1])
    graph = helper.make_graph([node], f"{conv_name}_t{seq_len}", [inp], [out], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def make_mlp_tp2_onnx(w1_np: np.ndarray, w3_np: np.ndarray, w2_np: np.ndarray, seq_len: int) -> onnx.ModelProto:
    """
    Half-width TP=2 SwiGLU MLP (`2048 -> 3840 -> 2048` per die):
      - `w1`, `w3`: `[3840, 2048]` split into 15 slices of 256 channels + `Concat_0`, `Concat_1`
      - `Sigmoid_0` + `Mul_0` + `Mul_1`
      - `w2`: `[2048, 3840]` split into 16 slices of 64 output channels + `Concat_2`
    Node names match `shared_mlp_table.json` so EsAAC consumes the calibrated INT8/INT16 table directly.
    """
    in_c, inter_c, out_c = 2048, 3840, 2048
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_c, 1, seq_len])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, out_c, 1, seq_len])

    nodes = []
    inits = []

    w1_outs = []
    for i in range(15):
        w_sl = w1_np[i * 256:(i + 1) * 256, :].astype(np.float32).reshape(256, in_c, 1, 1)
        w_name = f"w1_s{i}.weight"
        out_n = f"w1_s{i}_out"
        inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, [256, in_c, 1, 1], w_sl.tobytes(), raw=True))
        nodes.append(helper.make_node("Conv", ["x", w_name], [out_n], name=f"Conv_{i}", kernel_shape=[1, 1]))
        w1_outs.append(out_n)

    nodes.append(helper.make_node("Concat", w1_outs, ["w1_cat"], name="Concat_0", axis=1))
    nodes.append(helper.make_node("Sigmoid", ["w1_cat"], ["w1_sig"], name="Sigmoid_0"))
    nodes.append(helper.make_node("Mul", ["w1_cat", "w1_sig"], ["silu_out"], name="Mul_0"))

    w3_outs = []
    for i in range(15):
        w_sl = w3_np[i * 256:(i + 1) * 256, :].astype(np.float32).reshape(256, in_c, 1, 1)
        w_name = f"w3_s{i}.weight"
        out_n = f"w3_s{i}_out"
        inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, [256, in_c, 1, 1], w_sl.tobytes(), raw=True))
        nodes.append(helper.make_node("Conv", ["x", w_name], [out_n], name=f"Conv_{15 + i}", kernel_shape=[1, 1]))
        w3_outs.append(out_n)

    nodes.append(helper.make_node("Concat", w3_outs, ["w3_cat"], name="Concat_1", axis=1))
    nodes.append(helper.make_node("Mul", ["silu_out", "w3_cat"], ["hidden_mul"], name="Mul_1"))

    w2_outs = []
    for i in range(16):
        c0 = i * 64
        w_sl = w2_np[c0:c0 + 64, :inter_c].astype(np.float32).reshape(64, inter_c, 1, 1)
        w_name = f"w2_s{i}.weight"
        out_n = f"w2_s{i}_out"
        inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, [64, inter_c, 1, 1], w_sl.tobytes(), raw=True))
        nodes.append(helper.make_node("Conv", ["hidden_mul", w_name], [out_n], name=f"Conv_{30 + i}", kernel_shape=[1, 1]))
        w2_outs.append(out_n)

    nodes.append(helper.make_node("Concat", w2_outs, ["y"], name="Concat_2", axis=1))

    graph = helper.make_graph(nodes, f"mlp_tp2_t{seq_len}", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LFM3-1.7B TP=2 ONNX archetypes")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/chri/aixigua/deep/models/LFM3-1.7B"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/chri/aixigua/deep/artifacts/lfm3_tp2_onnx"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(str(args.model_dir / "model.safetensors"), framework="pt", device="cpu") as sf:
        conv_in_w = sf.get_tensor("model.layers.0.conv.in_proj.weight").to(torch.float32).numpy()
        conv_out_w = sf.get_tensor("model.layers.0.conv.out_proj.weight").to(torch.float32).numpy()
        q_w = sf.get_tensor("model.layers.2.self_attn.q_proj.weight").to(torch.float32).numpy()
        k_w = sf.get_tensor("model.layers.2.self_attn.k_proj.weight").to(torch.float32).numpy()
        v_w = sf.get_tensor("model.layers.2.self_attn.v_proj.weight").to(torch.float32).numpy()
        w1 = sf.get_tensor("model.layers.0.feed_forward.w1.weight").to(torch.float32).numpy()
        w3 = sf.get_tensor("model.layers.0.feed_forward.w3.weight").to(torch.float32).numpy()
        w2 = sf.get_tensor("model.layers.0.feed_forward.w2.weight").to(torch.float32).numpy()

    conv_in_d0 = np.concatenate([conv_in_w[0:1024], conv_in_w[2048:3072], conv_in_w[4096:5120]], axis=0)
    qkv_d0 = np.concatenate([q_w[0:1024], k_w[0:256], v_w[0:256]], axis=0)
    proj_d0 = conv_out_w[0:1024, :]

    for seq_len in (1, 64):
        onnx.save(make_mlp_tp2_onnx(w1[:3840], w3[:3840], w2[:, :3840], seq_len),
                  str(args.out_dir / f"mlp_tp2_t{seq_len}.onnx"))
        onnx.save(make_conv1x1_onnx(2048, 3072, conv_in_d0, seq_len, conv_name="conv_in"),
                  str(args.out_dir / f"conv_in_tp2_t{seq_len}.onnx"))
        onnx.save(make_conv1x1_onnx(2048, 1536, qkv_d0, seq_len, conv_name="attn_qkv"),
                  str(args.out_dir / f"attn_qkv_tp2_t{seq_len}.onnx"))
        onnx.save(make_conv1x1_onnx(2048, 1024, proj_d0, seq_len, conv_name="proj2048"),
                  str(args.out_dir / f"proj2048_tp2_t{seq_len}.onnx"))
        print(f"[+] Exported TP=2 archetypes for T={seq_len} to {args.out_dir}")


if __name__ == "__main__":
    main()
