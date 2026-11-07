#!/usr/bin/env python3
"""
Milestone 9: Export T=8 ([1, C, 8, 1]) Class B Archetype ONNX Graphs and EsAAC Configs
for LFM3-1.7B Speculative Decoding (both PP=2 and TP=2).

Exports:
  1. conv_in_t8.onnx   ([1, 2048, 8, 1] -> [1, 6144, 8, 1])
  2. proj2048_t8.onnx  ([1, 2048, 8, 1] -> [1, 2048, 8, 1])
  3. attn_qkv_t8.onnx  ([1, 2048, 8, 1] -> [1, 3072, 8, 1])
  4. mlp_t8.onnx       ([1, 2048, 8, 1] -> 7680 -> [1, 2048, 8, 1], PP=2 full-width SwiGLU MLP)
  5. mlp_tp2_t8.onnx   ([1, 2048, 8, 1] -> 3840 -> [1, 2048, 8, 1], TP=2 half-width SwiGLU MLP)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from safetensors import safe_open

DEFAULT_MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
DEFAULT_TABLES_DIR = Path("/home/chri/aixigua/deep/models/LFM3-1.7B/quant_tables")
DEFAULT_OUT_DIR = Path("/run/host-wayland/simulator-ipc/lfm3_t8_work")


def make_1x1_conv_onnx(name: str, w_fp16: np.ndarray, T: int, out_path: Path):
    c_out, c_in = w_fp16.shape
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, c_in, T, 1])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, c_out, T, 1])
    w_shape = [c_out, c_in, 1, 1]
    w_init = helper.make_tensor("w", TensorProto.FLOAT16, w_shape, w_fp16.tobytes(), raw=True)
    node = helper.make_node("Conv", inputs=["x", "w"], outputs=["y"], kernel_shape=[1, 1])
    graph = helper.make_graph([node], name, [x_info], [y_info], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 7
    onnx.save(model, str(out_path))


def make_mlp_onnx(name: str, w1_fp16: np.ndarray, w3_fp16: np.ndarray, w2_fp16: np.ndarray, T: int, out_path: Path):
    interm, hidden = w1_fp16.shape
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, hidden, T, 1])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, hidden, T, 1])

    w1_init = helper.make_tensor("w1", TensorProto.FLOAT16, [interm, hidden, 1, 1], w1_fp16.tobytes(), raw=True)
    w3_init = helper.make_tensor("w3", TensorProto.FLOAT16, [interm, hidden, 1, 1], w3_fp16.tobytes(), raw=True)
    w2_init = helper.make_tensor("w2", TensorProto.FLOAT16, [hidden, interm, 1, 1], w2_fp16.tobytes(), raw=True)

    n_conv1 = helper.make_node("Conv", inputs=["x", "w1"], outputs=["/Conv_output_0"], name="/Conv", kernel_shape=[1, 1])
    n_sig   = helper.make_node("Sigmoid", inputs=["/Conv_output_0"], outputs=["/Sigmoid_output_0"], name="/Sigmoid")
    n_mul0  = helper.make_node("Mul", inputs=["/Conv_output_0", "/Sigmoid_output_0"], outputs=["/Mul_output_0"], name="/Mul")
    n_conv3 = helper.make_node("Conv", inputs=["x", "w3"], outputs=["/Conv_1_output_0"], name="/Conv_1", kernel_shape=[1, 1])
    n_mul1  = helper.make_node("Mul", inputs=["/Mul_output_0", "/Conv_1_output_0"], outputs=["/Mul_1_output_0"], name="/Mul_1")
    n_conv2 = helper.make_node("Conv", inputs=["/Mul_1_output_0", "w2"], outputs=["y"], name="/Conv_2", kernel_shape=[1, 1])

    graph = helper.make_graph(
        [n_conv1, n_sig, n_mul0, n_conv3, n_mul1, n_conv2],
        name,
        [x_info],
        [y_info],
        [w1_init, w3_init, w2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 7
    onnx.save(model, str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Export LFM3-1.7B T=8 Class B Archetype ONNX graphs")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-t", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    T = args.batch_t

    st = safe_open(str(args.model_dir / "model.safetensors"), framework="pt")
    w_conv_in  = np.ascontiguousarray(st.get_tensor("model.layers.0.conv.in_proj.weight").float().numpy().astype(np.float16))
    w_proj2048 = np.ascontiguousarray(st.get_tensor("model.layers.0.conv.out_proj.weight").float().numpy().astype(np.float16))
    w_q = st.get_tensor("model.layers.2.self_attn.q_proj.weight").float().numpy().astype(np.float16)
    w_k = st.get_tensor("model.layers.2.self_attn.k_proj.weight").float().numpy().astype(np.float16)
    w_v = st.get_tensor("model.layers.2.self_attn.v_proj.weight").float().numpy().astype(np.float16)
    w_attn_qkv = np.ascontiguousarray(np.concatenate([w_q, w_k, w_v], axis=0))

    w1 = np.ascontiguousarray(st.get_tensor("model.layers.1.feed_forward.w1.weight").float().numpy().astype(np.float16))
    w3 = np.ascontiguousarray(st.get_tensor("model.layers.1.feed_forward.w3.weight").float().numpy().astype(np.float16))
    w2 = np.ascontiguousarray(st.get_tensor("model.layers.1.feed_forward.w2.weight").float().numpy().astype(np.float16))

    make_1x1_conv_onnx(f"conv_in_t{T}",  w_conv_in,  T, out_dir / f"conv_in_t{T}.onnx")
    make_1x1_conv_onnx(f"proj2048_t{T}", w_proj2048, T, out_dir / f"proj2048_t{T}.onnx")
    make_1x1_conv_onnx(f"attn_qkv_t{T}", w_attn_qkv, T, out_dir / f"attn_qkv_t{T}.onnx")
    make_mlp_onnx(f"mlp_t{T}",     w1, w3, w2, T, out_dir / f"mlp_t{T}.onnx")
    make_mlp_onnx(f"mlp_tp2_t{T}", np.ascontiguousarray(w1[:3840]), np.ascontiguousarray(w3[:3840]),
                  np.ascontiguousarray(w2[:, :3840]), T, out_dir / f"mlp_tp2_t{T}.onnx")
    print(f"[+] Exported 5 T={T} ONNX archetype graphs to {out_dir}")


if __name__ == "__main__":
    main()
