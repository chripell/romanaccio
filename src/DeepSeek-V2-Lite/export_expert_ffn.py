#!/usr/bin/env python3
"""Export one DeepSeek-V2-Lite MoE expert FFN to ONNX for EsAAC compilation.

Phase 1 of src/DeepSeek-V2-Lite/PLAN.md. One expert is a SwiGLU FFN:

    h = silu(x @ Wg^T) * (x @ Wu^T)      gate/up:  hidden -> moe_intermediate
    y = h @ Wd^T                         down:     moe_intermediate -> hidden

DeepSeek-V2-Lite geometry (from the verified config.json): hidden_size 2048,
moe_intermediate_size 1408. Shared experts use the same shape, so this script covers
both; only the weights differ, and with --convert-weights those are runtime-bound
anyway.

Follows the project's ONNX export constraints (QWEN.md / FINDINGS):
  * opset 13, static shapes, no external data file - EsAAC's MLIR/EHLO frontend
    SIGSEGVs on opset 18 or external .data.
  * 4D NCHW tensors [1, C, 1, 1] and Linear lowered to 1x1 Conv, which is what the
    NPU's CMAC executes.
  * SiLU expressed as Sigmoid * x, because opset 13 has no SiLU op and the vendor
    SwiGLU reference model uses exactly that decomposition (FINDINGS §8).

Weights are embedded as initializers on purpose: EsAAC segfaults on a graph with a
second dynamic input (FINDINGS §6.11-era probe), so runtime binding is applied
afterwards with `compiler.py --convert-weights`, not at export time.

Usage:
    python export_expert_ffn.py [--hidden 2048] [--intermediate 1408]
                                [--seed 0] [-o expert_ffn.onnx] [--zero-weights]

    # real DeepSeek-V2-Lite weights instead of the synthetic ones:
    python export_expert_ffn.py --safetensors ../../../../deep/models/DeepSeek-V2-Lite \
                                --layer 1 --expert 0 -o expert0.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13


def build_graph(hidden: int, intermediate: int, seed: int, zero_weights: bool,
                real: dict | None = None):
    """Builds the SwiGLU expert graph. Returns (graph, weight_arrays).

    @param real When given, its W_gate/W_up/W_down replace the synthetic weights.
    """
    rng = np.random.default_rng(seed)

    def w(out_c: int, in_c: int, name: str) -> np.ndarray:
        if real is not None:
            return real[name]
        if zero_weights:
            return np.zeros((out_c, in_c, 1, 1), dtype=np.float32)
        # Small init keeps activations inside fp16 range once EsAAC converts.
        return (rng.standard_normal((out_c, in_c, 1, 1)) * 0.02).astype(np.float32)

    w_gate = w(intermediate, hidden, "W_gate")
    w_up = w(intermediate, hidden, "W_up")
    w_down = w(hidden, intermediate, "W_down")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, hidden, 1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, hidden, 1, 1])

    nodes = [
        helper.make_node("Conv", ["input", "W_gate"], ["gate"], name="gate_proj",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        helper.make_node("Conv", ["input", "W_up"], ["up"], name="up_proj",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        # SiLU(gate) = gate * sigmoid(gate)
        helper.make_node("Sigmoid", ["gate"], ["gate_sig"], name="gate_sigmoid"),
        helper.make_node("Mul", ["gate", "gate_sig"], ["gate_act"], name="silu"),
        helper.make_node("Mul", ["gate_act", "up"], ["h"], name="swiglu_mul"),
        helper.make_node("Conv", ["h", "W_down"], ["output"], name="down_proj",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
    ]

    initializers = [
        numpy_helper.from_array(w_gate, name="W_gate"),
        numpy_helper.from_array(w_up, name="W_up"),
        numpy_helper.from_array(w_down, name="W_down"),
    ]

    graph = helper.make_graph(nodes, "deepseek_v2_lite_expert_ffn", [x], [y],
                              initializer=initializers)
    return graph, {"W_gate": w_gate, "W_up": w_up, "W_down": w_down}


def load_real_weights(model_dir: Path, layer: int, expert: int, hidden: int,
                      intermediate: int) -> dict:
    """Reads one expert's gate/up/down out of a HuggingFace safetensors checkpoint.

    Reads only the three tensors it needs, via the shard index -- `load_file()` would pull a
    whole 8.6 GB shard into RAM. torch is imported here rather than at module scope because
    the checkpoint is bfloat16, which numpy cannot represent, while the synthetic path needs
    neither torch nor safetensors.
    """
    import json

    import torch  # noqa: F401  (needed for the bf16 dtype safetensors returns)
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing shard index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    wanted = {
        "W_gate": (f"{prefix}.gate_proj.weight", (intermediate, hidden)),
        "W_up": (f"{prefix}.up_proj.weight", (intermediate, hidden)),
        "W_down": (f"{prefix}.down_proj.weight", (hidden, intermediate)),
    }

    handles: dict = {}
    out = {}
    for name, (tensor_name, expect) in wanted.items():
        shard = weight_map.get(tensor_name)
        if shard is None:
            raise SystemExit(f"{tensor_name} is not in the shard index -- check --layer/--expert "
                             f"(layer {layer} may be dense, not MoE)")
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
        t = handles[shard].get_tensor(tensor_name)
        if tuple(t.shape) != expect:
            raise SystemExit(f"{tensor_name} is {tuple(t.shape)}, expected {expect}")
        arr = t.to(torch.float32).numpy()
        out[name] = np.ascontiguousarray(arr.reshape(expect[0], expect[1], 1, 1))
        print(f"  {name:<7} <- {tensor_name}  {tuple(t.shape)} {str(t.dtype)} "
              f"absmax={np.abs(arr).max():.6f}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=2048, help="hidden_size (default 2048)")
    p.add_argument("--intermediate", type=int, default=1408,
                   help="moe_intermediate_size (default 1408)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", type=Path, default=Path("expert_ffn.onnx"))
    p.add_argument("--zero-weights", action="store_true",
                   help="Emit zero weights. Useful for structural/compile checks where "
                        "numerics do not matter and a small file is wanted.")
    p.add_argument("--save-weights", type=Path, default=None,
                   help="Also write the fp32 weights as an .npz for golden comparison.")
    p.add_argument("--safetensors", type=Path, default=None,
                   help="Checkpoint directory to read real weights from (needs "
                        "model.safetensors.index.json). Overrides --seed/--zero-weights.")
    p.add_argument("--layer", type=int, default=1,
                   help="Transformer layer to take the expert from (default 1; layer 0 is "
                        "dense because first_k_dense_replace=1)")
    p.add_argument("--expert", type=int, default=0, help="Routed expert index (default 0)")
    args = p.parse_args()

    real = None
    if args.safetensors:
        print(f"reading real weights: layer {args.layer} expert {args.expert} "
              f"from {args.safetensors}")
        real = load_real_weights(args.safetensors, args.layer, args.expert, args.hidden,
                                 args.intermediate)

    graph, weights = build_graph(args.hidden, args.intermediate, args.seed,
                                args.zero_weights, real)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)

    n_params = sum(v.size for v in weights.values())
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"  opset={OPSET} hidden={args.hidden} intermediate={args.intermediate} "
          f"params={n_params:,} ({n_params * 4 / 1e6:.1f} MB fp32)")
    print("  nodes: " + ", ".join(n.op_type for n in graph.node))
    print("  next: EsAAC --input-model=%s --gen-fp16-model=yes -o expert_ffn.model"
          % args.output.name)

    if args.save_weights:
        np.savez(str(args.save_weights), **weights)
        print(f"  weights saved to {args.save_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
