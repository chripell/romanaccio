#!/usr/bin/env python3
"""Attributes the INT8 accuracy gap between weight quantization and activation quantization.

The engine is provably correct: FP16 gives logits cosine 0.99999267 vs PyTorch FP32 and
reproduces bit-identically (FINDINGS 43.1). The INT8 build gives min-layer 0.729 / logits 0.887
(43.2), and making `proj2048` *more* precise raised min-layer while lowering logits -- a
non-monotonic response that ordinary accumulated error does not produce. So either the weights,
the activations, or their interaction is at fault, and the fix differs completely depending on
which.

Four configurations, same prompt, same fp32 reference, measured per layer and on the logits:

  A baseline    fp32 weights, fp32 activations            (the reference itself)
  B weights     per-channel INT8 weights, fp32 activations (what "8-bit weights alone" costs)
  C activations fp32 weights, engine boundary quantization (what the tables + dyntok cost alone)
  D both        = what the deployed INT8 package does

An external pre-quantized checkpoint (OpenVINO int8-ov, GGUF Q8) answers only question B; this
answers B, C and D locally with no download and no board time. If B is ~0.999 and C is ~0.73,
the fault is in boundary handling or the calibrated thresholds, not in weight precision -- and
`maxdiff = 0` on the NVDLA tiler cannot catch that, because it validates byte layout, not the
scales the tiler was handed.

Usage:
    python attribute_int8_error.py [--tokens N] [--scales PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
GOLDEN_PATH = Path("/home/chri/aixigua/deep/artifacts/lfm3_golden/golden_ref.npz")
TABLES_DIR = Path("/home/chri/aixigua/deep/artifacts/lfm3_quant_tables")
DEFAULT_SCALES = MODEL_DIR / "eswin_pkg" / "quant_scales.json"
FIXED_SCALES = MODEL_DIR / "eswin_pkg_int8" / "quant_scales.json"
FIXED_CHAN_BIN = MODEL_DIR / "eswin_pkg_int8" / "quant_chan_scales.bin"
# Linear modules that correspond to an NPU subgraph boundary in the engine.
ROLE_OF_ATTR = {
    "conv.in_proj": "conv_in",
    "conv.out_proj": "conv_out",
    "self_attn.q_proj": "attn_qkv",
    "self_attn.k_proj": "attn_qkv",
    "self_attn.v_proj": "attn_qkv",
    "self_attn.o_proj": "attn_out",
    "feed_forward.w1": "mlp",
    "feed_forward.w3": "mlp",
    "feed_forward.w2": "mlp",
}


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    denom = float(a.norm() * b.norm())
    return float((a @ b) / denom) if denom > 0 else 0.0


def quantize_weight_per_channel(w: torch.Tensor) -> torch.Tensor:
    amax = torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-8)
    step = amax / 127.0
    return torch.clamp(torch.round(w / step), -127.0, 127.0) * step


def quantize_weight_int16_per_channel(w: torch.Tensor) -> torch.Tensor:
    amax = torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-8)
    step = amax / 32767.0
    return torch.clamp(torch.round(w / step), -32767.0, 32767.0) * step


def sim_boundary(x: torch.Tensor, w: torch.Tensor, in_thresh: float, out_thresh: float,
                 out_int16: bool) -> torch.Tensor:
    """The engine's dyntok boundary: per-token input scale against a static table threshold."""
    x_amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    in_t = max(in_thresh, 1e-8)
    ratio = x_amax / in_t
    x_q = torch.clamp(torch.round(x / (ratio * (in_t / 127.0))), -127.0, 127.0)
    y = F.linear(x_q * (in_t / 127.0), w)
    out_div = 32767.0 if out_int16 else 127.0
    out_step = max(out_thresh, 1e-8) / out_div
    y_q = torch.clamp(torch.round(y / out_step), -out_div, out_div)
    return y_q * (ratio * out_step)


def sim_mlp_npu(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor,
                tbl: dict, quant_acts: bool) -> torch.Tensor:
    """The engine's fused SwiGLU MLP subgraph: static INT8 input + NVDLA QuantM + INT16 internals."""
    if not quant_acts:
        return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)
    in_s  = float(tbl["blob_info"]["input"]["int8_step"])
    c0_s  = float(tbl["blob_info"]["/Conv_output_0"]["int16_step"])
    c1_s  = float(tbl["blob_info"]["/Conv_1_output_0"]["int16_step"])
    s0_s  = float(tbl["blob_info"]["/Sigmoid_output_0"]["int16_step"])
    m0_s  = float(tbl["blob_info"]["/Mul_output_0"]["int16_step"])
    m1_s  = float(tbl["blob_info"]["/Mul_1_output_0"]["int16_step"])
    out_s = float(tbl["blob_info"]["output"]["int16_step"])

    x_q = torch.clamp(torch.round(x.double() / in_s), -127.0, 127.0)
    ws1 = w1.double().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    wq1 = torch.clamp(torch.round(w1.double() / ws1), -127.0, 127.0)
    qm1 = torch.round((in_s * ws1[:, 0] / c0_s) * (1 << 14))
    c0_q = torch.clamp(torch.round(F.linear(x_q, wq1) * (qm1 / float(1 << 14))), -32767.0, 32767.0)

    ws3 = w3.double().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    wq3 = torch.clamp(torch.round(w3.double() / ws3), -127.0, 127.0)
    qm3 = torch.round((in_s * ws3[:, 0] / c1_s) * (1 << 15))
    c1_q = torch.clamp(torch.round(F.linear(x_q, wq3) * (qm3 / float(1 << 15))), -32767.0, 32767.0)

    sig_q = torch.clamp(torch.round(torch.sigmoid(c0_q * c0_s) / s0_s), -32767.0, 32767.0)
    m0_q = torch.clamp(torch.round((c0_q * c0_s) * (sig_q * s0_s) / m0_s), -32767.0, 32767.0)
    m1_q = torch.clamp(torch.round((m0_q * m0_s) * (c1_q * c1_s) / m1_s), -32767.0, 32767.0)

    ws2 = w2.double().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 32767.0
    wq2 = torch.clamp(torch.round(w2.double() / ws2), -32767.0, 32767.0)
    qm2 = torch.round((m1_s * ws2[:, 0] / out_s) * (1 << 28))
    out_q = torch.clamp(torch.round(F.linear(m1_q, wq2) * (qm2 / float(1 << 28))), -32767.0, 32767.0)
    return (out_q * out_s).float()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=6, help="prompt tokens to feed (default 6)")
    p.add_argument("--scales", type=Path, default=DEFAULT_SCALES)
    args = p.parse_args()

    if not args.scales.is_file():
        print(f"missing scale table {args.scales}")
        return 1
    sub = json.loads(args.scales.read_text())["subgraphs"]
    sh_mlp_tbl = json.loads((TABLES_DIR / "shared_mlp_table.json").read_text())
    cfg = json.loads((MODEL_DIR / "config.json").read_text())

    print(f"[+] {len(sub)} subgraph scale entries; prompt = {args.tokens} golden tokens")
    gold = np.load(str(GOLDEN_PATH))
    ids = torch.from_numpy(gold["input_ids"].reshape(1, -1)[:, : args.tokens]).long()

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), trust_remote_code=True,
                                                 dtype=torch.float32)
    model.eval()

    # Collect the Linears that map onto NPU boundaries, and snapshot original weights.
    targets = []
    for l_idx, layer in enumerate(model.model.layers):
        for attr, role in ROLE_OF_ATTR.items():
            mod = layer
            ok = True
            for part in attr.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if not ok or not hasattr(mod, "weight"):
                continue
            key = f"lfm3_l{l_idx:02d}_{role}"
            if key not in sub:
                print(f"[!] no scale entry for {key}; skipping that boundary")
                continue
            targets.append((l_idx, attr, role, mod, key, mod.weight.data.clone()))

    def install(quant_weights: bool, quant_acts: bool, active_roles: set[str] | None = None):
        for layer in model.model.layers:
            for h in list(getattr(layer.feed_forward, "_qdq_hooks", []) or []):
                h.remove()
            layer.feed_forward._qdq_hooks = []
        for (l_idx, attr, role, mod, key, w0) in targets:
            role_active = (active_roles is None) or (role in active_roles)
            if quant_weights and role_active and not (quant_acts and role == "mlp"):
                mod.weight.data = (quantize_weight_int16_per_channel(w0)
                                   if key.endswith("_mlp") and attr.endswith("w2")
                                   else quantize_weight_per_channel(w0))
            else:
                mod.weight.data = w0
            for h in list(getattr(mod, "_qdq_hooks", []) or []):
                h.remove()
            mod._qdq_hooks = []
            if quant_acts and role_active and role != "mlp":
                e = sub[key]
                in_t = float(e["input_threshold"])
                out_t = float(e["output_threshold"])
                out16 = str(e.get("output_dtype", "")).lower() in ("int16", "int16_t")

                def hook(m, inp, out, in_t=in_t, out_t=out_t, out16=out16):
                    x = inp[0]
                    return sim_boundary(x, m.weight, in_t, out_t, out16)

                mod._qdq_hooks = [mod.register_forward_hook(hook)]

        if quant_acts and (active_roles is None or "mlp" in active_roles):
            for layer in model.model.layers:
                ff = layer.feed_forward
                ff._qdq_hooks = [ff.register_forward_hook(
                    lambda m, inp, out: sim_mlp_npu(inp[0], m.w1.weight, m.w3.weight, m.w2.weight,
                                                    sh_mlp_tbl, True))]

    def run(label: str, qw: bool, qa: bool, roles: set[str] | None = None):
        install(qw, qa, roles)
        layers = {}
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        logits = out.logits[0, -1, :].float().cpu()
        for i, hs in enumerate(out.hidden_states[1:]):   # block_out per layer
            layers[i] = hs[0, -1, :].float().cpu()
        print(f"[+] {label}: captured {len(layers)} layers, logits {tuple(logits.shape)}")
        return layers, logits

    base_layers, base_logits = run("A baseline (fp32)", False, False)
    w_layers, w_logits = run("B weights only (int8/16 per channel)", True, False)
    a_layers, a_logits = run("C activations only (engine boundaries)", False, True)
    d_layers, d_logits = run("D both (= deployed INT8 package)", True, True)
    m_layers, m_logits = run("E mixed (= FP16 proj2048, rest INT8)", True, True,
                             {"conv_in", "attn_qkv", "mlp"})
    c_out_l, c_out_lg  = run("F1 conv_out only (INT8)", True, True, {"conv_out"})
    mlp_l,   mlp_lg    = run("F2 mlp only (INT8/INT16)", True, True, {"mlp"})

    def summarise(name, layers, logits, ref_layers, ref_logits):
        per_layer = {i: cos(layers[i], ref_layers[i]) for i in ref_layers}
        worst_i = min(per_layer, key=per_layer.get)
        print(f"\n  {name}")
        print(f"    min-layer cosine      : {per_layer[worst_i]:.8f}  (layer {worst_i:02d})")
        print(f"    first layer below .99 : "
              f"{next((i for i in sorted(per_layer) if per_layer[i] < 0.99), None)}")
        print(f"    logits cosine         : {cos(logits, ref_logits):.8f}")
        return per_layer

    print("\n=== attribution (each vs the fp32 baseline, like the engine gate) ===")
    summarise("B weights only          ", w_layers, w_logits, base_layers, base_logits)
    summarise("C activations only      ", a_layers, a_logits, base_layers, base_logits)
    summarise("D both (orig INT8 pkg)  ", d_layers, d_logits, base_layers, base_logits)
    summarise("E mixed (FP16 proj2048) ", m_layers, m_logits, base_layers, base_logits)
    summarise("F1 conv_out only (INT8) ", c_out_l, c_out_lg, base_layers, base_logits)
    summarise("F2 mlp only (INT8/16)   ", mlp_l, mlp_lg, base_layers, base_logits)
    print("\n  engine orig INT8 pkg on roma  : min-layer 0.72907964, logits 0.88736760")
    print("  engine MIXED pkg on roma      : min-layer 0.85739772, logits 0.81591145")
    print("  engine fixed INT8 pkg on roma : min-layer 0.98464613, logits 0.99347687 (4/4 greedy PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
