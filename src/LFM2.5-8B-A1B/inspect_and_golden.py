#!/usr/bin/env python3
"""
Minimal Golden Reference Generation for LiquidAI/LFM2.5-8B-A1B.
Milestone 15 - Phase A.

Runs a prompt through the reference model on host CPU, captures:
- Final logits and greedy next tokens
- Intermediate MoE router decisions (selected experts and gating weights per layer)
- Generates 10 greedy tokens to verify coherent text generation
- Saves reference outputs to golden_ref.json and golden_ref.npz
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM2.5-8B-A1B")
ARTIFACTS_DIR = Path("/home/chri/aixigua/deep/artifacts/lfm2_moe_golden")
OUTPUT_JSON = ARTIFACTS_DIR / "golden_ref.json"
OUTPUT_NPZ = ARTIFACTS_DIR / "golden_ref.npz"

PROMPT = "Hello, world! Today is"

def main():
    print("=" * 80)
    print(" LiquidAI / LFM2.5-8B-A1B: Golden Reference Oracle")
    print("=" * 80)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer & Tokenize Prompt
    print(f"\n[1/5] Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    seq_len = input_ids.shape[1]
    print(f"  Prompt:        '{PROMPT}'")
    print(f"  Input IDs:     {input_ids[0].tolist()} (seq_len={seq_len})")
    print(f"  Token Strings: {tokens_list}")

    # 2. Load Model on Host CPU
    print(f"\n[2/5] Loading PyTorch model from {MODEL_DIR} on CPU (torch_dtype=bfloat16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model.eval()
    t1 = time.time()
    print(f"  Model loaded in {t1 - t0:.2f}s.")

    # 3. Attach Hooks to MoE Routers to Capture Routing Top-4 Experts
    print("\n[3/5] Setting up routing instrumentation hooks...")
    routing_log = {}

    def make_router_hook(layer_idx):
        def hook_fn(module, inp, out):
            # Lfm2MoeTopKRouter.forward returns: (router_logits, routing_weights, selected_experts)
            # shape of routing_weights: [batch, seq_len, top_k]
            # shape of selected_experts: [batch, seq_len, top_k]
            _, routing_weights, selected_experts = out
            routing_log[layer_idx] = {
                "selected_experts": selected_experts.detach().cpu().numpy(),
                "routing_weights": routing_weights.detach().cpu().to(torch.float32).numpy()
            }
        return hook_fn

    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer.feed_forward, "gate"):
            layer.feed_forward.gate.register_forward_hook(make_router_hook(idx))

    print(f"  Instrumented {len(routing_log) or 'all'} MoE router layers (Layers 2..23).")

    # 4. Run Forward Pass on Prompt
    print(f"\n[4/5] Executing forward pass on prompt ({seq_len} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    t1 = time.time()
    print(f"  Forward pass completed in {t1 - t0:.2f}s.")

    last_logits = outputs.logits[0, -1, :].to(torch.float32)
    probs = F.softmax(last_logits, dim=-1)

    topk_vals, topk_indices = torch.topk(last_logits, k=10)
    topk_probs = probs[topk_indices]

    print("\n  Top-10 Predicted Next Tokens:")
    print("  " + "-" * 55)
    print(f"  {'Rank':<5} | {'Token ID':<10} | {'Token':<18} | {'Logit':<8} | {'Prob %'}")
    print("  " + "-" * 55)
    for rank in range(10):
        tid = topk_indices[rank].item()
        tstr = repr(tokenizer.decode([tid]))
        lval = topk_vals[rank].item()
        pval = topk_probs[rank].item() * 100.0
        print(f"  {rank+1:<5} | {tid:<10} | {tstr:<18} | {lval:>8.3f} | {pval:>6.2f}%")
    print("  " + "-" * 55)

    greedy_next_token_id = topk_indices[0].item()
    greedy_next_token_str = tokenizer.decode([greedy_next_token_id])
    print(f"  Greedy next token: ID={greedy_next_token_id} -> '{greedy_next_token_str}'")

    # 5. Greedy Autoregressive Generation (10 tokens)
    print("\n[5/5] Generating 10 greedy continuation tokens...")
    curr_ids = input_ids.clone()
    generated_tokens = []
    
    t0 = time.time()
    with torch.no_grad():
        for step in range(10):
            out_step = model(curr_ids)
            next_id = torch.argmax(out_step.logits[0, -1, :]).item()
            generated_tokens.append(next_id)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]])], dim=1)
    t1 = time.time()

    full_output_text = tokenizer.decode(curr_ids[0].tolist())
    continuation_text = tokenizer.decode(generated_tokens)
    print(f"  Greedy continuation: '{continuation_text}'")
    print(f"  Full generated text: '{full_output_text}'")
    print(f"  Time for 10 tokens: {t1 - t0:.2f}s ({10 / (t1 - t0):.2f} tok/s on CPU)")

    # 6. Format and Save Artifacts
    print(f"\nSaving golden artifacts to {OUTPUT_JSON} and {OUTPUT_NPZ}...")

    # Convert routing log to serializable format
    serializable_routing = {}
    routing_npz_arrays = {}
    for layer_idx, rdata in routing_log.items():
        sel = rdata["selected_experts"] # [seq_len, 4] or [batch, seq_len, 4]
        wts = rdata["routing_weights"]   # [seq_len, 4] or [batch, seq_len, 4]
        routing_npz_arrays[f"layer_{layer_idx}_selected_experts"] = sel
        routing_npz_arrays[f"layer_{layer_idx}_routing_weights"] = wts
        
        last_sel = sel[-1].tolist() if sel.ndim == 2 else sel[0, -1].tolist()
        last_wts = wts[-1].tolist() if wts.ndim == 2 else wts[0, -1].tolist()
        serializable_routing[f"layer_{layer_idx}"] = {
            "last_token_selected_experts": last_sel,
            "last_token_routing_weights": last_wts
        }

    # Summary JSON
    golden_summary = {
        "model_id": "LiquidAI/LFM2.5-8B-A1B",
        "prompt": PROMPT,
        "input_ids": input_ids[0].tolist(),
        "seq_len": seq_len,
        "top10_next_tokens": [
            {
                "rank": rank + 1,
                "token_id": topk_indices[rank].item(),
                "token_str": tokenizer.decode([topk_indices[rank].item()]),
                "logit": float(topk_vals[rank].item()),
                "probability": float(topk_probs[rank].item())
            }
            for rank in range(10)
        ],
        "greedy_continuation_ids": generated_tokens,
        "greedy_continuation_text": continuation_text,
        "full_text": full_output_text,
        "sample_routing_decisions": serializable_routing
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(golden_summary, f, indent=2)

    # NPZ with exact tensors
    np.savez_compressed(
        OUTPUT_NPZ,
        input_ids=input_ids.numpy(),
        last_logits=last_logits.numpy(),
        top10_logits=topk_vals.numpy(),
        top10_indices=topk_indices.numpy(),
        generated_token_ids=np.array(generated_tokens, dtype=np.int64),
        **routing_npz_arrays
    )

    print(f"  Golden reference successfully saved to:")
    print(f"  • JSON: {OUTPUT_JSON}")
    print(f"  • NPZ:  {OUTPUT_NPZ}")
    print("=" * 80)
    print(" Golden Reference Generation: COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
