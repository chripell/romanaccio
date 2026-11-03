#!/usr/bin/env python3
"""
Milestone 1: Architecture Inspector & Golden Reference Oracle for LFM3-1.7B.
Runs a 5-token prompt through PyTorch and ONNX Runtime, verifies numerical
parity, captures per-layer intermediate states, and dumps golden_ref.npz.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import onnxruntime as ort

MODEL_DIR = "/home/chri/aixigua/deep/models/LFM3-1.7B"
ONNX_DIR = "/home/chri/aixigua/deep/models/LFM3-1.7B-ONNX/onnx"
ARTIFACTS_DIR = "/home/chri/aixigua/deep/artifacts/lfm3_golden"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def main():
    print("=================================================================")
    print(" LFM3-1.7B Architecture Inspector & Golden Reference Oracle")
    print("=================================================================")

    # 1. Assert Configuration Hyperparameters
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    print("\n[1/5] Validating config.json assertions...")
    assert cfg.get("model_type") == "lfm3", f"Unexpected model_type: {cfg.get('model_type')}"
    assert cfg.get("hidden_size") == 2048, f"Unexpected hidden_size: {cfg.get('hidden_size')}"
    assert cfg.get("intermediate_size") == 7680, f"Unexpected intermediate_size: {cfg.get('intermediate_size')}"
    assert cfg.get("num_hidden_layers") == 24, f"Unexpected num_hidden_layers: {cfg.get('num_hidden_layers')}"
    assert cfg.get("num_attention_heads") == 16, f"Unexpected num_attention_heads: {cfg.get('num_attention_heads')}"
    assert cfg.get("num_key_value_heads") == 4, f"Unexpected num_key_value_heads: {cfg.get('num_key_value_heads')}"
    assert cfg.get("vocab_size") == 128000, f"Unexpected vocab_size: {cfg.get('vocab_size')}"
    assert cfg.get("conv_L_cache") == 7, f"Unexpected conv_L_cache: {cfg.get('conv_L_cache')}"
    assert cfg.get("sliding_window") == 256, f"Unexpected sliding_window: {cfg.get('sliding_window')}"
    assert cfg.get("use_sandwich_norm") is True, "use_sandwich_norm not True"

    layer_types = cfg.get("layer_types", [])
    conv_count = sum(1 for t in layer_types if t == "conv")
    swa_count = sum(1 for t in layer_types if t == "sliding_attention")
    full_count = sum(1 for t in layer_types if t == "full_attention")
    print(f"  Hidden size: {cfg['hidden_size']} (multiple of 256)")
    print(f"  Intermediate size: {cfg['intermediate_size']} (multiple of 256)")
    print(f"  Vocab size: {cfg['vocab_size']} (64,000 per die, exact 256-alignment)")
    print(f"  Layer counts: {conv_count} conv, {swa_count} sliding_attention, {full_count} full_attention (total {len(layer_types)})")
    assert conv_count == 13 and swa_count == 5 and full_count == 6, "Layer schedule mismatch!"
    print("  Assertions: PASSED.")

    # 2. Tokenize Prompt
    print("\n[2/5] Initializing Tokenizer & Preparing Prompt...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    prompt = "Hello, world! Today is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    print(f"  Prompt: '{prompt}'")
    print(f"  Token IDs: {input_ids.tolist()[0]} (seq_len={seq_len})")

    # 3. Load PyTorch Model & Attach Intermediate Hooks
    print("\n[3/5] Loading PyTorch Model & Hooking Layer Activations...")
    pt_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float32)
    pt_model.eval()

    hooks_data = {}

    def get_hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                t = out[0]
            else:
                t = out
            hooks_data[name] = t.detach().cpu().to(torch.float32).numpy()
        return fn

    # Hook embedding
    pt_model.model.embed_tokens.register_forward_hook(get_hook("embed_tokens"))

    # Hook each layer's key points
    for idx, layer in enumerate(pt_model.model.layers):
        layer.operator_norm.register_forward_hook(get_hook(f"layer_{idx}_op_norm"))
        if layer.is_attention_layer:
            layer.self_attn.register_forward_hook(get_hook(f"layer_{idx}_attn_out"))
        else:
            layer.conv.register_forward_hook(get_hook(f"layer_{idx}_conv_out"))
        if hasattr(layer, "post_operator_norm"):
            layer.post_operator_norm.register_forward_hook(get_hook(f"layer_{idx}_post_op_norm"))
        layer.ffn_norm.register_forward_hook(get_hook(f"layer_{idx}_ffn_norm"))
        layer.feed_forward.register_forward_hook(get_hook(f"layer_{idx}_mlp_out"))
        if hasattr(layer, "post_ffn_norm"):
            layer.post_ffn_norm.register_forward_hook(get_hook(f"layer_{idx}_post_ffn_norm"))
        layer.register_forward_hook(get_hook(f"layer_{idx}_block_out"))

    pt_model.model.embedding_norm.register_forward_hook(get_hook("embedding_norm"))

    with torch.no_grad():
        pt_outputs = pt_model(input_ids)
        pt_logits = pt_outputs.logits.detach().cpu().to(torch.float32).numpy()

    print(f"  PyTorch forward pass completed. Logits shape: {pt_logits.shape}")

    # 4. Run ONNX Runtime FP16 Model for Parity Verification
    print("\n[4/5] Executing ONNX Runtime FP16 Model for Differential Parity...")
    onnx_model_path = os.path.join(ONNX_DIR, "model_fp16.onnx")
    sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # Prepare ORT inputs
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
        "logits_to_keep": np.array([seq_len], dtype=np.int64),
    }

    # Initialize zero past states for prefill
    for inp in sess.get_inputs():
        if inp.name.startswith("past_conv."):
            ort_inputs[inp.name] = np.zeros((1, 2048, 7), dtype=np.float16)
        elif inp.name.startswith("past_key_values."):
            ort_inputs[inp.name] = np.zeros((1, 4, 0, 128), dtype=np.float16)

    ort_outputs = sess.run(None, ort_inputs)
    ort_logits = ort_outputs[0].astype(np.float32)

    cos_sim = cosine_similarity(pt_logits, ort_logits)
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"  PyTorch vs. ONNX Runtime Logits Cosine Similarity: {cos_sim:.6f}")
    print(f"  Max Absolute Logit Difference: {max_diff:.6f}")
    assert cos_sim >= 0.9995, f"Cosine similarity {cos_sim} below acceptance threshold 0.9995!"
    print("  Differential verification: PASSED (cos >= 0.9995)")

    # 5. Greedy Autoregressive 4-Token Generation
    print("\n[5/5] Generating 4 Autoregressive Greedy Tokens...")
    curr_ids = input_ids.clone()
    generated_tokens = []

    with torch.no_grad():
        for step in range(4):
            out = pt_model(curr_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            token_str = tokenizer.decode([token_id])
            generated_tokens.append(token_id)
            print(f"  Step {step+1}: token_id={token_id} -> '{token_str}'")
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

    # 6. Save golden_ref.npz
    out_file = os.path.join(ARTIFACTS_DIR, "golden_ref.npz")
    print(f"\nSaving golden reference archive to {out_file}...")
    np.savez_compressed(
        out_file,
        input_ids=input_ids.numpy(),
        prompt_text=prompt,
        pt_logits=pt_logits,
        ort_logits=ort_logits,
        generated_token_ids=np.array(generated_tokens, dtype=np.int64),
        **hooks_data
    )
    print(f"Golden oracle archived successfully ({len(hooks_data) + 5} arrays saved).")
    print("\nMILESTONE 1 COMPLETE: ACCEPTANCE GATES SATISFIED.\n")

if __name__ == "__main__":
    main()
