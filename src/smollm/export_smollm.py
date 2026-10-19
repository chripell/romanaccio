#!/usr/bin/env python3
"""
export_smollm.py - Export SmolLM-135M ONNX model to binary format for C++ and Python runtime.

Reads goal/SmolLM-135M.onnx and exports:
1. src/smollm/models/smollm_135m.bin: Header + raw FP32 weights for all 30 layers
2. src/smollm/models/tokenizer.bin: Fast binary vocabulary for BPE decoding/encoding
"""

import os
import sys
import json
import struct
import numpy as np

def export_model_weights(onnx_path, output_bin_path):
    print(f"[*] Loading ONNX model from {onnx_path}...")
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    init_map = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}
    node_map = {n.name: n for n in model.graph.node}

    dim = 576
    hidden_dim = 1536
    n_layers = 30
    n_heads = 9
    n_kv_heads = 3
    vocab_size = 49152
    seq_len = 2048

    print(f"[*] Exporting SmolLM-135M weights (layers={n_layers}, dim={dim}, vocab={vocab_size})...")

    with open(output_bin_path, 'wb') as f:
        # Write header: 7 int32 values (28 bytes)
        header = struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        f.write(header)

        # 1. token_embedding_table: [vocab_size, dim]
        embed_tokens = init_map['model.embed_tokens.weight']
        assert embed_tokens.shape == (vocab_size, dim)
        embed_tokens.tofile(f)
        print("  -> Exported embed_tokens")

        # 2. rms_att_weight: [n_layers, dim]
        rms_att = np.zeros((n_layers, dim), dtype=np.float32)
        for L in range(n_layers):
            rms_att[L] = init_map[f'model.layers.{L}.input_layernorm.weight']
        rms_att.tofile(f)
        print("  -> Exported rms_att_weight")

        # 3. wq: [n_layers, dim, dim]
        wq = np.zeros((n_layers, dim, dim), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/self_attn/q_proj/MatMul']
            wq[L] = init_map[node.input[1]]
        wq.tofile(f)
        print("  -> Exported wq")

        # 4. wk: [n_layers, dim, n_kv_heads * 64]
        wk = np.zeros((n_layers, dim, n_kv_heads * 64), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/self_attn/k_proj/MatMul']
            wk[L] = init_map[node.input[1]]
        wk.tofile(f)
        print("  -> Exported wk")

        # 5. wv: [n_layers, dim, n_kv_heads * 64]
        wv = np.zeros((n_layers, dim, n_kv_heads * 64), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/self_attn/v_proj/MatMul']
            wv[L] = init_map[node.input[1]]
        wv.tofile(f)
        print("  -> Exported wv")

        # 6. wo: [n_layers, dim, dim]
        wo = np.zeros((n_layers, dim, dim), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/self_attn/o_proj/MatMul']
            wo[L] = init_map[node.input[1]]
        wo.tofile(f)
        print("  -> Exported wo")

        # 7. rms_ffn_weight: [n_layers, dim]
        rms_ffn = np.zeros((n_layers, dim), dtype=np.float32)
        for L in range(n_layers):
            rms_ffn[L] = init_map[f'model.layers.{L}.post_attention_layernorm.weight']
        rms_ffn.tofile(f)
        print("  -> Exported rms_ffn_weight")

        # 8. w_gate: [n_layers, dim, hidden_dim]
        w_gate = np.zeros((n_layers, dim, hidden_dim), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/mlp/gate_proj/MatMul']
            w_gate[L] = init_map[node.input[1]]
        w_gate.tofile(f)
        print("  -> Exported w_gate")

        # 9. w_down: [n_layers, hidden_dim, dim]
        w_down = np.zeros((n_layers, hidden_dim, dim), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/mlp/down_proj/MatMul']
            w_down[L] = init_map[node.input[1]]
        w_down.tofile(f)
        print("  -> Exported w_down")

        # 10. w_up: [n_layers, dim, hidden_dim]
        w_up = np.zeros((n_layers, dim, hidden_dim), dtype=np.float32)
        for L in range(n_layers):
            node = node_map[f'/model/layers.{L}/mlp/up_proj/MatMul']
            w_up[L] = init_map[node.input[1]]
        w_up.tofile(f)
        print("  -> Exported w_up")

        # 11. rms_final_weight: [dim]
        rms_final = init_map['model.norm.weight']
        assert rms_final.shape == (dim,)
        rms_final.tofile(f)
        print("  -> Exported rms_final_weight")

    file_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    print(f"[+] Successfully exported {output_bin_path} ({file_size_mb:.2f} MB)")


def export_tokenizer(tokenizer_json_path, output_bin_path):
    print(f"[*] Exporting tokenizer from {tokenizer_json_path}...")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    # Standard GPT-2 bytes_to_unicode mapping
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}

    vocab = t['model']['vocab']
    # Invert vocab: id -> string
    id_to_str = {idx: token for token, idx in vocab.items()}
    vocab_size = len(vocab)
    print(f"  -> Vocab size: {vocab_size}")

    with open(output_bin_path, 'wb') as f:
        # Write vocab size (int32)
        f.write(struct.pack('i', vocab_size))

        for idx in range(vocab_size):
            token_str = id_to_str.get(idx, "")
            # Convert GPT-2 unicode characters back to true raw bytes
            if all(c in byte_decoder for c in token_str):
                token_bytes = bytes([byte_decoder[c] for c in token_str])
            else:
                token_bytes = token_str.encode('utf-8')

            # Write len (int32) + bytes
            f.write(struct.pack('i', len(token_bytes)))
            f.write(token_bytes)

    file_size_kb = os.path.getsize(output_bin_path) / 1024
    print(f"[+] Successfully exported {output_bin_path} ({file_size_kb:.1f} KB)")


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    onnx_file = os.path.join(base_dir, 'goal', 'SmolLM-135M.onnx')
    model_bin = os.path.join(base_dir, 'src', 'smollm', 'models', 'smollm_135m.bin')
    tokenizer_json = os.path.join(base_dir, 'src', 'smollm', 'models', 'tokenizer.json')
    tokenizer_bin = os.path.join(base_dir, 'src', 'smollm', 'models', 'tokenizer.bin')

    export_model_weights(onnx_file, model_bin)
    export_tokenizer(tokenizer_json, tokenizer_bin)
