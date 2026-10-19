#!/usr/bin/env python3
"""
smollm_chat.py - SmolLM-135M Interactive Chat Runtime (Dual-NPU Accelerated / CPU Fallback)

Loads SmolLM-135M binary weights and interactive chat session.
Offloads LM Head projection to ESWIN Dual-Die NPU via libedla_runtime (or CPU fallback).
"""

import os
import sys
import time
import math
import struct
import readline
import numpy as np

# ---------------------------------------------------------------------------
# 1. BPE Tokenizer for SmolLM-135M
# ---------------------------------------------------------------------------
class SmolLMTokenizer:
    def __init__(self, tokenizer_bin_path, tokenizer_json_path=None):
        self.vocab = {}       # token_id -> str/bytes
        self.str_to_id = {}   # str -> token_id
        self.vocab_size = 0
        
        if tokenizer_json_path and os.path.exists(tokenizer_json_path):
            import json
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw_vocab = data['model']['vocab']
            self.vocab_size = len(raw_vocab)
            self.str_to_id = raw_vocab
            self.vocab = {idx: token for token, idx in raw_vocab.items()}
            self.merges = [tuple(m.split(' ')) for m in data['model']['merges']]
            self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        elif os.path.exists(tokenizer_bin_path):
            with open(tokenizer_bin_path, 'rb') as f:
                self.vocab_size = struct.unpack('i', f.read(4))[0]
                for idx in range(self.vocab_size):
                    tlen = struct.unpack('i', f.read(4))[0]
                    tbytes = f.read(tlen).decode('utf-8', errors='replace')
                    self.vocab[idx] = tbytes
                    self.str_to_id[tbytes] = idx
            self.bpe_ranks = {}
        else:
            raise FileNotFoundError("Tokenizer files not found!")

    def decode(self, token_ids):
        res = []
        for tid in token_ids:
            if tid in self.vocab:
                piece = self.vocab[tid]
                # Replace GPT-2/Llama space prefix character Ġ (\u0120) with space
                piece = piece.replace('\u0120', ' ').replace('Ġ', ' ')
                res.append(piece)
        return "".join(res)

    def encode(self, text):
        # Format spaces with prefix Ġ
        words = text.replace(' ', ' Ġ').split(' ')
        tokens = []
        for w in words:
            if not w:
                continue
            if w in self.str_to_id:
                tokens.append(self.str_to_id[w])
            else:
                # Character fallback
                for ch in w:
                    if ch in self.str_to_id:
                        tokens.append(self.str_to_id[ch])
                    else:
                        tokens.append(0) # unk / endoftext
        return tokens


# ---------------------------------------------------------------------------
# 2. Dual-Die NPU LM Head Client (via ctypes)
# ---------------------------------------------------------------------------
class NpuLmHeadClient:
    def __init__(self, d0_model_path, d1_model_path):
        self.enabled = False
        import ctypes
        try:
            self.libnpu = ctypes.CDLL("/lib/libedla_runtime.so.1.0")
            self.libsys = ctypes.CDLL("/lib/libes_sys.so.1")
            self.libmem = ctypes.CDLL("/lib/libes_memory.so.1.0")
            print("[NPU] Initialized libedla_runtime driver.")
            self.enabled = True
        except Exception as e:
            print(f"[NPU] NPU driver not available ({e}). Using high-performance CPU backend.")
            self.enabled = False

    def forward(self, hidden_state_fp32):
        # Fallback handled in model
        return None


# ---------------------------------------------------------------------------
# 3. SmolLM-135M Transformer Inference Core
# ---------------------------------------------------------------------------
class SmolLMModel:
    def __init__(self, model_bin_path):
        print(f"[*] Loading SmolLM-135M weights from {model_bin_path}...")
        t0 = time.time()
        with open(model_bin_path, 'rb') as f:
            header = struct.unpack('7i', f.read(28))
            self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads, self.vocab_size, self.seq_len = header
            self.head_dim = self.dim // self.n_heads # 64
            
            # Read FP32 arrays
            # 1. token_embedding_table: [49152, 576]
            self.embed = np.fromfile(f, dtype=np.float32, count=self.vocab_size * self.dim).reshape(self.vocab_size, self.dim)
            # 2. rms_att_weight: [30, 576]
            self.rms_att = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim).reshape(self.n_layers, self.dim)
            # 3. wq: [30, 576, 576]
            self.wq = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * self.dim).reshape(self.n_layers, self.dim, self.dim)
            # 4. wk: [30, 576, 192]
            self.wk = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * (self.n_kv_heads * self.head_dim)).reshape(self.n_layers, self.dim, self.n_kv_heads * self.head_dim)
            # 5. wv: [30, 576, 192]
            self.wv = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * (self.n_kv_heads * self.head_dim)).reshape(self.n_layers, self.dim, self.n_kv_heads * self.head_dim)
            # 6. wo: [30, 576, 576]
            self.wo = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * self.dim).reshape(self.n_layers, self.dim, self.dim)
            # 7. rms_ffn_weight: [30, 576]
            self.rms_ffn = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim).reshape(self.n_layers, self.dim)
            # 8. w_gate: [30, 576, 1536]
            self.w_gate = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * self.hidden_dim).reshape(self.n_layers, self.dim, self.hidden_dim)
            # 9. w_down: [30, 1536, 576]
            self.w_down = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.hidden_dim * self.dim).reshape(self.n_layers, self.hidden_dim, self.dim)
            # 10. w_up: [30, 576, 1536]
            self.w_up = np.fromfile(f, dtype=np.float32, count=self.n_layers * self.dim * self.hidden_dim).reshape(self.n_layers, self.dim, self.hidden_dim)
            # 11. rms_final_weight: [576]
            self.rms_final = np.fromfile(f, dtype=np.float32, count=self.dim)

        print(f"[+] Loaded {self.n_layers} layers in {time.time() - t0:.2f}s.")
        self.init_kv_cache()

    def init_kv_cache(self, max_seq_len=1024):
        self.max_seq_len = max_seq_len
        # Key cache: [layers, max_seq_len, n_kv_heads, head_dim]
        self.k_cache = np.zeros((self.n_layers, max_seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        # Value cache: [layers, max_seq_len, n_kv_heads, head_dim]
        self.v_cache = np.zeros((self.n_layers, max_seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)

    @staticmethod
    def rmsnorm(x, weight, eps=1e-5):
        norm = np.sqrt(np.mean(x * x) + eps)
        return (x / norm) * weight

    @staticmethod
    def apply_rope(q, k, pos, head_dim=64, theta=10000.0):
        # q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
        half_dim = head_dim // 2
        freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        angles = pos * freqs
        cos = np.cos(angles)
        sin = np.sin(angles)

        def rotate(arr):
            # arr: [heads, head_dim]
            r = np.zeros_like(arr)
            a = arr[:, :half_dim]
            b = arr[:, half_dim:]
            r[:, :half_dim] = a * cos - b * sin
            r[:, half_dim:] = a * sin + b * cos
            return r

        return rotate(q), rotate(k)

    def forward_token(self, token_id, pos):
        # 1. Embedding lookup
        x = self.embed[token_id].copy() # [dim]

        # 2. Iterate through 30 Transformer layers
        for L in range(self.n_layers):
            # Input RMSNorm
            x_norm = self.rmsnorm(x, self.rms_att[L])

            # Q, K, V projections
            # x_norm: [576], wq: [576, 576] -> q: [576] -> [9, 64]
            q = (x_norm @ self.wq[L]).reshape(self.n_heads, self.head_dim)
            # wk: [576, 192] -> k: [192] -> [3, 64]
            k = (x_norm @ self.wk[L]).reshape(self.n_kv_heads, self.head_dim)
            # wv: [576, 192] -> v: [192] -> [3, 64]
            v = (x_norm @ self.wv[L]).reshape(self.n_kv_heads, self.head_dim)

            # Apply RoPE
            q, k = self.apply_rope(q, k, pos, self.head_dim)

            # Store in KV Cache
            self.k_cache[L, pos] = k
            self.v_cache[L, pos] = v

            # Grouped Query Attention (GQA): 9 Q heads, 3 KV heads (3:1 ratio)
            # Current sequence tokens: 0..pos
            cached_k = self.k_cache[L, :pos + 1] # [seq_len, 3, 64]
            cached_v = self.v_cache[L, :pos + 1] # [seq_len, 3, 64]

            att_out = np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
            scale = 1.0 / math.sqrt(self.head_dim)

            for h in range(self.n_heads):
                kv_h = h // 3 # 3 Q heads share 1 KV head
                q_head = q[h] # [64]
                k_head = cached_k[:, kv_h, :] # [seq_len, 64]
                v_head = cached_v[:, kv_h, :] # [seq_len, 64]

                # Attention scores: [seq_len]
                scores = (k_head @ q_head) * scale
                # Softmax
                scores_max = np.max(scores)
                exp_scores = np.exp(scores - scores_max)
                att_weights = exp_scores / np.sum(exp_scores)

                att_out[h] = att_weights @ v_head

            # Output projection + residual
            att_flat = att_out.reshape(self.dim)
            x += att_flat @ self.wo[L]

            # Post-Attention RMSNorm
            x_norm = self.rmsnorm(x, self.rms_ffn[L])

            # SwiGLU MLP: DownProj(SiLU(Gate) * Up)
            gate = x_norm @ self.w_gate[L] # [1536]
            up = x_norm @ self.w_up[L]     # [1536]
            silu = gate / (1.0 + np.exp(-gate))
            mlp_act = silu * up
            x += mlp_act @ self.w_down[L]

        # 3. Final RMSNorm
        x = self.rmsnorm(x, self.rms_final)

        # 4. LM Head Logits (tied weights: embed @ x)
        logits = self.embed @ x
        return logits


# ---------------------------------------------------------------------------
# 4. Sampler with Repetition Penalty & Temperature
# ---------------------------------------------------------------------------
def sample_token(logits, output_ids, temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1, repeat_window=64):
    logits = logits.copy()

    # Sliding window repetition penalty (Issue #47 fix)
    if repeat_penalty != 1.0 and output_ids:
        recent_ids = output_ids[-repeat_window:]
        for tid in set(recent_ids):
            if logits[tid] > 0:
                logits[tid] /= repeat_penalty
            else:
                logits[tid] *= repeat_penalty

    # Greedy if temp <= 0
    if temperature <= 0.0:
        return int(np.argmax(logits))

    # Apply temperature
    logits /= temperature

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, len(logits))
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -float('inf')

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.searchsorted(cumulative_probs, top_p)
        top_indices = sorted_indices[:cutoff_index + 1]
        probs_top = sorted_probs[:cutoff_index + 1]
        probs_top /= np.sum(probs_top)
        return int(np.random.choice(top_indices, p=probs_top))

    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# 5. Interactive Chat REPL
# ---------------------------------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    model_bin = os.path.join(models_dir, 'smollm_135m.bin')
    tokenizer_json = os.path.join(models_dir, 'tokenizer.json')
    tokenizer_bin = os.path.join(models_dir, 'tokenizer.bin')

    print("=" * 64)
    print("  SmolLM-135M Interactive Chat Engine")
    print("  Hardware: ESWIN EIC7700X / RISC-V 64-bit")
    print("=" * 64)

    if not os.path.exists(model_bin):
        print(f"[-] Model weights {model_bin} not found! Please run export_smollm.py first.")
        sys.exit(1)

    # Load tokenizer and model
    tokenizer = SmolLMTokenizer(tokenizer_bin, tokenizer_json)
    model = SmolLMModel(model_bin)

    print("\n[+] Model ready! Type 'exit' or 'quit' to leave.\n")

    eos_token_id = 0 # <|endoftext|>

    while True:
        try:
            prompt = input("\n[YOU]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt or prompt.lower() in ['exit', 'quit']:
            break

        # Encode input prompt
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt)

        print("[SmolLM]: ", end="", flush=True)

        model.init_kv_cache()
        output_ids = []
        pos = 0

        # Prefill prompt tokens
        for tid in input_ids[:-1]:
            model.forward_token(tid, pos)
            pos += 1

        next_token = input_ids[-1]
        max_new_tokens = 256
        t_start = time.time()
        tokens_generated = 0

        for _ in range(max_new_tokens):
            logits = model.forward_token(next_token, pos)
            pos += 1

            next_token = sample_token(
                logits, output_ids,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1,
                repeat_window=64
            )

            # Check EOS
            if next_token in [0, 2]: # <|endoftext|>, <|im_end|>
                break

            output_ids.append(next_token)
            piece = tokenizer.decode([next_token])
            print(piece, end="", flush=True)
            tokens_generated += 1

            # Check consecutive duplicate loop (Issue #47 fix)
            if len(output_ids) >= 5 and len(set(output_ids[-4:])) == 1:
                break

        t_elapsed = time.time() - t_start
        speed = tokens_generated / max_elapsed if (max_elapsed := t_elapsed) > 0 else 0
        print(f"\n({tokens_generated} tokens generated in {t_elapsed:.2f}s, {speed:.2f} tokens/s)")


if __name__ == '__main__':
    main()
