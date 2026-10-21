# Porting & Running Qwen2.5-Coder-1.5B on Dual-Die EIC7700X NPU

## 1. Overview & Executive Summary

We have successfully ported, compiled, and executed **Qwen2.5-Coder-1.5B-Instruct** on the dual-die ESWIN EIC7700X (FML13V03) hardware platform using the **Open ESWIN Runtime** (`libes_runtime.so`) and our custom C++ REPL engine (`qwen2_v2`).

- **Architecture**: 28 transformer layers, 12 attention heads, 2 KV heads (GQA), hidden dimension 1536, intermediate size 8960 (SwiGLU), vocabulary size 151,936.
- **Hardware Distribution**: Fully offloaded to both NPU dies (Die 0 and Die 1):
  - **Die 0**: Layers 0 to 13 ($QKV$, Attention Out, SwiGLU MLP) + Lower Vocab LM Head slice ($76,032 \times 1536$).
  - **Die 1**: Layers 14 to 27 ($QKV$, Attention Out, SwiGLU MLP) + Upper Vocab LM Head slice ($76,032 \times 1536$).
- **Zero Proprietary Runtimes**: Zero reliance on vendor binary closed-source libraries (`libennp.so`, `libmodel_runner.so`, or Python vendor wrappers). Direct hardware communication via `/dev/npu0`, `/dev/npu1`, and `/dev/malloc_dmabuf`.
- **Performance**: Sustained **4.2 to 5.5 tokens/second** generation across all 28 layers + dual LM head, producing bit-exact, coherent code and ChatML conversational responses.

---

## 2. Key Architecture Discoveries & Parity Fixes

### 2.1 The Q, K, V Learned Projection Biases
Unlike LLaMA, SmolLM, and Mistral, the Qwen2 / Qwen2.5 architecture incorporates learned additive bias vectors in the attention projections:
$$Q = X W_q + b_q, \quad K = X W_k + b_k, \quad V = X W_v + b_v$$
Where:
- $W_q \in \mathbb{R}^{1536 \times 1536}, \quad b_q \in \mathbb{R}^{1536}$
- $W_k \in \mathbb{R}^{256 \times 1536}, \quad b_k \in \mathbb{R}^{256}$
- $W_v \in \mathbb{R}^{256 \times 1536}, \quad b_v \in \mathbb{R}^{256}$

Omitting $b_q, b_k, b_v$ caused immediate logit degradation and gibberish token emission. In `export_qwen2_5_coder.py`, we concatenate both weights and biases into a fused $1\times 1$ 2D Convolution:
$$\text{Weight}_{\text{fused}} \in \mathbb{R}^{2048 \times 1536 \times 1 \times 1}, \quad \text{Bias}_{\text{fused}} \in \mathbb{R}^{2048}$$
Compiled through `EsAAC --gen-fp16-model=yes`, `EsSimulator` verified exact bit-level alignment with PyTorch:
- **Cosine Distance**: `1.000000`
- **Relative Difference**: `0.000000`

### 2.2 RoPE Scaling & Base Frequency
Qwen2.5 uses an extended RoPE base frequency:
$$\theta_{\text{base}} = 1,000,000.0 \quad (10^6)$$
compared to standard LLaMA ($\theta = 10,000.0$). The rotary position embeddings were computed on CPU with `rope_theta = 1000000.0` and `head_dim = 128`.

### 2.3 Tokenizer BPE Space & Atomic Special Delimiters
1. **GPT-2 Byte-Fallback Decoding**:
   In HuggingFace `tokenizer.json`, the BPE merge vocabulary stores strings already decoded from unicode bytes into UTF-8 characters (e.g. `Ġ` becomes standard ASCII space `0x20`). An erroneous runtime pass that re-encoded spaces into `\xC4\xA0` was removed, restoring pristine 4-space Python indentation.
2. **Atomic Special Token Extraction**:
   Greedy longest-prefix matching could split or merge delimiters with neighboring punctuation (e.g., `.<` consuming `<|im_start|>`). We introduced atomic delimiter pre-scanning:
   ```cpp
   // Atomic special token isolation for ChatML delimiters
   static const std::vector<std::string> special_tokens = {
       "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"
   };
   ```
   Special tokens are extracted into single atomic token IDs before greedy BPE subword segmentation runs on the surrounding text.

### 2.4 Dual-Die NUMA Pipeline & Minimal Inter-Die Latency
To eliminate NUMA bus thrashing between the two RISC-V compute dies:
1. All subgraphs for Layers 0–13 are allocated strictly from NUMA node 0 memory (`/dev/npu0`).
2. All subgraphs for Layers 14–27 are allocated strictly from NUMA node 1 memory (`/dev/npu1`).
3. During autoregressive decoding, execution stays entirely on Die 0 for the first 14 layers, transfers a single 3 KB hidden state vector across dies once per token, executes Layers 14–27 on Die 1, and concurrently executes the dual-die LM head.

---

## 3. Package Structure on Board `roma`

The complete standalone model is stored in `/home/chri/models/qwen2_5_coder/`:
```
models/qwen2_5_coder/
├── manifest.json                                # Open ESWIN runtime package descriptor
├── tokenizer.json                               # 151,665 token BPE vocabulary & merges
├── embedding.bin                                # 466,747,392 bytes (151936 x 1536 FP16)
├── rms_norms.bin                                # 350,224 bytes (57 FP32 tensors)
├── qwen2_5_lm_head_d0_npu_b1.model              # Die 0 LM Head slice (76,032 x 1536)
├── qwen2_5_lm_head_d1_npu_b1.model              # Die 1 LM Head slice (76,032 x 1536)
├── qwen2_5_layer0_qkv_npu_b1.model              # Layer 0 QKV projection (Die 0)
├── qwen2_5_layer0_attn_out_npu_b1.model         # Layer 0 Attn Out projection (Die 0)
├── qwen2_5_layer0_mlp_npu_b1.model              # Layer 0 SwiGLU MLP block (Die 0)
...
├── qwen2_5_layer27_qkv_npu_b1.model             # Layer 27 QKV projection (Die 1)
├── qwen2_5_layer27_attn_out_npu_b1.model        # Layer 27 Attn Out projection (Die 1)
└── qwen2_5_layer27_mlp_npu_b1.model             # Layer 27 SwiGLU MLP block (Die 1)
```
Total model package: **86 `.model` NPU execution graphs**, **467 MB FP16 embeddings**, and **350 KB FP32 norm scales**.

---

## 4. Hardware Verification & Benchmarks

### 4.1 Numerical Parity Against Golden Reference
Using the prompt `def is_prime(n):` in raw code autocompletion mode:
- **PyTorch Golden Top 5 Logits (Step 4)**:
  1. `#198` (`\n`): $19.27$
  2. `#715` (` \n`): $16.81$
  3. `#220` (` `): $16.42$
  4. `#671` (` #`): $16.30$
  5. `#201` (`\r`): $16.16$
- **Dual-Die Hardware NPU Output**:
  Top token: `#198` (`\n`). Followed by:
  `if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):...`
  Identical token-for-token output with zero drift.

### 4.2 Benchmark Results on Hardware Silicon
| Metric | Measurement on EIC7700X (FML13V03) |
| :--- | :--- |
| **Model** | `Qwen2.5-Coder-1.5B-Instruct` |
| **Active Parameters** | 1.54 Billion (FP16 weights) |
| **Layers Offloaded to NPU** | **28 / 28 (100% hardware acceleration)** |
| **LM Head Offloaded to NPU** | **Dual-Die Parallel (76,032 vocab slice per die)** |
| **Prefill Latency** | **230 ms / prompt token** |
| **Autoregressive Decode Speed** | **4.18 – 5.49 tokens/second** |
| **Tokens Generated** | 256 tokens in 61.2 seconds |
| **Zero Memory Faults** | Stable multi-turn REPL and one-shot execution |
