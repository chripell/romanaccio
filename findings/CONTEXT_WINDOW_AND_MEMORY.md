# Context Window Scaling, KV Cache Memory & Output Size Analysis

## 1. Executive Summary & Problem Statement

In early vendor prototypes (such as `deepseek_7b_1k_int8_peer`) and initial test engines, both the context window ($L_{ctx}$) and maximum output size ($L_{out}$) were hard-coded to **1024 tokens**. 

For real-world workflows, a 1024-token window is severely restrictive:
1. **Coding & Software Engineering**: A single source file with header inclusions and user query easily exceeds 2,000–4,000 tokens. Multi-file context or git diffs routinely require **16,384 tokens (16K)** or more.
2. **Roleplaying & Creative Writing (RPG)**: Character cards, world lorebooks, scenario rules, and multi-turn chat history rapidly fill 4,096 to 8,192 tokens.
3. **Chain-of-Thought Reasoning (DeepSeek-R1)**: The `<think> ... </think>` block frequently consumes 400 to 800 tokens on its own. When the total output limit is 1024, the actual visible answer is truncated after 200–400 tokens.

This report establishes the **mathematical feasibility of expanding context to 4K, 8K, 16K, and 32K** on the ESWIN EIC7700X dual-die hardware, analyzes KV-cache memory footprints, and outlines the **Chunked Prefill** and **Output Decoupling** mechanisms required.

---

## 2. KV Cache Memory Mathematical Modeling

In autoregressive Transformer models, the Key ($K$) and Value ($V$) activation states for all past tokens must be preserved in memory so subsequent tokens can attend to them without recomputing past layers.

The exact memory required for the KV Cache is given by:

$$M_{KV} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times L_{\text{ctx}} \times S_{\text{dtype}}$$

Where:
- Factor of $2$: One tensor for Keys, one tensor for Values.
- $N_{\text{layers}}$: Total transformer layers in the model.
- $N_{\text{kv\_heads}}$: Number of Key/Value heads. In Grouped Query Attention (GQA), $N_{\text{kv\_heads}} \ll N_{\text{q\_heads}}$, dramatically reducing KV footprint compared to Multi-Head Attention (MHA).
- $d_{\text{head}}$: Head dimension (typically $128$ for Llama/Qwen, $256$ for Gemma).
- $L_{\text{ctx}}$: Context length in tokens ($L_{\text{prompt}} + L_{\text{generated}}$).
- $S_{\text{dtype}}$: Bytes per element ($2$ for FP16, $1$ for INT8 quantized KV cache).

---

## 3. KV Cache Memory Footprint Across Model Scales

The table below calculates the exact memory required for FP16 KV Caches across model families on the EIC7700X hardware. All values represent **total system KV cache**, which is split symmetrically ($50\% / 50\%$) across Die 0 and Die 1:

| Model Scale | Layer Config | Per-Token Cost | 1,024 Tokens | 4,096 Tokens (4K) | 8,192 Tokens (8K) | 16,384 Tokens (16K) | 32,768 Tokens (32K) | % of 22 GiB MMZ at 16K |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.5B Class**<br>(Qwen2.5-Coder-1.5B) | 28 layers, 2 KV heads, dim 128 | **28.67 KiB** | 28.7 MiB | 114.7 MiB | 229.4 MiB | **458.8 MiB** | 917.5 MiB | **2.0%** |
| **7B Class**<br>(Qwen2.5-7B, DeepSeek-7B) | 28 layers, 4 KV heads, dim 128 | **57.34 KiB** | 57.3 MiB | 229.4 MiB | 458.8 MiB | **917.5 MiB** | 1.83 GiB | **4.1%** |
| **8B Class**<br>(Llama-3.1-8B, Stheno-8B) | 32 layers, 8 KV heads, dim 128 | **131.07 KiB**| 131.1 MiB | 524.3 MiB | 1.05 GiB | **2.10 GiB** | 4.19 GiB | **9.5%** |
| **12B Class**<br>(Mistral-Nemo-12B) | 40 layers, 8 KV heads, dim 128 | **163.84 KiB**| 163.8 MiB | 655.4 MiB | 1.31 GiB | **2.62 GiB** | 5.24 GiB | **11.9%** |
| **12B Hybrid**<br>(Gemma-4-12B) | 40 sliding (1k) + 8 global | Hybrid | 352.0 MiB | 384.0 MiB | 448.0 MiB | **576.0 MiB** | 832.0 MiB | **2.6%** |

### Key Architectural Takeaways:
1. **The 22 GiB MMZ VRAM is more than sufficient**:
   - For a 7B model (weights = ~7.6 GiB), adding a **16,384-token KV cache** requires only **~0.92 GiB total** (~0.46 GiB per die).
   - Total VRAM consumed = $7.60 + 0.92 = 8.52 \text{ GiB}$ out of 22 GiB (**only 38.7% of total MMZ capacity**).
2. **For 1.5B Models, 32K or 64K is easily achievable**:
   - Qwen2.5-Coder-1.5B weights take only 1.55 GiB. Even with a massive 32,768 context, total memory is under 2.5 GiB!

---

## 4. Output Size ($L_{out}$) Decoupling from Context Size ($L_{ctx}$)

A common misconception is that increasing the maximum output tokens (e.g. from 1024 to 4096) requires pre-allocating large memory buffers.

In reality:
- Tokens are generated **one at a time**.
- The generation loop simply appends one new $(K, V)$ vector of size $(2 \times N_{\text{kv\_heads}} \times d_{\text{head}} \times 2 \text{ bytes})$ to the KV cache buffer per step.
- The output text buffer is a standard dynamic string / token vector on the host CPU, consuming negligible memory (<64 KiB).

### Solution for Thinking Models (DeepSeek-R1):
We configure separate controls:
```c
struct GenerationConfig {
    uint32_t max_context_len = 16384; // Total sequence budget (prompt + output)
    uint32_t max_new_tokens  = 4096;  // Generation output ceiling
    uint32_t max_think_tokens = 1024; // Optional limit on <think> tokens
};
```
If DeepSeek spends 600 tokens in `<think> ... </think>`, the model still has **3,496 tokens remaining** to complete its answer without premature cutoff.

---

## 5. Prefill Scaling & Chunked Prefill Architecture

While autoregressive decoding is $O(L_{\text{ctx}})$ per token step, the **prefill phase** (processing the user prompt) has an attention complexity of $O(L_{\text{prompt}}^2)$.

### The 16K Prefill Challenge:
If a user pastes a 16,000-token prompt:
- A naive single forward pass attempts to materialize an attention matrix of $16,384 \times 16,384 \times 2 \text{ bytes} \approx 536 \text{ MiB}$ per head!
- This will violently exceed the EIC7700X NPU's 4 MiB on-chip SPRAM, causing massive LPDDR5 memory thrashing or driver out-of-memory errors.

### The Solution: Chunked Prefill (Tiling)
Instead of feeding all 16,384 tokens in a single giant tensor, the prompt is divided into chunks of $C_{\text{chunk}} = 512$ or $1024$ tokens:

```
[ Long User Prompt: 16,384 Tokens (e.g. Entire Code Repository File) ]
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼ (Chunk 0: 0..1023)    ▼ (Chunk 1: 1024..2047) ▼ (Chunk 15: 15360..16383)
[ Forward Pass 0 ]      [ Forward Pass 1 ]      [ Forward Pass 15 ]
  • Computes KV 0..1023   • Attends to KV 0..1023 • Attends to all prior KV
  • Appends to KV Cache   • Computes KV 1024..2047• Computes final KV
  • Activation: <32 MiB   • Activation: <32 MiB   • Ready for Token Gen!
```

**Benefits**:
1. Peak intermediate activation memory is strictly capped at $<32 \text{ MiB}$ throughout the entire prefill phase.
2. The user interface receives real-time progress updates (`Ingesting prompt: [====>     ] 45% (7168/16384 tokens)`).
3. Eliminates NPU hardware driver buffer overflow errors.

---

## 6. RoPE Frequency Scaling for Long Context Coherence

Models must maintain positional coherence when operating at 4K, 16K, or 32K context lengths.

### Native Long-Context Models:
Modern models are trained with high base frequencies ($\theta$):
- **Llama-3.1-8B**: Native **128,000 tokens** ($\theta = 500,000.0$).
- **Qwen2.5-Coder / 7B**: Native **128,000 tokens** ($\theta = 1,000,000.0$).
- **Mistral-Nemo-12B**: Native **128,000 tokens** ($\theta = 1,000,000.0$).

Because their base frequency is configured for 128k, running them at 4K, 8K, or 16K on the EIC7700X requires **no interpolation or fine-tuning**—they operate with full mathematical accuracy out of the box!

For models originally trained on 2K or 4K (such as older Qwen1.5 or Llama 2), our runtime integrates **YaRN (Yet another RoPE extensioN)** scaling, multiplying position indices by a frequency scale factor $s = L_{\text{target}} / L_{\text{train}}$ to extend context without loss of perplexity.
