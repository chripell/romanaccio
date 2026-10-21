# DeepSeek-V2-Lite MoE (16B / 2.4B Active) Dual-Die Acceleration Plan

## 1. Executive Summary & Architectural Motivation

### 1.1 The Fundamental Breakthrough: Overcoming the Memory Wall
On the **ESWIN EIC7702** dual-die SoC (board `roma` FML13V03), memory bandwidth measurements demonstrate an aggregate sustainable read bandwidth of **~28 GB/s** across both LPDDR5 controllers.

In dense models (such as Qwen2.5-7B or DeepSeek-7B), autoregressive decoding requires streaming **every parameter from DRAM for every generated token**:
$$\text{Speed}_{\text{dense-7B}} \le \frac{28.0\text{ GB/s}}{7.6\text{ GB/token}} \approx \mathbf{3.7\text{ tokens/second}}$$

**DeepSeek-V2-Lite** completely changes this equation through its **Mixture of Experts (MoE)** and **Multi-head Latent Attention (MLA)** architecture:
- **Total Parameters**: 15.7 Billion.
- **Active Parameters Per Token**: Only **~2.40 Billion** (2 shared experts + 6 routed experts per layer).
- **DRAM Streamed Per Token**: Only **~2.40 GB** (in INT8).
- **Theoretical Decoding Throughput**:
  $$\text{Speed}_{\text{MoE-V2-Lite}} \le \frac{28.0\text{ GB/s}}{2.4\text{ GB/token}} \approx \mathbf{11.7\text{ tokens/second}}$$

**Result**: DeepSeek-V2-Lite runs **$3.2\times$ faster than a dense 7B model**, while delivering the superior knowledge capacity and reasoning of a 16B parameter model!

---

## 2. Model Architecture Specifications

| Property | DeepSeek-V2-Lite Value | Architectural Implication on EIC7702 |
| :--- | :--- | :--- |
| **Total Parameters** | 15.71 Billion | 15.7 GiB (INT8) — Fits completely in 27 GiB MMZ VRAM |
| **Active Parameters** | 2.40 Billion per token | Streams only 2.4 GB/token across dual-die memory channels |
| **Total Layers** | 27 Transformer Layers | Layers 0–13 mapped to Die 0, Layers 14–26 mapped to Die 1 |
| **Hidden Dimension ($d$)** | 2,048 | Modest activation tensor size (< 16 MiB per step) |
| **Attention Mechanism** | Multi-head Latent Attention (MLA) | Compresses KV cache into low-rank latent vector ($d_c = 512$) |
| **KV Cache Footprint** | **~190 MiB** for 16,384 tokens | Massive $5\times$ memory reduction compared to standard MHA |
| **Shared Experts** | 2 per layer (always active) | Evaluated concurrently on both dies |
| **Routed Experts** | 64 per layer | Distributed across Die 0 (0..31) and Die 1 (32..63) |
| **Active Routed Experts** | Top-$k = 6$ per layer | Only 6 out of 64 expert sub-models executed per token |
| **Intermediate Size** | 1,408 per expert | Compact expert feed-forward networks (FFN) |

---

## 3. Dual-Die NUMA Memory & Expert Partitioning

With the board configured for **27.0 GiB total NPU MMZ** (13.5 GiB on Die 0, 13.5 GiB on Die 1, leaving 5.0 GiB for Linux):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        DEEPSEEK-V2-LITE DUAL-DIE MEMORY TOPOLOGY (27 GiB MMZ)                   │
│                                                                                                 │
│  [ NUMA Node 0 / Die 0 ]                                             [ NUMA Node 1 / Die 1 ]    │
│  +----------------------------------------------+                     +-----------------------+ │
│  | MMZ Carveout Pool 0: 13.5 GiB Total          |                     | MMZ Carveout Pool 1   | │
│  |                                              |                     | 13.5 GiB Total        | │
│  | • Embedding Table Slice: 0.20 GiB            |  D2D Interconnect   | • LM Head Slice:      | │
│  | • MLA Attention Blocks (L0-L26): 1.80 GiB    | <=================> |   0.50 GiB            | │
│  | • Shared Experts (2 per layer): 1.50 GiB     |  (~22 GB/s Cap)     | • Routed Experts      | │
│  | • Routed Experts 0..31 (L0-L26): 4.10 GiB    |                     |   32..63: 4.10 GiB    | │
│  | • 16K MLA KV Cache: 0.10 GiB                 |                     | • 16K MLA KV Cache:   | │
│  | -------------------------------------------- |                     |   0.10 GiB            | │
│  | Total Committed: 7.70 GiB / 13.50 GiB        |                     | Total Committed:      | │
│  | Headroom Free:   5.80 GiB (43% Free)         |                     |   4.70 GiB / 13.50GB  | │
│  +----------------------------------------------+                     +-----------------------+ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Partitioning Innovations:
1. **Expert Localization by NUMA Node**:
   - Experts 0–31 reside in `mmz_nid_0_part_0`.
   - Experts 32–63 reside in `mmz_nid_1_part_0`.
   - When the host CPU evaluates the router gate logits, it groups activated expert indices by die. Tasks are dispatched to Worker 0 (Cores 0–3) or Worker 1 (Cores 4–7) with zero remote memory crossing!
2. **Shared Expert Offload**:
   - The 2 shared experts capture general linguistic representations and are evaluated every token.
   - Shared Expert 0 runs on Die 0; Shared Expert 1 runs on Die 1 in parallel.
3. **Multi-head Latent Attention (MLA) Footprint**:
   - Because MLA compresses $K$ and $V$ activations into a low-rank latent representation ($d_c = 512$), a **16,384-token context cache requires only ~190 MiB total** (~95 MiB per die).
   - This eliminates the KV-cache memory bottleneck entirely.

---

## 4. Execution Pipeline & Software Architecture

```
[ Input Token ]
       │
       ▼
[ Host CPU: mmap Embedding Lookup ] (0.20 GiB table)
       │
       ▼
[ Loop Over Layers L = 0 to 26 ]:
       │
       ├─► 1. MLA Attention Block (NPU 0 + NPU 1):
       │      • Project hidden state to compressed latent key/value vector
       │      • Append to circular MLA KV Cache in MMZ
       │      • Compute self-attention output
       │
       ├─► 2. Router Gating (Host CPU Fast Vector Kernel):
       │      • Compute gate logits: $s = \text{Softmax}(W_{\text{gate}} \cdot x)$
       │      • Select top-6 routed experts
       │      • Group selected indices: $E_{\text{die0}} = \{e \in \text{top6} \mid e < 32\}$,
       │                               $E_{\text{die1}} = \{e \in \text{top6} \mid e \ge 32\}$
       │
       ├─► 3. Parallel Expert Execution (Dual-Die Worker Pool):
       │      ┌─────────────────────────────┴─────────────────────────────┐
       │      ▼ (Worker 0 / Die 0)                                        ▼ (Worker 1 / Die 1)
       │      • Execute Shared Expert 0                                   • Execute Shared Expert 1
       │      • Execute $E_{\text{die0}}$ routed experts                  • Execute $E_{\text{die1}}$ routed experts
       │      └─────────────────────────────┬─────────────────────────────┘
       │
       └─► 4. Expert Accumulation & Residual Addition:
              • $x_{L+1} = x_L + \text{MLA}(x_L) + \sum_{i=1}^2 \text{Shared}_i(x_L) + \sum_{j \in \text{top6}} s_j \text{Routed}_j(x_L)$
       │
[ Final Layer Norm ]
       │
       ▼
[ Dual-Die LM Head Reduction ] (Die 0: 0..51199, Die 1: 51200..102399)
       │
       ▼
[ Issue #47 Anti-Looping Sampler ] (Temperature, Top-P, Cyclic Penalty)
       │
       ▼
[ Emit Generated Token & Stream Callback ]
```

---

## 5. Phased Implementation Roadmap

### Phase 1: Architecture Ingestion & Sub-Model Decomposition (Milestone 1–2)
- [ ] Ingest `deepseek-ai/DeepSeek-V2-Lite` weights and config from Hugging Face.
- [ ] Export ONNX subgraphs:
  - `mla_block_layer_N.onnx`: Layer norm + MLA latent projection + attention score.
  - `shared_expert_layer_N.onnx`: 2 shared FFN blocks.
  - `routed_expert_layer_N_expert_E.onnx`: Compact single-expert FFN blocks.
  - `lm_head_d0.onnx` and `lm_head_d1.onnx`: Dual-die logit projection.
- [ ] Quantize all subgraphs to INT8 using symmetric per-channel quantization.
- [ ] Compile `.model` blocks using `EsAAC` / `eswin_compiler`.

### Phase 2: Host Router & Dynamic Expert Dispatcher in C++ (Milestone 3)
- [ ] Implement `src/llm/moe_router.hpp` and `src/llm/moe_router.cpp` in `eswin_runtime`:
  - Fast AVX/RVV dot-product kernel for $W_{\text{gate}} \cdot x$.
  - Top-$k$ selection ($k=6$) with softmax normalization.
  - Zero-copy dispatch of active expert jobs to `WorkerPool` on Die 0 and Die 1.
- [ ] Benchmark CPU routing latency: target $< 60\,\mu\text{s}$ per layer.

### Phase 3: Simulator Differential Verification (Milestone 4)
- [ ] Ingest MLA and expert blocks into `eswin_simulator`.
- [ ] Verify cosine similarity against PyTorch reference outputs for:
  - Compressed latent attention projection ($\ge 0.9999$).
  - SwiGLU expert feed-forward activations ($\ge 0.9999$).
- [ ] Validate end-to-end token forward pass in software simulation.

### Phase 4: Physical Hardware Deployment on Board `roma` (Milestone 5)
- [ ] Reconfigure U-Boot MMZ memory to **27 GiB** (Config A: `0x360000000` per die).
- [ ] Deploy compiled DeepSeek-V2-Lite model package with `manifest.json`.
- [ ] Run verification inference with `eswin-chat`:
  - Assert sustained generation speed $\ge 10.0\text{ tokens/second}$.
  - Validate chain-of-thought `<think>` reasoning without looping.
  - Measure first-token latency and KV-cache stability over 4K context.
