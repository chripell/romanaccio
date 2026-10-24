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

---

# 6. Technical Review & Revised Plan (2026-10-24)

Review of sections 1–5 against measured silicon data (`findings/NUMA_MEMORY_AND_HARDWARE_BENCHMARK.md`,
`findings/SYSTEM_CAPABILITIES_AND_MODELS.md`) and the one calibrated on-board LLM data point we
have (Qwen2.5-Coder-1.5B). **Three numbers in the plan are wrong, one architectural
contradiction decides the entire design, and one hardware constraint is not addressed at all.**
Read this section before starting Phase 1.

## 6.1 The blocking contradiction: layer-parallel vs expert-parallel

Section 2 says *"Layers 0–13 mapped to Die 0, Layers 14–26 mapped to Die 1"* (layer/pipeline
parallelism). Section 3.1 says *"Experts 0–31 reside in `mmz_nid_0_part_0`, Experts 32–63 in
`mmz_nid_1_part_0`"* (expert parallelism). **These are mutually exclusive**, and the choice
decides whether the 28 GB/s figure is reachable at all.

Measured facts: one die saturates at **~14.0 GB/s** local read; both dies reading their own
local memory concurrently reach **47.93 GB/s** triad (the plan's "~28 GB/s practical
sustained read" is the conservative reading of the same data).

- **Layer-pipeline** (the section 2 choice): a single token flows through Die 0's layers,
  *then* Die 1's. At batch 1 only one die streams weights at any instant, so effective
  bandwidth is **~14 GB/s**, not 28. Die 1 sits idle for the first half of every token.
- **Expert-parallel** (the section 3.1 choice): every layer's experts are split by index
  across both dies, so for each layer both dies stream simultaneously → **~28 GB/s** is
  genuinely available.

**Decision: use expert parallelism.** It is the only layout that delivers the bandwidth
aggregation the whole feasibility argument rests on. Layer splitting should be dropped for
this model (it is the right choice for dense models like Qwen2.5-Coder, where there is no
other axis to split on).

## 6.2 Throughput: the 11.7 tok/s figure is ~2× optimistic

$\text{tok/s} = \text{bandwidth} / \text{bytes streamed per token}$. The plan uses 28 GB/s at
**100% efficiency**. We can calibrate efficiency instead of assuming it:

> Qwen2.5-Coder-1.5B, 28 layers split 14/14, measured **3.88–5.49 tok/s** on `roma`
> (2026-10-21 run: 3.88 tok/s at 619 generated tokens). Streaming ~1.5 GB/token ⇒
> **5.8–8.2 GB/s effective** against a 14 GB/s single-die ceiling ⇒ **41–59% efficiency**.

The losses are per-submit driver/IPC overhead, CPU-side RoPE/softmax/KV work, D2D transfers
and cache maintenance — all of which apply here too. Recomputing the per-token traffic
(§6.3) at 2.18 GiB and applying the measured efficiency band:

| Layout | Bandwidth | ×41–59% efficiency | **Projected tok/s** |
| :--- | :--- | :--- | :--- |
| Expert-parallel (recommended) | 28 GB/s | 11.5–16.5 GB/s | **5.3 – 7.6** |
| Layer-pipeline (as written) | 14 GB/s | 5.7–8.3 GB/s | **2.6 – 3.8** |
| Plan's claim (§1.1) | 28 GB/s | 100% | ~~11.7~~ |

**Action: change the Phase 4 gate from `≥ 10.0 tok/s` to `≥ 5.0 tok/s`.** A 10 tok/s gate
would fail a correctly-built engine and invite chasing a number the hardware cannot deliver.
Expert-parallel still beats the dense-7B case and beats Qwen2.5-Coder-1.5B's measured
3.88–5.49 tok/s while carrying 16B of knowledge — the thesis survives, the multiplier doesn't.

## 6.3 Corrected memory arithmetic

Recomputed from the architecture (INT8 weights, fp16 KV):

| Item | Plan says | Correct value | Note |
| :--- | :--- | :--- | :--- |
| MLA attention blocks (all 27 layers) | 1.80 GiB | **0.23 GiB** | q_a 2048×576 + q_b 512×2048 + kv_a 2048×576 + kv_b 512×3072 + o 2048×2048 = 9.17 M params/layer × 27 = 248 M ⇒ 0.23 GiB INT8. **Plan is 7.8× high.** |
| Routed experts (64 × 26 layers) | 4.10 GiB per die | **6.7 GiB per die** | 8.65 M params/expert = 8.05 MiB INT8; 1664 experts = 13.4 GiB total, 6.7 GiB/die. **Plan is ~1.6× low.** |
| Shared experts | 1.50 GiB | **0.42 GiB** | 2 × 26 × 8.05 MiB. |
| 16K MLA KV cache | ~190 MiB | **486 MiB** (243 MiB/die) | $(512 + 64) \times 2\,\text{B} \times 27 \times 16384$. **Plan is 2.6× low.** |
| Bytes streamed per token | 2.40 GB | **2.18 GiB** | 208 active experts (1.76 GiB) + attention (0.23) + LM head (0.195). Plan is ~10% conservative — fine. |

Corrected per-die budget with **expert parallelism** (all 27 layers on both dies):

```
Die 0 / Die 1 (11 GiB default MMZ each; 13.5 GiB if the U-Boot reconfig is applied)
  routed experts (32 of 64, all 26 MoE layers)   6.70 GiB
  shared experts (1 of 2, all layers)            0.21 GiB
  MLA attention (replicated or split)            0.12–0.23 GiB
  LM head slice (51,200 of 102,400)              0.10 GiB
  16K MLA KV cache                               0.24 GiB
  activation ping-pong + scratch                 0.05 GiB
  ----------------------------------------------
  TOTAL                                          7.42–7.53 GiB  / 11 GiB
```

**This fits in the default 22 GiB MMZ (11 GiB/die) with ~3.5 GiB headroom per die.** The
27 GiB U-Boot reconfiguration in Phase 4 is therefore **not required** — drop it from the
critical path, since a DTB/`fdt mmz` change is a board-recovery risk (see
`findings/VRAM_RECONFIGURATION.md`). Revisit only if attention is replicated *and* 32K
context is wanted.

Note the embedding table (102,400 × 2048) is looked up one row at a time, so it costs
**4 KiB of traffic per token**, not 0.20 GiB — it belongs in host RAM via the Milestone 8
`EmbeddingTable` mmap, exactly as §4 already draws it.

## 6.4 The unaddressed problem: compiled loadables have *static* weights

This is the hardest issue in the project and §4/§5 do not mention it.

An `EsAAC`-compiled `.model` bakes its weights into blobs. MoE needs **per-token weight
selection**: which 6 of 64 experts run is decided by the router at run time. Three options:

| Approach | Files | Traffic/token | Verdict |
| :--- | :--- | :--- | :--- |
| **(a) One loadable per expert** | 1664 routed + 52 shared | optimal (only selected experts read) | Works, but see the load-time problem below |
| **(b) One loadable per layer, all 64 experts, weights as a dynamic input tensor bound at a runtime-chosen `DevBuf.offset`** | 27 | optimal | **Best if the compiler supports dynamic weight inputs** |
| (c) Compute all 64 experts | 27 | 64× worse (13.4 GiB/token) | Destroys the entire MoE advantage — never |

(b) is attractive because `DevBuf` already carries `{mem_fd, offset, size}`: one big weight
blob per die plus a single loadable per layer, with expert selection reduced to *changing an
input offset*. That eliminates 1700+ files and 1700+ `ES_NPU_LoadModelFromFile` calls.

**Why (a) is risky:** 1716 loadables at ~9 MiB each must all be loaded and kept resident.
If each load costs 50–200 ms, startup is **1.5–6 minutes**, and the driver's limit on
concurrently loaded `modelId`s is unknown. This must be measured, not assumed.

**Phase 0 (new, do this first — it is cheap and decides everything):**
1. Export a single MoE expert FFN (2048→1408 SwiGLU→2048) to ONNX with the **weights as
   graph inputs rather than initializers**, compile with `EsAAC`, then inspect with
   `eswin-sim -m expert.model --info -v`. If `weight` appears in *Input Tensors*, approach
   (b) is viable and the file-count problem disappears.
2. If it does not, compile the same expert with static weights, then time
   `ES_NPU_LoadModelFromFile` on 32 of them on `roma` to extrapolate the 1716-load startup
   cost, and check whether 32 concurrent `modelId`s coexist.
3. Separately: submit one expert 100× and record per-submit wall time. With ~318 submits per
   token (208 experts + ~108 attention ops + LM head), **submit overhead above ~0.3 ms would
   itself cap throughput below 10 tok/s**, independent of bandwidth. Measured single-op
   graphs so far: SwiGLU MLP 0.28 ms, QKV 0.19 ms — encouraging but not conclusive.

## 6.5 Architecture details the plan omits

1. **Layer 0 is dense, not MoE.** DeepSeek-V2-Lite sets `first_k_dense_replace = 1`, so layer 0
   uses a plain MLP (`intermediate_size = 768`) with no router and no experts. §2's
   "Shared Experts 2 per layer / Routed Experts 64 per layer" and §4's loop over L=0..26 are
   wrong for L=0. There are **26 MoE layers**, not 27 — every count in §6.3 uses 26.
2. **Router activation must be confirmed, not assumed.** §4 says `Softmax`. V2-Lite uses
   `topk_method: "greedy"` with `norm_topk_prob: false`, and the V2 family's gate has
   historically been **sigmoid** in some releases and softmax in others (V3 is sigmoid).
   Read `gate`'s activation and `norm_topk_prob` from the actual `config.json` before
   implementing `moe_router`: with `norm_topk_prob=false` the top-6 weights are used
   **unnormalised**, which changes the accumulation in §4 step 4 and will silently scale
   outputs wrong if assumed otherwise.
3. **MLA RoPE is partial.** Only the 64-dim `qk_rope_head_dim` slice of q and of the latent
   k gets rotated; the 64-dim nope slice does not. The existing DSP RoPE kernel
   (`libes_qk_rotary_position_embedding_f16f16.so`) rotates a contiguous head — a variant or
   a pre/post permutation is needed. Budget for this in the compiler work.
4. **`--kv-cache-len` is a compile-time constant.** A 16K-context build is a *different*
   compile from a 1K build, and attention traffic grows with context: at 16K the MLA cache
   adds 486 MiB/token on top of 2.18 GiB (+22% traffic, ≈ −18% tok/s). Decide the target
   context **before** compiling; 4K is the sensible first target (§6.3 budget scales linearly).
5. **The runtime pieces this needs already exist** as of Milestones 7–11 (2026-10-24):
   `PipelineExecutor` for chained per-die execution with ping-pong MMZ activations;
   `EmbeddingTable` for the mmap'd 102,400×2048 table; `HardwareKvCache` for MMZ-resident
   MLA latent caches (note: it needs `token_n2` from the package — do **not** let it guess);
   `ModelManifest` with `blocks`/`blocks_for_die()` and vendor `config.json` support;
   `es_session_perplexity()` plus `tools/e2e_regression.sh` for the Phase 4 gate.
   None are wired into `TransformerSession` yet — that switch-over is the open Milestone 10
   item and should be done (and board-baselined) before this model attempts to use them.

## 6.6 Revised roadmap

**Phase 0 — Decide the execution model (before any weights are downloaded).**
- [ ] Dynamic-weight-input experiment (§6.4.1) → picks approach (a) or (b).
- [ ] Measure `ES_NPU_LoadModelFromFile` time and concurrent-`modelId` limits on `roma`.
- [ ] Measure per-submit overhead for one expert FFN (100 submits).
- [ ] Read the real `config.json`: confirm `first_k_dense_replace`, `n_routed_experts`,
      `n_shared_experts`, `num_experts_per_tok`, `norm_topk_prob`, gate activation,
      `kv_lora_rank`, `qk_rope_head_dim`, `v_head_dim`, `vocab_size`.
- [ ] **Gate:** if approach (a) startup exceeds ~2 min *or* per-submit overhead implies
      < 4 tok/s, stop and re-scope (e.g. fewer experts resident, or a smaller MoE).

**Phase 1 — Ingestion.** Unchanged, except: 26 MoE layers + 1 dense layer; weights are
~14.6 GiB INT8 / ~31 GiB fp16, which are **not in `deep/models/`** — the download is a
prerequisite and needs a decision (see §6.7).

**Phase 2 — Host router (`src/llm/moe_router.{hpp,cpp}`).** Fully host-testable with no
weights and no board: gate GEMM (2048×64), top-6 selection, `norm_topk_prob` handling, and
grouping selected indices by die. Add golden tests against a PyTorch reference router.
Target < 60 µs/layer — at 26 layers that is 1.6 ms/token, ~1% of budget, so this is not
where the risk is.

**Phase 3 — Simulator verification.** Add to `eswin_simulator`: MLA latent projection,
partial RoPE, and a single-expert SwiGLU. Differential cosine ≥ 0.9999 vs PyTorch per block,
as written.

**Phase 4 — Hardware.** Drop the 27 GiB U-Boot reconfig (§6.3 shows it is unnecessary).
Change the throughput gate to **≥ 5.0 tok/s** (§6.2). Use
`tools/e2e_regression.sh --host roma --record` before and after every change to the decode
path. Measure first-token latency and KV stability at 4K, not 16K, first.

## 6.7 Open decisions needing the user

1. **Weights are not local.** `deep/models/` holds only `qwen2_5_coder_1.5b` and `smollm`.
   DeepSeek-V2-Lite is ~31 GiB fp16 / ~14.6 GiB INT8. Download, or work from a smaller MoE
   first?
2. **Board SSH is currently failing** (`Permission denied (publickey,password)` as of
   2026-10-24, though it worked earlier the same day). Phase 0 is entirely board-dependent.
3. **Target context length** (4K recommended first) — it fixes the compile and the memory
   budget, and cannot be changed afterwards without recompiling.

## 6.8 Knock-on work in the other components

- **`eswin_compiler`**: dynamic-weight-input support (§6.4b) is a new capability, not a
  parameter tweak — it means emitting a loadable whose weight tensors are graph inputs with
  runtime-bound addresses. Also: partial-dimension RoPE, and an expert-FFN archetype.
- **`eswin_simulator`**: needs the same MLA/expert ops to be useful for Phase 3; its DSP
  fallbacks already cover RMSNorm/Softmax/RoPE/TopK, and `topk.py` is directly reusable for
  router golden tests.
- **`eswin_runtime`**: `moe_router` (new), plus the deferred Milestone 10 switch-over to
  `PipelineExecutor`/`HardwareKvCache`/`EmbeddingTable`. Expert-parallel dispatch also wants
  concurrent submission to both dies for the *same* layer, which is a different synchronisation
  pattern from the pipeline handoff `PipelineExecutor` implements today.

---

## 6.9 Phase 0 result: config.json verified (2026-10-24)

`deepseek-ai/DeepSeek-V2-Lite/config.json` fetched and read. **This closes §6.5 items 1, 2
and 3, and corrects two numbers in §6.3 that I had wrong.**

| Field | Value | Consequence |
| :--- | :--- | :--- |
| `num_hidden_layers` / `first_k_dense_replace` / `moe_layer_freq` | 27 / 1 / 1 | **26 MoE layers** + 1 dense layer, confirmed |
| `intermediate_size` (dense layer 0) | **10944** | §6.5 said 768 — wrong. Layer 0 MLP is 67.3 M params (62.6 MiB INT8), not 4.7 M |
| `hidden_size` | 2048 | as assumed |
| `moe_intermediate_size` | 1408 | as assumed; expert = 3 × 2048 × 1408 = 8.65 M params = 8.05 MiB INT8 |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 64 / 2 / 6 | 8 active FFNs per MoE layer, as assumed |
| **`scoring_func`** | **`softmax`** | §4's Softmax was **right**; §6.5 item 2 resolved |
| **`norm_topk_prob`** | **`false`** | top-6 weights are used **unnormalised** (sum < 1). Silently rescales the residual if assumed otherwise |
| `topk_method` / `n_group` / `topk_group` | `greedy` / 1 / 1 | group-limited top-k is **degenerate** — plain top-6 over 64 |
| `routed_scaling_factor` | 1.0 | no-op, but implement it |
| **`q_lora_rank`** | **`None`** | **No query-side low-rank compression** in V2-Lite. One `q_proj` 2048→3072, not q_a/q_b. Simpler graph than §6.3 assumed |
| **`qk_nope_head_dim`** | **128** (not 64) | q head = 128 + 64 = **192**; §6.5 item 3 was wrong |
| `qk_rope_head_dim` / `v_head_dim` | 64 / 128 | partial RoPE rotates **64 of 192** q dims; kv_b_proj = 512 → 16×(128+128) = 4096 |
| `num_key_value_heads` | 16 | MLA expands the latent to all 16 heads |
| `kv_lora_rank` | 512 | latent KV = 512 + 64 = 576/token/layer ⇒ §6.3's 486 MiB at 16K stands |
| **`torch_dtype`** | **`bfloat16`** | HF weights are bf16, not fp16. Milestone 8 `EmbeddingTable` already handles `Bf16`; the export path must convert deliberately (bf16→fp16 loses range) |
| **`rope_scaling`** | `type: yarn, factor: 40, original_max_position_embeddings: 4096, mscale: 0.707` | At a **4K target the context equals the YaRN original max**, so interpolation should not engage — but `mscale` attention scaling may still apply. Must be checked against the HF reference, not assumed away |
| `rope_theta` | 10000 | not 1e6 (that is Qwen2.5-Coder) |
| `vocab_size` / `tie_word_embeddings` | 102400 / **false** | LM head is a separate 102400×2048 matrix; 51200/die = 200×256 ✓ hardware-aligned |
| `hidden_act` / `attention_bias` | `silu` / `false` | SwiGLU experts, no attention biases (unlike Qwen2) |
| `max_position_embeddings` | 163840 | model supports far more than we can compile for |

**Corrected attention size:** with `q_lora_rank=None` and `qk_nope_head_dim=128`, per layer
= q_proj 6.29 M + kv_a 1.18 M + kv_b 2.10 M + o_proj 4.19 M = **13.76 M**, so 27 layers =
371.5 M params = **0.35 GiB INT8** — not the 0.23 GiB in §6.3 (which assumed a q_a/q_b
split and a 64-dim nope head). The plan's original 1.80 GiB is therefore ~5× high, not 7.8×.

**Corrected per-token traffic:** attention 0.35 + dense layer 0 MLP 0.06 + 208 active
experts 1.68 + LM head 0.20 = **2.28 GiB/token**. At 28 GB/s and the calibrated 41–59%
efficiency band this gives **5.0–7.2 tok/s** (§6.2 said 5.3–7.6; the conclusion and the
recommended ≥5 tok/s gate are unchanged).

Cross-check: 371.5 M + 67.3 M + 26×66×8.65 M + 209.7 M + 209.7 M = **15.69 B params**, which
matches the published 15.71 B. The arithmetic is now closed.

**Model download status:** `deepseek-ai/DeepSeek-V2-Lite` is ungated and downloading to
`deep/models/DeepSeek-V2-Lite/`. The other five planned repos
(`google/gemma-3-12b-it`, `Rin247/gemma-4-12B-it-Uncensored-Aquarion-INT8`,
`litert-community/TranslateGemma-4B-IT`, `Sao10K/L3.1-8B-Stheno-v3.3`,
`huihui-ai/Qwen2.5-7B-Instruct-abliterated`) all return **401 Unauthorized** because the
host `hf` CLI is **not logged in** — this is an auth problem, not evidence the repos are
missing. They need `hf auth login` with a token that has accepted each model's licence
(gemma repos are licence-gated even for valid tokens).

---

## 6.10 Phase 2 COMPLETE: MoE router implemented and measured on silicon (2026-10-24)

`src/eswin_runtime/src/llm/moe_router.{hpp,cpp}` + `tests/test_moe_router.cpp`, wired into
both build systems. All of §5 Phase 2's items are done except the RVV-specific kernel.

**Measured on `roma` (RISC-V, 1.4 GHz), 64×2048 gate, top-6: 53.4 µs average per layer**
— inside the <60 µs target. Over 26 MoE layers that is 1.39 ms/token, **0.87% of a
6.25 tok/s budget**, so routing is confirmed *not* to be the bottleneck, as §6.6 predicted.
For comparison the x86 host measures 47.5 µs, i.e. the board is within ~12% of host speed
for this kernel; both runs show a multi-ms worst-case outlier (3.4 ms board / 10.6 ms host)
from scheduler preemption, which matters for a latency-sensitive decode loop and is the
argument for pinning the routing thread to a specific core.

What it implements, driven by the verified config rather than assumption:

- Gate GEMM `[64][2048]` with OpenMP over experts; all scratch pre-sized in `init()` so
  `route()` performs **no allocation** (it runs 26× per token).
- `Softmax` (V2-Lite's `scoring_func`) and `Sigmoid` (V3-family), numerically stable.
- `norm_topk_prob=false` honoured: the top-6 weights stay as raw softmax probabilities and
  sum to **less than 1** (measured 0.144 on a random gate). Renormalising them would
  silently rescale every expert contribution — the test asserts both modes and that they
  select the same experts and differ only by a uniform rescale.
- `Greedy` top-k and `GroupLimitedGreedy` (degenerate at `n_group=1` for V2-Lite, but
  implemented and tested for the V3 family). `init()` rejects a `topk_group × group_size`
  that cannot supply `top_k`, which would otherwise select `-inf`-scored experts.
- Deterministic tie-breaking toward the lower expert index — `std::sort` is not stable, so
  the index is part of the comparator rather than left to the sort.
- Expert-parallel dispatch grouping (`die_experts[2]`) and the addressing helpers that make
  the §6.4(b) execution model possible: `expert_blob_offset(layer, expert, bytes)` returns
  the `DevBuf.offset` for one expert inside **its own die's** blob, so a single per-layer
  loadable can run any of the 64 experts by changing an offset instead of a model. Offsets
  are per-die-relative by design (die 0 and die 1 both start at 0) since each die has a
  separate `DmaBuffer`/`mem_fd`.

Verification: 200 random tokens differentially matched against an **independent
double-precision reference** written inline in the test (sharing no code with the router) on
expert set, ordering, weights and argmax; plus sigmoid/scaling, gate bias, group masking,
per-die offset density and collisions, determinism across repeat calls, and 11 rejected
invalid configurations. Clean `riscv64-linux-gnu-` cross-compile.

Not done: an RVV-vectorised gate kernel (the scalar+OpenMP version already meets target, so
this is optimisation, not a gap), and the `PyTorch` golden comparison §6.6 asked for — the
inline double-precision reference covers selection semantics but not HF's exact
`DeepseekV2MoE` gating code path, which is worth adding once the weights finish downloading.

**Correction to §6.9's download note:** the host `hf` CLI *is* now authenticated
(`chripell`), and the five other repos still fail — for two different reasons.
`google/gemma-3-12b-it` and `litert-community/TranslateGemma-4B-IT` return
**"Access denied … requires approval"** (licence-gated; accept the licence on the HF site).
`Sao10K/L3.1-8B-Stheno-v3.3`, `Rin247/gemma-4-12B-it-Uncensored-Aquarion-INT8` and
`huihui-ai/Qwen2.5-7B-Instruct-abliterated` return **404/not found**, so those repo IDs in
the `rpg_uncensored` and `gemma-4-12B` plans need to be re-checked — note `gemma-4` is not a
model family that exists, and the plan's own parameter table matches Gemma **3**.
`deepseek-ai/DeepSeek-V2-Lite` is ungated and downloading normally.

---

## 6.11 Phase 0 result: EsAAC rejects dynamic weight inputs (2026-10-24) — approach (b) is closed via the vendor compiler

§6.4's decisive experiment was run. **Result: approach (b) is not reachable through
`EsAAC`, so the 27-loadable design needs the open compiler, not the vendor one.**

Probe (`deep/artifacts/scratch/probe_dynamic_weight.py`, opset 13, static shapes, per the
EsAAC constraints): one `MatMul` `[1,2048] × [2048,1408]`, emitted twice — weight as an
initializer (11.5 MB ONNX) and weight as a second graph input (131 bytes ONNX, no embedded
weights).

| Probe | EsAAC result |
| :--- | :--- |
| `MatMul`, W as **initializer** | **compiles in 1.3 s** ✅ |
| `MatMul`, W as **graph input** | **SIGSEGV, exit 139, core dumped**, no `.model` |
| same + `--fix-shape` | SIGSEGV |
| same + `--batchsize=1` | SIGSEGV |
| same + both flags | SIGSEGV |
| `Add` with **two dynamic inputs** (control) | SIGSEGV |

The control matters: the crash is not specific to MatMul weights. `EsAAC`'s frontend
segfaults on a graph with a **second dynamic input** at all, and no flag combination tested
changes that. The identical graph with the weight folded in as an initializer compiles
cleanly, so the shapes and the op are fine — the second runtime input is the trigger.

**Why this is not proof that dynamic weights are impossible.** Vendor packages demonstrably
contain multi-input loadables: the Qwen2-0.5B block inspected in §6.5 has **four** input
tensors (hidden, token step, V cache, K cache). The difference is that those extra inputs
are the KV-cache pair, which is produced by the LLM-specific flags (`--kv-cache-len`,
`--kv-layers-desc`). The evidence therefore says EsAAC's multi-input support is tied to its
LLM KV-cache path, not that the loadable format cannot express a runtime-bound weight.

**Consequences for the plan:**

1. **Approach (b) moves to `eswin_compiler`.** Our open compiler emits loadables directly
   (`schema/loadable.fbs`, `backend/flatbuffer_emitter.py`) and is not bound by EsAAC's
   frontend. Emitting the expert weight as a `tensor_desc_list` entry with a `reloc_list`
   binding — rather than as a blob — is the mechanism, and it is exactly what
   `MoERouter::expert_blob_offset()` was written to feed (`DevBuf.offset`). This becomes the
   highest-value compiler work item, and it needs its own milestone in
   `src/eswin_compiler/PLAN.md`.
2. **Approach (a) is the fallback and is now on the critical path.** If the open compiler
   cannot bind a weight at submit time, we are back to 1716 per-expert loadables, which
   makes the §6.4 load-time and concurrent-`modelId` measurements mandatory rather than
   optional. Those are still unrun.
3. **Verify before building on this.** The probe used a bare `MatMul`. Before concluding
   anything about the *runtime* (as opposed to the compiler frontend), inspect a
   vendor-compiled multi-input loadable and confirm whether its non-activation inputs are
   `tensor_desc_list` entries with relocations or blobs. An attempt to retrieve the
   compiled static probe failed (0-byte transfer — the output path was wrong, EsAAC's
   naming was not confirmed), so **this inspection is outstanding**.

**Not run:** the load-time / concurrent-`modelId` / per-submit-overhead measurements
(§6.4 items 2 and 3). Per-submit overhead remains the other open risk: ~318 submits per
token means anything above ~0.3 ms caps throughput below 10 tok/s regardless of bandwidth.
