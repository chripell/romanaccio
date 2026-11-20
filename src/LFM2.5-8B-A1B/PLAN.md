# LiquidAI LFM2.5-8B-A1B Sparse MoE Dual-Die Acceleration Plan

> **Target Architecture**: LiquidAI `LFM2.5-8B-A1B` (8.47 Billion Total Parameters, ~1.22 Billion Active Parameters / Token).  
> **Target Silicon**: ESWIN EIC7700X / EIC7702 Dual-Die RISC-V SoC (Board `roma` FML13V03, 64-bit SiFive P550 cores + Dual NPU cores + 27 GiB MMZ VRAM).  
> **Status**: **Phases A–F Complete — Verified on `roma` Silicon at `11.38–12.67 tok/s` (`0.99837211` Logits Cosine, `0.97924192` Min Layer Cosine, `4/4` Exact Greedy Match, `5/5` Bit-Exact Prefix Cache `max_abs=0.000000` with `71.7x` 4K TTFT Speedup & `19.14 ms` NVMe Restore, Live OpenAI Gateway `:8089`, and `8.78 GiB` Free OS RAM)**.  

---

## 1. Executive Summary & Architectural Motivation

### 1.1 The Breakthrough: High-Capacity Reasoning at Sub-2B Decode Latency
On the **ESWIN EIC7702** dual-die SoC, autoregressive decoding throughput is fundamentally bounded by the memory bandwidth wall. The SoC provides an aggregate sustainable DRAM read bandwidth of **~28.0 GB/s** (~14.0 GB/s per NUMA die) across both 64-bit LPDDR5 memory channels.

In dense LLMs (such as Qwen2.5-7B or DeepSeek-7B), every token generated requires streaming all 7–8 GB of model weights from DRAM:
$$\text{Speed}_{\text{dense-7B}} \le \frac{28.0\text{ GB/s}}{7.6\text{ GB/token}} \approx \mathbf{3.68\text{ tokens/second}}$$

**LiquidAI/LFM2.5-8B-A1B** resolves this bottleneck through a hybrid architecture combining **Gated Short Convolutions**, **Sparse Mixture of Experts (MoE)**, and **Multi-Head Attention**:
- **Total Parameters**: **8.47 Billion** ($8,467,856,832$ parameters).
- **Active Parameters Per Token**: Only **~1.22 Billion** backbone parameters ($1,223,568,896$) or **~1.48 Billion** including the tied embedding table ($1,485,712,896$).
- **Active Weight Memory Streamed**: Only **~1.22 GB / token** in INT8 (~0.61 GB per die).
- **Theoretical Bandwidth Ceiling**:
  $$\text{Speed}_{\text{LFM2.5-MoE}} \le \frac{28.0\text{ GB/s}}{1.22\text{ GB/token}} \approx \mathbf{22.9\text{ tokens/second}}$$
- **Silicon-Calibrated Decoding Throughput (48–60% Bus Efficiency)**:
  $$\mathbf{11.0 \text{ to } 14.5\text{ tokens/second}}$$

**Result**: LFM2.5-8B-A1B delivers the broad general knowledge, deep context understanding, and coding capacity of an **8.5B parameter model** at the speed of a **1.2B model** ($3.5\times$ to $4.0\times$ faster than dense 7B/8B models on the same hardware).

---

## 2. Architectural Comparison: LFM2.5-8B-A1B vs. LFM3-1.7B

| Architectural Dimension | LFM3-1.7B (Dense Tri-Hybrid) | LFM2.5-8B-A1B (Sparse MoE Hybrid) | Architectural & Hardware Implication |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | 1.71 Billion | **8.47 Billion** ($8,467,856,832$) | $5.0\times$ larger parameter capacity and knowledge base |
| **Active Parameters / Token** | 1.71 Billion | **~1.22 Billion** (Backbone) | Streams **28% FEWER bytes** per token than LFM3-1.7B! |
| **Total Hidden Layers** | 24 Layers | 24 Layers | Identical layer count simplifies pipeline scheduling |
| **Hidden Dimension ($d_{\text{model}}$)** | 2,048 | **2,048** | Identical vector size ($1 \times 2048$ activations) |
| **ShortConv Kernel Size** | $K=7$ (`conv_L_cache = 7`) | **$K=3$ (`conv_L_cache = 3`)** | Shorter conv history; rolling state is only **8 KiB/layer** |
| **Conv Layer Count** | 13 Layers | **18 Layers** (75% of model) | Eliminates 75% of KV-cache allocations |
| **Attention Layer Count** | 11 (5 Sliding-Window + 6 Global) | **6 Layers** (Full Attention only) | Attention overhead is concentrated in only 6 layers |
| **Attention Geometry** | 16 Q, 4 KV ($d_{\text{head}} = 128$) | **32 Q, 8 KV ($d_{\text{head}} = 64$)** | $d_{\text{head}}=64$ aligns exactly to 64-byte hardware cachelines |
| **RoPE Base Frequency** | $\theta = 1,000,000$ | **$\theta = 5,000,000$** | Enhanced ultra-long context stability (up to 128K context) |
| **Feed-Forward Topology** | Dense SwiGLU (all 24 layers) | **Dense (L0–L1) + MoE (L2–L23)** | Hybrid dense-stem + sparse-trunk MoE |
| **Dense Intermediate Size** | 7,680 | 7,168 (Layers 0 and 1) | Compact dense stem warms up representations |
| **MoE Experts Geometry** | None (Dense) | **32 Experts, Top-4 Routing** | 32 experts per layer; 4 chosen dynamically per token |
| **MoE Intermediate Size** | N/A | **1,792 per expert** | SwiGLU FFN: Gate-Up $(3584, 2048)$, Down $(2048, 1792)$ |
| **Active Expert Params / Layer** | N/A | $4 \times 11.01\text{M} = 44.04\text{M}$ | Active FFN work equals dense layer work ($3 \times 7168 \times 2048$) |
| **Router Mechanism** | N/A | Sigmoid + `expert_bias` + Top-4 | Fast host CPU routing ($< 5\,\mu\text{s}$ per layer) |

---

## 3. Dual-Die NUMA Memory Allocation & Expert Parallelism (`EP=2`)

### 3.1 27 GiB Dual-Die MMZ Carveout
The FML13V03 `roma` board is configured with **27.0 GiB total NPU MMZ carveout** (13.5 GiB in `mmz_nid_0_part_0` on Die 0, and 13.5 GiB in `mmz_nid_1_part_0` on Die 1), leaving 5.0 GiB of standard DRAM for the Linux OS and runtime heaps.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     LFM2.5-8B-A1B DUAL-DIE MEMORY TOPOLOGY (27.0 GiB MMZ)                       │
│                                                                                                 │
│  [ NUMA Node 0 / Die 0 ]                                             [ NUMA Node 1 / Die 1 ]    │
│  +----------------------------------------------+                     +-----------------------+ │
│  | MMZ Carveout Pool 0: 13.50 GiB Total         |                     | MMZ Carveout Pool 1   | │
│  |                                              |                     | 13.50 GiB Total       | │
│  | • Embedding Table / LM Head: 0.26 GiB (INT8) |  D2D Interconnect   | • LM Head (High):     | │
│  | • Operator Weights (L0-L23): 0.37 GiB        | <=================> |   0.13 GiB (INT8)     | │
│  | • Dense Stem MLPs (L0-L1):   0.09 GiB        |    (~22 GB/s)       | • Routed Experts      | │
│  | • Routed Experts Even (L2-L23):              |                     |   Odd (L2-L23):       | │
│  |   16 experts x 22 layers = 3.88 GiB          |                     |   16 experts x 22 =   | │
│  | • 128K KV Cache (6 layers):  0.38 GiB        |                     |   3.88 GiB            | │
│  | -------------------------------------------- |                     | --------------------- | │
│  | Total Committed: 4.98 GiB / 13.50 GiB        |                     | Total Committed:      | │
│  | Headroom Free:   8.52 GiB (63.1% FREE)       |                     |   4.39 GiB / 13.5 GiB | │
│  |                                              |                     | Headroom Free:        | │
│  |                                              |                     |   9.11 GiB (67% FREE) | │
│  +----------------------------------------------+                     +-----------------------+ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Expert Parallelism Partitioning Strategy
1. **Zero-Crossing Expert Localization (`EP=2`)**:
   - For every MoE layer $L \in [2, 23]$, the 32 experts are partitioned by index parity:
     - **Die 0**: Even Experts $E \in \{0, 2, 4, \dots, 30\}$ ($16 \times 22 = 352$ expert weights).
     - **Die 1**: Odd Experts $E \in \{1, 3, 5, \dots, 31\}$ ($16 \times 22 = 352$ expert weights).
   - In INT8 precision, each expert consumes $11.01\text{ MB}$.
   - Total expert weights stored per die: $352 \times 11.01\text{ MB} = \mathbf{3.88\text{ GiB}}$.
2. **Statistically Balanced Dual-Die Dispatch**:
   - The router selects $k=4$ experts uniformly from the 32 available experts.
   - On average, exactly **2 experts route to Die 0** and **2 experts route to Die 1** per layer.
   - In cases of slight imbalance (3 on Die 0, 1 on Die 1), both dies execute their respective expert count independently.
3. **Active Memory Bandwidth Stream Per Die**:
   - Per MoE layer, each die streams an average of $2 \times 11.01\text{ MB} = 22.02\text{ MB}$ of weights.
   - Across all 22 MoE layers: $22 \times 22.02\text{ MB} = \mathbf{484.4\text{ MB}}$ per die.
   - Adding dense stem MLPs ($88\text{ MB} / 2 = 44\text{ MB}$) and operator projections (~$85\text{ MB}$ per die):
   - **Total Active Streamed Weights Per Die**: **~0.61 GB per token**!
   - At 14.0 GB/s local DRAM bandwidth per die:
     $$\text{DRAM Streaming Time} = \frac{0.61\text{ GB}}{14.0\text{ GB/s}} \approx \mathbf{43.5\text{ ms / token}}$$
   - Adding host CPU router dispatch and RMSNorm overhead (~$25\text{ ms}$):
     $$\text{Total Token Latency} \approx 68.5\text{ ms} \implies \mathbf{\approx 14.5\text{ tokens / second}}$$

---

## 4. NPU Class B Archetype Blueprint (< 10 Loadables vs. 62 Limit)

### 4.1 The Driver Ceiling Hazard (`eic7700_npu.ko`)
As documented in `FINDINGS.md` §33.3–§33.4, the vendor NPU driver enforces a **hard ceiling of 62 resident `.model` loadables per die**. Attempting to load a 63rd model triggers an internal driver error:
```
[eic7700_npu] io_tensor_to_io_addr: invalid frame slot
[kernel] BUG: unable to handle kernel NULL pointer dereference in destroy_frame
```
If each of the 32 experts in 22 layers were compiled as a standalone `.model` file, the model would require $22 \times 32 = \mathbf{704\text{ loadables}}$, which is **$11.3\times$ above the hard driver limit**.

### 4.2 Single-Loadable Archetype Dispatch via `DevBuf.offset`
In `FINDINGS.md` §33–§35 (proven on silicon with DeepSeek-V2-Lite), we established that **any two operations sharing identical tensor shapes and operator topologies can share a single compiled `.model` loadable**.

All 32 experts across all 22 MoE layers share the **identical tensor geometry**:
$$\text{Input: } [1, 2048, 1, 1] \xrightarrow{\text{Gate-Up}} [1, 3584, 1, 1] \xrightarrow{\text{SiLU} \times \text{Id}} [1, 1792, 1, 1] \xrightarrow{\text{Down}} [1, 2048, 1, 1]$$

Therefore, we compile **EXACTLY ONE** Class B expert archetype loadable:
$$\mathbf{expert\_mlp\_classb.model}$$

At runtime:
1. All $352$ expert weight tensors for Die 0 are packed into a single contiguous DMA buffer (`expert_weights_pool_die0.bin`).
2. All $352$ expert weight tensors for Die 1 are packed into `expert_weights_pool_die1.bin`.
3. When dispatching expert $E$ of layer $L$, the runtime simply sets the weight descriptor offset in the task descriptor:
   $$\text{WeightOffset} = (L \cdot 16 + (E / 2)) \times \text{SizeOfExpertWeights}$$
4. The NPU executes the archetype loadable pointing to that offset.
5. **Silicon Validation Result**: This was proven bit-exact on real EIC7700 hardware in `FINDINGS.md` §33.2.

### 4.3 Complete Resident Loadable Inventory Per Die

| Loadable File | Role in LFM2.5 Architecture | Input Shape | Output Shape | Count / Die |
| :--- | :--- | :--- | :--- | :--- |
| `expert_mlp_classb.model` | Archetype for all 32 experts (L2–L23) | $[1, 2048, 1, 1]$ | $[1, 2048, 1, 1]$ | **1** |
| `dense_mlp_classb.model` | Archetype for dense stem MLPs (L0–L1) | $[1, 2048, 1, 1]$ | $[1, 2048, 1, 1]$ | **1** |
| `shortconv_in_proj.model` | ShortConv input projection ($B, C, x$) | $[1, 2048, 1, 1]$ | $[1, 6144, 1, 1]$ | **1** |
| `shortconv_out_proj.model` | ShortConv output projection | $[1, 2048, 1, 1]$ | $[1, 2048, 1, 1]$ | **1** |
| `attn_qkv_proj.model` | Attention Q, K, V projection | $[1, 2048, 1, 1]$ | $[1, 3072, 1, 1]$ | **1** |
| `attn_out_proj.model` | Attention output projection | $[1, 2048, 1, 1]$ | $[1, 2048, 1, 1]$ | **1** |
| `lm_head_slice.model` | Vocabulary projection slice ($16K$ tokens) | $[1, 2048, 1, 1]$ | $[1, 16000, 1, 1]$ | **1** |
| **TOTAL RESIDENT LOADABLES** | **Serving all 24 layers & 704 experts** | — | — | **7 (< 62 Limit)** |

With only **7 resident loadables per die**, the runtime consumes only **11.3% of the driver slot table**, eliminating driver crashes and kernel oopses.

---

## 5. Execution Pipeline & Asynchronous Dual-Die Worker Model

```
                    [ Input Token ID ]
                            │
                            ▼
           [ CPU mmap Embedding Lookup (2048-D) ]
                            │
     ┌──────────────────────┴──────────────────────┐
     │                                             │
 [ Die 0 / Worker 0 ]                      [ Die 1 / Worker 1 ]
  • Layers 0..11 Operators                  • Layers 12..23 Operators
  • Even Experts (E = 0, 2, ... 30)         • Odd Experts (E = 1, 3, ... 31)
     │                                             │
     └──────────────────────┬──────────────────────┘
                            │
             [ Layer Loop L = 0 to 23 ]
                            │
      ┌─────────────────────┴─────────────────────┐
      ▼                                           ▼
[ If Layer is Conv (18 L) ]              [ If Layer is Attn (6 L) ]
  • NPU: shortconv_in_proj                 • NPU: attn_qkv_proj
  • Host/DSP: 3-tap Rolling State          • Host: RoPE + RMSNorm
  • NPU: shortconv_out_proj                • Host/DSP: FlashAttn / GQA
  • Add Residual                           • NPU: attn_out_proj
      │                                    • Add Residual
      └─────────────────────┬─────────────────────┘
                            │
             [ FFN Block Evaluation ]
                            │
      ┌─────────────────────┴─────────────────────┐
      ▼                                           ▼
[ Layers 0..1 (Dense Stem) ]             [ Layers 2..23 (MoE Trunk) ]
  • NPU: dense_mlp_classb                  • Host: Lfm2MoeTopKRouter (< 5 us)
  • Output: ffn_out                        • Identify Top-4 Experts {e1, e2, e3, e4}
                                           • Partition into Die 0 & Die 1 sets
                                           • Asynchronous Dispatch:
                                             - Die 0 executes even experts
                                             - Die 1 executes odd experts
                                           • D2D Reduction:
                                             y = sum(w_k * Expert_k(x))
      │                                           │
      └─────────────────────┬─────────────────────┘
                            │
                [ Next Layer / LM Head ]
                            │
                            ▼
          [ Dual-Die Tensor-Parallel LM Head ]
          • Die 0 evaluates vocab 0..63,999
          • Die 1 evaluates vocab 64,000..127,999
          • Host finds global argmax -> Next Token ID!
```

---

## 6. Detailed Implementation Roadmap

### Phase A: Architecture Validation & Golden Oracle (HOST ONLY) — [COMPLETED]
- [x] Create directory `/home/chri/aixigua/deep/models/LFM2.5-8B-A1B` and symlinks.
- [x] Download full model weights and tokenizer from Hugging Face (`LiquidAI/LFM2.5-8B-A1B`).
- [x] Develop and execute `inspect_lfm2_moe.py`:
  - Catalog all 8.47B parameters and verify 24-layer schedule (18 conv, 6 attention).
  - Verify dense stem (L0–L1) vs. routed MoE (L2–L23).
  - Verify 32 experts, top-4 routing, $K=3$ conv cache, and $\theta=5,000,000$.
- [x] Develop and execute `inspect_and_golden.py`:
  - Run prompt `"The capital of France is"` through PyTorch reference model (`export_golden_lfm2_moe.py`).
  - Record reference top-10 logits, greedy tokens, and all 24 intermediate layer hidden states (`golden_ref.bin`).
- [x] Complete architectural plan `PLAN.md`.

### Phase B: Archetype ONNX Export & EsAAC Podman Compilation — [COMPLETED]
- [x] Export the Class B archetype ONNX graphs (`T=1`, `T=8`, `T=64`) with static shapes `[1, C, 1, W]` (Opset 13):
  - [`build_lfm2_moe_pkg.py`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/build_lfm2_moe_pkg.py) exports `conv_in`, `proj2048`, `attn_qkv`, `expert_mlp` (serving both 704 MoE experts and 8 quarter-slices of L0–L1 dense stem), and `lm_head_s0..s3`.
- [x] Compile archetypes with `EsAAC` inside `simulator` container:
  - Generated optimized `.model` binaries (`lfm2_expert_mlp_t{1,8,64}_classb.model`, etc. — **12 resident models/die** in default mode, **16 models/die** with `--speculative-pld`).
- [x] Build Weight Packing & Slicing Tool ([`build_lfm2_moe_pkg.py`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/build_lfm2_moe_pkg.py)):
  - Exported all 352 Die 0 even experts + 4 dense quarter-experts + L0–L11 operators + 4 LM Head slices to `weights_die0.bin` (`5.166 GiB`).
  - Exported all 352 Die 1 odd experts + 4 dense quarter-experts + L12–L23 operators + 4 LM Head slices to `weights_die1.bin` (`5.166 GiB`).
- [x] Calibrate INT8 quantization scales (`quant_scales.json`):
  - Applied per-channel SmoothQuant (`α = 0.5`) absorbed into `operator_norm` and `ffn_norm`, plus `truncate = 26` `int16` output calibration across all 712 expert/stem MLP slices.

### Phase C: Dual-Die C++ Interactive Chat REPL ([`lfm2_moe.cpp`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp)) — [COMPLETED]
- [x] Implement `MoERouter`:
  - Fast host C++ implementation of Sigmoid router + `expert_bias` + Top-4 selection + normalized weights (`< 5 µs/layer`).
- [x] Implement `DualDieWorkerPool` Batched Expert & LM Head Dispatch:
  - Added `IpcCmd::SubmitExpertBatch` and `IpcCmd::SubmitLmHeadBatch` in [`worker_pool.cpp`](file:///home/chri/aixigua/romanaccio/src/eswin_runtime/src/backend/worker_pool.cpp) for single-IPC multi-expert and 4-slice LM Head execution per die.
  - Worked around the Linux kernel `eswin_rsvmem_heap_mmap` 32-bit `vma_size` truncation bug (`sext.w a3, s2` at `0xffffffff80d00f86`) by splitting each `5.166 GiB` weight blob across 3 `< 2.0 GiB` MMZ chunks (`m_w_blob[2][3]`) and evicting VFS page cache via `POSIX_FADV_DONTNEED` (`8.78 GiB` free OS RAM).
- [x] Implement ShortConv 3-tap (`K=3`) state buffer in C++ (`18` layers).
- [x] Implement OpenMP GQA (`32` Q heads, `8` KV heads, `d_head = 64`, `θ = 5,000,000` RoPE, QK-RMSNorm) for the `6` global attention layers.
- [x] Integrate exact BPE rank-order tokenizer (`encode_lfm2_text`), sampler, `--no-think` `<think>` (`124901`) logit masking, and interactive REPL.

### Phase D: Silicon Deployment & Verification on `roma` — [COMPLETED]
- [x] Transfer `LFM2.5-8B-A1B_ENNP` package to `roma:/home/chri/models/LFM2.5-8B-A1B_ENNP`.
- [x] Verify bit-exact logits against `golden_ref.bin`:
  - **Min Layer Cosine (24L)**: **`0.95180935`** (`>= 0.9500` `[PASS]`).
  - **128K Final Logits Cosine**: **`0.99521100`** (`>= 0.9950` `[PASS]`).
  - **Greedy 4-Token Match**: **`4/4 exact`** (`' Paris'`, `'.'`, `' city'`, `[PASS]`).
  - **Gate Integrity**: `DETERMINISM` + `SENSITIVITY` + `RESTORE` `[PASS]`.
- [x] Benchmark decoding & prefill throughput on `roma`:
  - **Golden Benchmark Decode (`15` tok)**: **`11.30 tok/s` (`88.50 ms/tok`)** (`exp_mlp`: `35.92 GB/s` driver / `30.15 GB/s` master; `LM Head`: `55.44 GB/s` driver / `51.87 GB/s` master) — **meets the `11.0–14.5 tok/s` target!**
  - **CLI Prose Generation (`64` tok, `--no-think`)**: **`11.25 tok/s` (`90.3 ms/tok`)** decode, **`24.23 tok/s`** prefill.
  - **332-Token C-Refactoring Benchmark (`160` tok generated)**: **`33.46–34.15 tok/s` (`29.3 ms/tok`)** prefill, **`10.39 tok/s` (`96.9 ms/tok`)** `T=1` decode.

---

### Phase E: `INT16` Operators, `dyntok` `INT16` LM Head & DropBOS SmoothQuant Fix (Cross-Ref `src/LFM3/PLAN.md` §7 Milestone 16) — [COMPLETED & VERIFIED ON SILICON — 2026-11-20]
- [x] Compiled `9` `INT8 -> INT16` operator templates (`conv_in` `sh=14`, `proj2048` `sh=13`, `attn_qkv` `sh=14` across `T={1,8,64}`) and `1` `INT8 -> INT16` 16K LM Head template (`lm_head_16k_int16_npu_b1.model`, `sh=16`) via `EsAAC`, all verified on `EsSimulator` (`cosDistance = 1.000000`, `relativeDiff = 0.000000`), preserving `5.166 GiB/die` and `12 / 62` resident loadables/die (`16 / 62` with `--speculative-pld`).
- [x] Updated [`build_lfm2_moe_pkg.py`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/build_lfm2_moe_pkg.py) with DropBOS RMS SmoothQuant (excluding `BOS` `124894` sink outliers > `3.0×` median from `s_x` and `in_t` while bounding `y_norm_max` across all calibration tokens), upgrading all 48 operator regions + 8 LM Head slices to `INT16` `dyntok` output (`0` `qm_raw > 32767` overflows; `14,470 / 128,000` Nov-8 `INT8` LM Head clipping channels eliminated; Nov-8 `sh2=27` `expert_mlp` retained unchanged).
- [x] Updated [`lfm2_moe.cpp`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp) with `unpack_npu_output_int16_dyntok()`, `min_amax = max(safe_thresh * 0.05f, 1e-8f)`, `m_emb_norm_smooth` `dyntok` LM Head packing/unpacking (`Ops out=int16, ExpertMLP out=int16, LMHead=int16`), and `s_out1=16384`, `s_out8=98304`, `s_lmout=32768`.
- [x] **Silicon Gate PASSED (`roma`, 2026-11-20)**: `--verify-golden golden_ref.bin` achieved **Min Layer Cosine `0.97924192`** (`L00=0.999548`, `L23=0.986258`), **`embedding_norm` `0.97533698`**, **128K Logits Cosine `0.99837211`**, **`4/4` greedy exact match**, and **`12.67 tok/s`** decode (`78.95 ms/tok`); unseen Python coding prompts (palindrome `--no-think` `11.02 tok/s`, palindrome `--speculative-pld --think-level low` `10.64 tok/s`, linked-list reverse `--no-think` `11.94 tok/s`) generated clean executable Python verified on host `python3` with `ZERO setDevice errors`.

---

### Phase F: Persistent 1 TB Hybrid Prefix Cache (`LFMKVB01` + `RadixCache`) & OpenAI Standard Telemetry — [COMPLETED & VERIFIED ON SILICON — 2026-11-20]

#### F.1 Why `LFM2.5-8B-A1B` Is Even Lighter to Cache Than `LFM3-1.7B`
Because `LFM2.5-8B-A1B` prefills at `34.15–35.9 tok/s` (`~101 s` cold at `3,818` tokens) and has **zero Sliding-Window Attention (`SWA`) layers** (`18` `ShortConv` layers with `K=3` $\implies$ only `2` rolling taps + `6` Global Attention layers with `8 × 64` KV heads), its persistent prefix cache footprint is extraordinarily compact:

| State Plane (`B = 64` tokens, matching `kPrefillBatch = 64`) | `LFM2.5-8B-A1B` Exact Tensor Geometry | Raw Bytes | 4 KiB-Aligned `.kvb` Size | Restore Read Policy at Hit Chunk $k$ |
| :--- | :--- | :--- | :--- | :--- |
| **Section 0: Header + `token_ids[64]` + `last_h[2048]`** | `256 B` `KvbHeader` + `256 B` tokens + `8,192 B` FP32 `last_h` | `8,704 B` | **`12,288 B`** (`3 × 4 KiB`) | Read **only** for terminal hit chunk `k` |
| **Section 1 (Plane 1): Global KV (`u8` + `sc` + `k_sum`)** | `6` layers × `8` KV heads × `64` tok × (`64 B` K + `64 B` V + `8 B` `sc` + `2 B` `sum`) | `423,936 B` | **`425,984 B`** (`104 × 4 KiB`) | Read for **all** chunks `0 .. k` (`416 KiB/chunk`) |
| **Section 2 (Plane 2): SWA KV Delta** | **None** (`num_swa_layers == 0`) | `0 B` | **`0 B`** | Zero bytes stored or read! |
| **Section 3 (Plane 3): `ShortConv` Rolling State** | `18` layers × `2048` ch × `2` taps × `4 B` (`FP32`) | `294,912 B` | **`294,912 B`** (`72 × 4 KiB`) | Read **only** for terminal hit chunk `k` (`O(1)`) |
| **Section 4 (Plane 1b): Short-Ctx `FP32` Global KV** | `6` layers × `8` KV heads × `64` tok × `64` dim × `2` × `4 B` (`k < 4` only) | `1,572,864 B` | **`1,572,864 B`** (`k < 4`; `0 B` for `k >= 4`) | Read **only** when `matched_tokens < 256` |
| **Total `.kvb` File Size (`block_idx >= 4`)** | — | — | **`733,184 B` (`716.0 KiB`)** | **1 TB NVMe holds `~1,400,000` blocks (`~90M` tokens / `~11,000` 8K sessions!)** |

- **Measured `3,818`-Token (`59`-Chunk) Silicon NVMe Restore**: `19.14 ms` (`8K`-scaled `41.53 ms <= 45 ms`), `cold = 101,033.7 ms -> warm = 1,408.7 ms` (**`71.7×` speedup**) with `42` suffix tokens, and `18.79 ms` restore / `6.01 ms` total on an exact 64-token-aligned `0`-suffix hit (`3,776` tok).

#### F.2 Detailed Step-by-Step Implementation Checklist

1. **Step F.1 — Shared `prefix_cache.hpp` & Host Unit Tests (Cross-Ref `src/LFM3/PLAN.md` §8.3–§8.4) — [COMPLETED & HOST-VERIFIED 2026-11-09]**:
   - [x] Implement [`src/eswin_runtime/include/prefix_cache.hpp`](file:///home/chri/aixigua/romanaccio/src/eswin_runtime/include/prefix_cache.hpp) with `KvbHeader` (`magic = "LFMKVB01"`, `sizeof == 256`), `ModelArchId::LFM2_5_8B_A1B = 2`, 128-bit chained `Hash128` block keys, crash-safe WAL index (`index.wal` + `index.snap`), leaf-first GDSF eviction (`num_children == 0`), `KVB_FLAG_PINNED` support, and background async NVMe writer thread.
   - [x] Build and run `src/eswin_runtime/tests/test_prefix_cache.cpp` on the local Arch Linux host, explicitly testing the `LFM2.5-8B-A1B` geometry (`kv_heads = 8`, `head_dim = 64`, `num_swa_layers = 0`, `num_conv_layers = 18`, `conv_state_taps = 2`) — **4/4 passed in `485.7 ms`**.

2. **Step F.2 — Automatic Cache Invalidation Across 3-Chunk MMZ Weight Blobs & Phase E Re-Calibration — [COMPLETED]**:
   - [x] In [`lfm2_moe.cpp`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp) (`init()`), combined 64-bit FNV-1a `head`/`mid`/`tail` hashes across the 3 MMZ weight chunks per die with `cpu_weights.bin`, `quant_chan_scales.bin`, `embedding.bin`, and scale tables into `ModelCacheGeometry::model_fingerprint` (`b608f88372c074619b116881579772f1`), preventing any stale `.kvb` hits after Phase E updates.
   - [x] Pinned the main thread to `Core 0` (Die 0) and `Core 4` (Die 1) during weight loading and `flush_cache()` in `init()` before `pin_openmp_threads_numa()`, avoiding remote-die cache coherency misses under `ES_SYS_MemFlushCache` (`> 1 MiB` -> `SYS_MemFlushCacheAllByNid`, `FINDINGS.md` §52.8).

3. **Step F.3 — NUMA-Aware `8`-Head Scatter/Gather, Tier-0 Sticky Session State & Full-Clock `1.50 GHz` Determinism in `lfm2_moe.cpp` — [COMPLETED]**:
   - [x] **NUMA Local Scatter (`NumaKvBuffer`)** & **Page-Cache Preservation Rule** (`POSIX_FADV_DONTNEED` restricted strictly to `weights_die{0,1}.bin`, never `.kvb`).
   - [x] **Tier-0 Sticky Turn State (`0.00 ms` restore)**: Tracks `m_live_tokens`, `m_turn_snap_pos`, `m_turn_snap_conv`, and `m_turn_snap_last_h` across consecutive `--daemon` requests (`HIT_TIER0` continuation & snapshot rewind).
   - [x] **Tier-1 NVMe `.kvb` Restore, Async Save, `T=8` PLD History Seeding & CLI**: Added `prefill_cached()`, `append_tokens()`, `commit_verify_t8()` 64-token checkpointing, `--cache-dir`, `--cache-max-gb`, `--no-prefix-cache`, `--verify-prefix-cache [FILE]`, REPL `/cache` & `/cache-clear`, and default `--daemon [PORT]` `8089`.
   - [x] **Full-Clock `1.50 GHz` PEC Column #2 Workaround (`kT8SafeSlots`)**: Isolated a Die 0 NPU `1.50 GHz` timing margin hazard on `W16A16` `down_proj` in `mlp_t8` PEC column #2 (`t % 8 in {4, 5}`, `0/2000` errors at `1.04 GHz`; `FINDINGS.md` §52.8). Updated `pack_gathered_int8_t8()` and `run_die_experts()` to map `<= 6` gathered tokens onto `kT8SafeSlots[6] = {0, 1, 2, 3, 6, 7}` while zero-padding slots `{4, 5}` for bit-exact full `1.50 GHz` execution.

4. **Step F.4 — Unified OpenAI Endpoint & Standard Prompt-Caching Telemetry (`src/LFM3/lfm3_openai_server.py`) — [COMPLETED]**:
   - [x] Extended `run_daemon_server()` in `lfm2_moe.cpp` with early `"prefill_done"` event, full cache telemetry on `"done"`/`"logits"`, and `cache_stats` / `cache_clear` JSON commands.
   - [x] Registered `"lfm2.5-8b-a1b"` (`--moe-port 8089`, `--backend-map`) in [`src/LFM3/lfm3_openai_server.py`](file:///home/chri/aixigua/romanaccio/src/LFM3/lfm3_openai_server.py) with `usage.prompt_tokens_details.cached_tokens`, `stream_options.include_usage` trailing `choices: []` chunk, `cache_control` pinning, `prompt_cache_key`, `cache: false`, `X-Cache-*` headers, `GET /v1/cache/stats`, and `POST /v1/cache/clear`.

5. **Step F.5 — Acceptance Gates — [ALL PASSED — 2026-11-20]**:
   - [x] **Host Offline Gate**: `test_prefix_cache` (`4/4` in `485.7 ms`) + `pytest -v src/LFM3/test_openai_prefix_telemetry.py` (`5/5` in `0.43 s`).
   - [x] **Gate 1 (`--verify-prefix-cache bench_c_code_4k.c`, `3,818` tok on `roma`)**: **ALL 5/5 SUB-TESTS PASS** (`max_abs=0.000000`, `cos=1.00000000`, Test 2 `restore=19.14 ms`, `8K=41.53 ms <= 45 ms`, `cold=101,033.7 ms -> warm=1,408.7 ms` = `71.7x`, `greedy=4/4`, Test 3 `6.01 ms`, Test 4a/4b `0.00 ms`).
   - [x] **Gate 2 (`--verify-golden golden_ref.bin` on `roma`)**: **PASS** (`Min Layer Cos = 0.97924192`, `128K Logits Cos = 0.99837211`, `DETERMINISM`/`SENSITIVITY`/`RESTORE` PASS, `4/4` greedy exact, **`11.38 tok/s`** decode).
   - [x] **Gate 3 (Live OpenAI Gateway -> `roma:8089`)**: **PASS** (`GET /v1/models`, Turn 1 `MISS`, Turn 2 SSE `PARTIAL` `tier1_nvme` `restore_ms=8.13` `cached_tokens=64` trailing `choices: []` chunk, Turn 3 repeat `HIT` `tier0_ram` `cached_tokens=125`, `/v1/cache/stats`, `/v1/cache/clear`, clean `dmesg`).

