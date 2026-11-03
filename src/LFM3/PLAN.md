# LiquidAI LFM3-1.7B (Tri-Hybrid) & LFM3 Multimodal Family Porting Plan on ESWIN EIC7700X

> **Revision 2 (2026-11-02) — corrections after review against measured silicon.**
> §2.1's hyperparameters and §2.2's parameter arithmetic were independently re-verified against the
> upstream `config.json` and are correct as written. The following were wrong or unproven and are now
> fixed: **(1)** prefill was absent from the plan entirely — at `~230 ms/prompt token` (measured on
> this runtime for Qwen2.5-Coder-1.5B) a 4K-token agentic prompt would take ~15 minutes, so
> **Milestone 3.5 (T-wide batched graphs) is now mandatory, not optional**; **(2)** host-CPU
> attention was never budgeted and becomes dominant past ~32K context (§1.1 item 3);
> **(3)** the "conv on CPU saves 3.5 ms/token" claim assumed a 3-subgraph NPU alternative that
> multi-op compilation makes unlikely — now a gated experiment (§3.1); **(4)** the §1.2 byte column
> double-counted the tied LM head, and Mode C's `6.6–9.5 tok/s` ignored per-layer cross-die
> reduction; **(5)** "FP16 Class B loadables are byte-identical across experts" was labelled proven
> but never tested (§6.3); **(6)** the chat template cited `<|think|>/<|etik|>` tokens that are not
> in the tokenizer config; **(7)** there is no `LFM3-8B-A1B` — the MoE tier is LFM2.5 (§6.1);
> **(8)** vendor benchmark provenance is now stated (§1.3).
>
> **Sequencing note.** This plan supersedes active work on `src/DeepSeek-V2-Lite/`, which is *paused*,
> not blocked: its MoE dispatch primitives are proven (`FINDINGS.md` §33, §35 — offset-selected
> experts byte-exact, 104 experts collapsing to 15 interchangeable loadables) and what remains there
> is volume (MLA attention, 1664 experts to compile, class map, router integration). LFM3-1.7B is
> dense, reuses only primitives already proven on this hardware, and ships sooner. The DeepSeek work
> is not wasted — `DmaBufferSlice`, `count_stream_classes.py` and Milestone 15's `ExpertLoadableSet`
> are exactly what an `LFM2.5-8B-A1B` port needs (§6.3).

## 1. Executive Summary & Calibrated Feasibility Assessment

### 1.1 Feasibility Verdict: **HIGH (Best-in-Class Sub-2B Long-Context & Agentic Architecture for EIC7702)**

Porting **[`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B)** (and using **[`LiquidAI/LFM3-1.7B-ONNX`](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX)** as the bit-exact golden reference and quantization oracle) to the dual-die **ESWIN EIC7700X / EIC7702** platform (`roma` FML13V03) is **highly feasible**, with zero hardware blockers and substantial architectural advantages over standard dense Transformers of the same parameter count.

However, a rigorous analysis grounded in the updated silicon findings (`FINDINGS.md` §21–§35 and `deep/BUGS.md`) reveals **four critical engineering realities** that must govern the port:

1. **The Prefill Wall — for an agentic model, prompt ingestion, not decode, is the bottleneck**:
   - Every subgraph in this plan is static `[1, C, 1, 1]`, i.e. **one token per submit set**. Prefill therefore re-streams all `1.47 GB` of weights *per prompt token*, and the cost scales linearly with prompt length.
   - This is measured, not hypothetical: the same runtime records **230 ms per prompt token** for `Qwen2.5-Coder-1.5B` (`findings/QWEN2_5_CODER_PORTING_AND_DUAL_DIE.md` §4.2), whose bytes/token (`~1.5 GB`) is within 2% of LFM3's. Scaled to LFM3-1.7B that is **~200–250 ms per prompt token**:

     | prompt length | unbatched prefill (T=1) | batched prefill (T=64) |
     | :--- | :--- | :--- |
     | 512 tokens | **~2 min** | ~1–2 s |
     | 4K tokens (system prompt + tool schemas + transcript) | **~15 min** | ~7–15 s |
     | 32K tokens (document ingest) | **~2 h** | ~1–2 min |

   - LFM3's stated advantage is agentic/tool-calling work, which is prompt-heavy. Shipping 5 tok/s decode with 4–5 token/s prefill makes the model unusable for exactly the workload it was selected for. **This plan originally treated prefill only as a 5-token functional test; that is the single largest gap in it.**
   - **Solution — Milestone 3.5**: compile T-wide variants of the pointwise subgraphs (`[1, C, T, 1]`) and prefill in chunks. Note that batching *only* the 24 MLPs (`37 + 12 = 49` loadables/die, `< 62`) still leaves the 48 non-MLP projections (`334M` params = `0.668 GB` FP16 / `0.334 GB` INT8, plus 48 submits) running at `T=1`, which floors prefill at `~108 ms/tok` in FP16 (`~61 ms/tok` in INT8). To batch **100% of layer weights** (`≤ 35 ms/tok` FP16, `≤ 20 ms/tok` INT8): either **(a)** use fused ShortConv from Milestone 3 (`25` decode + `24` `T=64` prefill = **`49` loadables/die**, zero swapping), or **(b)** swap a full 36-loadable `T=64` prefill set in once per turn (`~1.2 s` parallel swap across both dies at `34 ms`/load, `FINDINGS.md` §33.3).
   - `findings/CONTEXT_WINDOW_AND_MEMORY.md` §5 already specifies chunked prefill for the *attention-memory* reason; Milestone 3.5 implements it for the *bandwidth* reason, which is roughly 60× larger at these sizes.

2. **`LiquidAI/LFM3-1.7B-ONNX` Cannot Be Fed Monolithically to `EsAAC` — It Must Be Decomposed**:
   - The upstream `LFM3-1.7B-ONNX` checkpoint is an ONNX Runtime stateful graph with dynamic sequence axes (`[batch, seq_len, hidden]`), external `.onnx_data` shard files, ORT contrib operators (`GroupQueryAttention`, `MatMulNBits`, `RotaryEmbedding`), and **39 dynamic inputs/outputs** (`input_ids`, `attention_mask`, `position_ids`, `logits_to_keep`, 13 `past_conv.N` state tensors, and 11 `past_key_values.N.{key,value}` tensors).
   - As established in `FINDINGS.md` §22.1 and §25 (`DeepSeek-V2-Lite/PLAN.md` §6.11), `EsAAC` requires **ONNX Opset 13**, **static 4D `[1, C, 1, 1]` shapes**, **single-file embedded initializers**, and **SIGSEGVs (exit 139)** on general multi-input dynamic graphs.
   - **Solution**: Follow the proven `Qwen2.5-Coder-1.5B` (`FINDINGS.md` §22–§23) and `DeepSeek-V2-Lite` (`FINDINGS.md` §29–§30) methodology: export static 4D `[1, C, 1, 1]` Opset-13 ONNX subgraphs per layer from `LiquidAI/LFM3-1.7B` SafeTensors, while using `LiquidAI/LFM3-1.7B-ONNX` via `onnxruntime` for automated per-layer differential verification and activation range calibration.

3. **Why LFM3's Tri-Hybrid Backbone Outperforms Standard 1.5B–2B Transformers on EIC7700X**:
   - Out of 24 layers, **only 6 layers use full global attention**. **13 layers are Gated Short Convolutions (`k=7`)** with an $O(1)$ fixed 6-token rolling state (`24 KiB`/layer), and **5 layers are Sliding-Window Attention (SWA)** capped at a **256-token window** (`512 KiB`/layer).
   - Consequently, KV-cache memory growth is **4.0× smaller** than a 24-layer full-attention model (`12 KiB/token` across the entire model vs. `48 KiB/token`), and CPU/DSP attention score overhead at long context (4K–32K tokens) is **4.4× lower** because 18 of the 24 layers never attend past 256 tokens.
   - **Caveat — that 4.4× is relative, and the absolute cost still has to be paid.** All 11 attention layers run on the host CPU in this design, and the 6 global layers read `12,288 · N` bytes of KV per decoded token at context `N`. Budgeted against the `~210 ms/token` NPU weight stream (see §1.2):

     | context | global-KV read/token | ≈ CPU attention time | share of the token budget |
     | :--- | :--- | :--- | :--- |
     | 4K | 50 MB | ~8 ms | negligible |
     | 32K | 400 MB | ~65 ms | ~30% |
     | 128K | 1.6 GB | ~260 ms | **CPU-bound** |
     | 256K | 3.2 GB | ~520 ms | **~5× the NPU cost** |

     So the 256K context is genuine in *memory* terms (`3.2 GB` of host RAM) but decode becomes CPU-attention-bound somewhere between 32K and 128K. Mitigations to evaluate in Milestone 4, in order of cost: OpenMP over the 16 heads pinned to the owning die's 4 cores; INT8 global KV (halves the traffic); and offloading scores to the 8× Vision Q7 DSPs (512-bit vectors, `findings/NPU_OPERATIONS.md`). **Add an on-board attention-kernel microbenchmark as a Milestone 4 gate** — this is currently the largest unmeasured term in the design.

4. **Calibrated Decode Throughput (`FINDINGS.md` §24) & The 62-Model/Die Driver Limit (`FINDINGS.md` §33.3–§33.4)**:
   - Per `FINDINGS.md` §24, single-die local read bandwidth is **~14.0 GB/s** (`41–59%` realized efficiency on `Qwen2.5-Coder-1.5B`), and layer-pipeline parallelism (`PP=2`, Layers 0–11 on Die 0, Layers 12–23 on Die 1) streams weights from one die at a time (except for the tensor-parallel LM Head, which aggregates both dies at **~28 GB/s**).
   - Per `FINDINGS.md` §33.3–§33.4, `eic7700_npu.ko` enforces a **hard ceiling of 62 resident `.model` loadables per die** (loading a 63rd triggers `ERR_PTR(-EINVAL)` in `io_tensor_to_io_addr` and a kernel Oops in `destroy_frame`).
   - Under `PP=2`, LFM3-1.7B requires **37 loadables on Die 0** and **37 loadables on Die 1** (or **25 loadables/die** if `Lfm3ShortConv` is compiled as a single multi-op graph), staying safely below the 62-model ceiling.

---

### 1.2 Calibrated Resource Footprint & Throughput Projections

All throughput projections below use the silicon-calibrated model from `FINDINGS.md` §24 (`14.0 GB/s` single-die ceiling for `PP=2` layers, `28.0 GB/s` dual-die ceiling for `TP=2` layers/LM head, and the measured `41–59%` end-to-end efficiency band):

| Deployment Mode | Weight Precision | Total NPU MMZ (Die 0 + Die 1) | Host RAM (`mmap` Embedding) | Resident Loadables / Die | Bytes Streamed / Token (decode) | Ceiling Speed | **Calibrated Realistic Speed** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode A: Layer-Pipeline (`PP=2`, Default)** | **INT8** (`make_quant_table.py`) | **~1.73 GiB** (`0.87 GiB/die`) | `0.50 GiB` (FP16) | **37** (`< 62` safe) | **`1.47 GB` total** = `1.21 GB` layers (one die at a time, `14 GB/s`) + `0.26 GB` tied head (both dies, `28 GB/s`) | `9.5–10.5 tok/s` | **3.9 – 5.6 tokens/s** |
| **Mode B: Layer-Pipeline (`PP=2`, Baseline)** | **FP16** (`--gen-fp16-model=yes`) | **~3.46 GiB** (`1.73 GiB/die`) | `0.50 GiB` (FP16) | **37** (`< 62` safe) | **`3.46 GB` total** = `2.93 GB` layers + `0.52 GB` head | `4.1–4.4 tok/s` | **1.8 – 2.6 tokens/s** |
| **Mode C: Tensor-Parallel (`TP=2`, Optimization)** | **INT8** (`make_quant_table.py`) | **~1.73 GiB** (`0.87 GiB/die`) | `0.50 GiB` (FP16) | **49** (`< 62` safe, 2/layer + head) | **`1.47 GB` split across both dies** (`28 GB/s` aggregate local reads) | `~19 tok/s` | **5.5 – 8.0 tokens/s** (see caveat) |

> **Corrections to the byte accounting (the earlier version of this table double-counted the LM head).**
> The tied head is `128,000 × 2048` = `262.1M` params = `0.26 GB` INT8 / `0.52 GB` FP16, and it is
> *part of* the 1.47 GB / 3.46 GB model, not additional to it. Mode A's `9.5 tok/s` ceiling
> corresponds to `1.47 GB ÷ 14 GB/s = 105 ms`; splitting the head across both dies
> (`1.21/14 + 0.26/28 = 96 ms`) gives `10.5`. Either way the **41–59% band yields 3.9–5.6 tok/s**,
> which is the number to plan against — and note it is essentially the *same* speed as the already
> ported `Qwen2.5-Coder-1.5B` (measured 4.18–5.49 tok/s), because decode is bytes-per-token bound
> and the two models stream within 2% of each other. **LFM3's win over Qwen2.5-1.5B is quality per
> byte, KV footprint and context length — not tokens/second.**

> **Mode C caveat — do not quote 6.6–9.5 tok/s until measured.** `TP=2` splits every layer across
> both dies, which adds (a) a cross-die reduction of the `2048`-wide activation after each
> row-split `w2`/`o_proj` — small in bytes (`4 KB`) but a *sync point* per layer, and (b) roughly
> `+23` submits/token (`~+6 ms`) versus `PP=2`. The `41–59%` band was measured on a `PP`-style
> pipeline with one sync at the head, not 24 syncs per token, so it does not transfer unchanged.
> Treat Mode C as an optimization to be benchmarked after Mode A works, and expect the lower half
> of the range.

> **Prefill is a separate budget and is not in this table.** Decode streams `1.47 GB` once per
> generated token; unbatched prefill streams it once per *prompt* token, i.e. `~230 ms/token`
> measured on this runtime (§1.1 item 1). Milestone 3.5 exists to fix that. A plan that only
> quotes decode throughput for an agentic model is quoting the cheaper half of the workload.

> **Why start with Mode B (FP16 `PP=2`) then Mode A (INT8 `PP=2`)?**
> `EsAAC --gen-fp16-model=yes` requires zero quantization tables and eliminates the INT8 subgraph boundary scaling gap (`FINDINGS.md` §31 item 1.6), allowing end-to-end functional bring-up and perplexity verification first. Milestone 5 then layers in calibrated INT8 (`make_quant_table.py`, `FINDINGS.md` §30 & §33.5) with explicit FP16 subgraph I/O boundaries to double throughput to **3.9–5.6 tok/s**.

---

### 1.3 Provenance of the Quality Claims (read before repeating them)

The "best-in-class sub-2B" framing comes from Liquid AI's own launch material
(`liquid.ai/blog/lfm3`, Oct 27 2026). Two cautions:

- The headline average (LFM3-1.7B **49.37** vs Qwen3.5-9B **47.24**) is over an agentic suite the
  vendor selected, and the same table shows it **losing** on long-context retrieval (AA-LCR 52.83
  vs 62.00), BrowseComp-Plus (37.00 vs 38.00) and EnterpriseOps-Gym (28.20 vs 28.97). It is a
  strong result for a 1.7B model; it is not a uniform win over a 9B model, and where it wins
  (BFCL V4 71.39 vs 62.83, IFEval 96.66 vs 92.08) is tool-calling and instruction-following.
- Long context is precisely where *this port* carries its own risk (§1.1 item 3: host-CPU attention
  on the 6 global layers becomes dominant past ~32K tokens). The capability the vendor benchmarks
  worst is the one our hardware taxes most. Validate the 256K claim locally with a
  needle-in-a-haystack sweep at 4K / 32K / 128K after Milestone 4, reporting measured accuracy
  **and** ms/token — not the model card's context length.

Treat vendor numbers as motivation for choosing the model, never as an acceptance gate. Every gate
in this plan is a measurement on `roma`.

---

## 2. LFM3-1.7B Architecture Deep-Dive & Exact Parameter Arithmetic

### 2.1 Hyperparameters (Verified from `LiquidAI/LFM3-1.7B/config.json` & `modeling_lfm3.py`)

| Property | Value | Architectural & EIC7700X Hardware Implication |
| :--- | :--- | :--- |
| `model_type` | `lfm3` | Dense Tri-Hybrid architecture (ShortConv + SWA + Global Attention) |
| `num_hidden_layers` | `24` | 12 layers on Die 0 (`L0..L11`), 12 layers on Die 1 (`L12..L23`) |
| `hidden_size` ($D$) | `2048` | Exact multiple of 256 (`8 × 256`); ideal for NPU CMAC $C_0=32$ alignment |
| `intermediate_size` ($D_{\text{ffn}}$) | `7680` | Exact multiple of 256 (`30 × 256`); SwiGLU MLP `w1`, `w3` (`2048→7680`), `w2` (`7680→2048`) |
| `layer_types` | 13 `conv`, 5 `sliding_attention`, 6 `full_attention` | Deterministic schedule: `full_attention` if $i \bmod 4 == 2$; `sliding_attention` if $i \bmod 4 == 0 \land i > 0$; else `conv` |
| `conv_L_cache` / `conv_bias` | `7` / `false` | Causal depthwise 1D convolution with kernel size $K=7$, state cache $[2048, 6]$ |
| `num_attention_heads` ($H_q$) | `16` | Query projection $16 \times 128 = 2048$ |
| `num_key_value_heads` ($H_{kv}$) | `4` | GQA 4:1 ratio; Key/Value projections $4 \times 128 = 512$ |
| `head_dim` ($d_h$) | `128` | Standard 128-dim heads with per-head `q_norm` and `k_norm` (`Lfm3RMSNorm(128)`) |
| **Attention Output Gate** | `gate_proj: 2048 → 16` | Head-wise sigmoid gate $\sigma(W_{\text{gate}} x) \in \mathbb{R}^{16 \times 1}$ scales each Q head prior to `o_proj` |
| `sliding_window` | `256` | Applied to the 5 `sliding_attention` layers (`L4, L8, L12, L16, L20`) |
| **RoPE (SWA Layers)** | $\theta = 10,000.0$, factor $= 1.0$ | Full 128-dim RoPE rotation on all 5 `sliding_attention` layers |
| **RoPE (Global Layers)** | $\theta = 1,000,000.0$, factor $= 0.5$ | **Zero-Tail Partial RoPE**: rotates first 64 dims (32 pairs) using full-128 frequency progression; last 64 dims unrotated ($\cos=1, \sin=0$) |
| `use_sandwich_norm` | `true` (`norm_eps = 1e-5`) | **4 RMSNorms per layer**: `operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm` |
| `vocab_size` | `128,000` | Split `64,000` per die (`250 × 256` channels — exact 256-alignment without padding!) |
| `tie_word_embeddings` | `true` | Input embedding (`mmap` on CPU) and LM Head (`lm_head_d0`/`d1` on NPU) share weights |
| `max_position_embeddings` | `262,144` (256K) | Compile-time static KV cache only needed for the **6 `full_attention` layers** |

### 2.2 Exact Layer-by-Layer Topology & Parameter Audit

The 24 layers follow the exact pattern defined in `configuration_lfm3.py`:

```text
Die 0 (Layers 0..11):
  L00: conv               L04: sliding_attention  L08: sliding_attention
  L01: conv               L05: conv               L09: conv
  L02: full_attention     L06: full_attention     L10: full_attention
  L03: conv               L07: conv               L11: conv
  --> Summary Die 0: 7x conv, 2x sliding_attention, 3x full_attention

Die 1 (Layers 12..23):
  L12: sliding_attention  L16: sliding_attention  L20: sliding_attention
  L13: conv               L17: conv               L21: conv
  L14: full_attention     L18: full_attention     L22: full_attention
  L15: conv               L19: conv               L23: conv
  --> Summary Die 1: 6x conv, 3x sliding_attention, 3x full_attention
```

**Closed Parameter Arithmetic**:
- **Tied Token Embedding / LM Head**: $128,000 \times 2048 = \mathbf{262,144,000}$ params ($262.14\text{M}$).
- **13 Gated Short Convolution Operators (`Lfm3ShortConv`)**:
  - `in_proj` ($2048 \to 6144$): $12,582,912$
  - `conv.weight` ($2048 \times 1 \times 7$ depthwise): $14,336$
  - `out_proj` ($2048 \to 2048$): $4,194,304$
  - Subtotal: $16,791,552 \times 13 = \mathbf{218,290,176}$ params ($218.29\text{M}$).
- **11 Gated GQA Attention Operators (`Lfm3Attention`)**:
  - `q_proj` ($2048 \to 2048$): $4,194,304$
  - `k_proj` ($2048 \to 512$): $1,048,576$
  - `v_proj` ($2048 \to 512$): $1,048,576$
  - `gate_proj` ($2048 \to 16$): $32,768$
  - `o_proj` ($2048 \to 2048$): $4,194,304$
  - `q_norm` ($128$) + `k_norm` ($128$): $256$
  - Subtotal: $10,518,784 \times 11 = \mathbf{115,706,624}$ params ($115.71\text{M}$).
- **24 SwiGLU Feed-Forward Networks (`Lfm3MLP`)**:
  - `w1` ($2048 \to 7680$) + `w3` ($2048 \to 7680$) + `w2` ($7680 \to 2048$) = $3 \times 15,728,640 = 47,185,920$ per layer.
  - Subtotal: $47,185,920 \times 24 = \mathbf{1,132,462,080}$ params ($1,132.46\text{M}$).
- **97 RMSNorm Vectors** ($4 \times 24$ Sandwich Norms + $1$ Final `embedding_norm` $\times 2048$): $\mathbf{198,656}$ params ($0.20\text{M}$).
- **Grand Total**: $262.14\text{M} + 218.29\text{M} + 115.71\text{M} + 1,132.46\text{M} + 0.20\text{M} = \mathbf{1,728,801,536\text{ parameters (1.73B)}}$ — matching Liquid AI's published 1.73B figure to three decimal places.

---

## 3. Mathematical Formulation & Hardware Operator Mapping

Every LFM3 decoder layer $l \in [0, 23]$ executes a **Sandwich-Normalized** two-stage residual block:

$$\begin{aligned}
u_l &= \text{RMSNorm}_{\text{op\_pre}}(h_{l}) \\
a_l &= \begin{cases}
\text{Lfm3ShortConv}_l(u_l, \text{state}^{\text{conv}}_l) & \text{if } \text{layer\_types}[l] = \text{"conv"} \\
\text{Lfm3Attention}_l(u_l, K^{\text{cache}}_l, V^{\text{cache}}_l) & \text{if } \text{layer\_types}[l] \in \{\text{"sliding\_attention"}, \text{"full\_attention"}\}
\end{cases} \\
m_l &= h_l + \text{RMSNorm}_{\text{op\_post}}(a_l) \\
v_l &= \text{RMSNorm}_{\text{ffn\_pre}}(m_l) \\
h_{l+1} &= m_l + \text{RMSNorm}_{\text{ffn\_post}}\big(W_2 (\text{SiLU}(W_1 v_l) \odot W_3 v_l)\big)
\end{aligned}$$

### 3.1 Stage A1: Gated Short Convolution Block (`Lfm3ShortConv`, 13 Layers)

From `modeling_lfm3.py`, for single-token decoding ($u_l \in \mathbb{R}^{2048}$):
1. **Input Projection (NPU Subgraph `conv_in_l{N}.model`)**:
   - $P = W_{\text{in\_proj}} u_l \in \mathbb{R}^{6144}$, lowered as a $1\times 1$ Conv2D $[1, 2048, 1, 1] \to [1, 6144, 1, 1]$.
   - Split $P$ along channel axis into three vectors of length $D = 2048$:
     $$B = P[0:2048], \quad C = P[2048:4096], \quad x' = P[4096:6144]$$
2. **Pre-Gate, Rolling State Update & 7-Tap Depthwise Conv1D (Host CPU L1 Cache — `< 5 µs`)**:
   - Compute pre-gated input: $\tilde{x}_t = B \odot x' \in \mathbb{R}^{2048}$.
   - Shift rolling state buffer $S_l \in \mathbb{R}^{2048 \times 7}$ left by 1 tap and insert $\tilde{x}_t$ at tap 6 (or use a 7-slot ring index).
   - Compute causal 7-tap depthwise convolution and post-gate in one fused OpenMP/RVV pass:
     $$y_t[c] = C[c] \cdot \sum_{k=0}^{6} W_{\text{conv}}[c, k] \cdot S_l[c, t - 6 + k] \quad \forall c \in [0, 2047]$$
   - **Why run the 7-tap depthwise step on CPU instead of NPU?** *(decision provisional — see the gate below)*
     $2048 \times 7 = 14,336$ MACs (`28.6 KB` of weights) is trivial arithmetic, whereas an NPU submission costs **233–305 µs** of IPC/driver overhead (`FINDINGS.md` §28.2). The CPU path is therefore *safe*.
     Two claims in the original version of this section do not survive scrutiny, and neither changes the safety conclusion:
     - *"Fits entirely inside the P550's 32 KB L1D"* — the `28.6 KB` of **weights** do; the **state** does not. `13 × 2048 × 7` taps is `319 KiB` in FP16 (`638 KiB` in FP32), i.e. L2/L3-resident, and each token reads `2048 × 7` per layer. Realistic cost is ~10 µs/layer (`~130 µs/token`), not `< 4 µs`. Still immaterial next to a ~210 ms token budget.
     - *"Saves 13 × 0.27 ms = 3.5 ms/token"* — this assumes the NPU alternative is **three** subgraphs per conv layer (`in_proj`, depthwise, `out_proj`). But `EsAAC` compiles multi-op graphs into one loadable (proven: the 6-op SwiGLU expert FFN, `FINDINGS.md` §28–§29), and a rolling state is expressible as a Class B input/output buffer pair exactly like a KV cache (§25). A **fused** conv layer would then be **1 submit**, versus **2** for the CPU split — so the CPU path may *cost* `13 × 0.27 ms` rather than save it.
     **Gate (Milestone 3, cheap):** compile one fused `conv_in → depthwise k=7 over a 7-slot state → conv_out` loadable for layer 0 and check (a) that `EsAAC` accepts a depthwise `Conv` with kernel `(7,1)` at all — every loadable verified so far on this project is `1×1` — and (b) whether it reproduces the CPU result byte-for-byte. If it does, prefer the fused graph and delete 13 loadables *and* 13 submits per token. If `EsAAC` rejects or mis-compiles depthwise convs, keep the CPU path and record why. **Do not carry this as a settled 3.5 ms saving either way.**
3. **Output Projection (NPU Subgraph `conv_out_l{N}.model`)**:
   - $a_l = W_{\text{out\_proj}} y_t \in \mathbb{R}^{2048}$, lowered as a $1\times 1$ Conv2D $[1, 2048, 1, 1] \to [1, 2048, 1, 1]$.

### 3.2 Stage A2: Head-Wise Gated Attention Block (`Lfm3Attention`, 11 Layers)

From `modeling_lfm3.py`, for single-token decoding ($u_l \in \mathbb{R}^{2048}$):
1. **Fused QKV Projection (NPU Subgraph `attn_qkv_l{N}.model`)**:
   - Concatenate $W_q \in \mathbb{R}^{2048\times 2048}$, $W_k \in \mathbb{R}^{512\times 2048}$, $W_v \in \mathbb{R}^{512\times 2048}$ into a single $1\times 1$ Conv2D $[1, 2048, 1, 1] \to [1, 3072, 1, 1]$ ($3072 = 12 \times 256$ channels, exact hardware alignment).
   - **What about `gate_proj` ($2048 \to 16$)?**
     Do **not** concatenate the 16-channel `gate_proj` into `attn_qkv_l{N}.model`, because $3072 + 16 = 3088$ breaks 256-channel alignment and causes `NVDLAGEMMBuilder` / `EsAAC` channel-padding waste (`FINDINGS.md` §31 item 1.4). Instead, evaluate $g = \sigma(W_{\text{gate}} u_l) \in \mathbb{R}^{16}$ on the host CPU in **~13 µs** (4× smaller than the $64\times 2048$ MoE router gate benchmarked at $53.4\,\mu\text{s}$ on `roma` in `FINDINGS.md` §31) while the NPU is executing `attn_qkv_l{N}.model`!
2. **Per-Head QK RMSNorm & Zero-Tail Partial RoPE (Host CPU in FP32)**:
   - Reshape $Q \to [16, 128]$, $K \to [4, 128]$, $V \to [4, 128]$.
   - Apply `q_norm` (`Lfm3RMSNorm(128, eps=1e-5)`) independently to each of the 16 query heads, and `k_norm` independently to each of the 4 key heads.
   - Apply RoPE **in FP32** (as required by `modeling_lfm3.py` to avoid ~5% per-layer BF16/FP16 ULP accumulation error):
     - **If `sliding_attention` ($l \in \{4, 8, 12, 16, 20\}$)**: Full RoPE ($\theta = 10,000.0$, all 64 pairs rotated):
       $$\omega_i = 10000^{-2i/128} \quad \text{for } i \in \{0, \dots, 63\}$$
     - **If `full_attention` ($l \in \{2, 6, 10, 14, 18, 22\}$)**: Zero-Tail Partial RoPE ($\theta = 1,000,000.0$, `partial_rotary_factor = 0.5`):
       $$\omega_i = \begin{cases} 1000000^{-2i/128} & \text{for } i \in \{0, \dots, 31\} \\ 0.0 \ (\cos=1, \sin=0) & \text{for } i \in \{32, \dots, 63\} \end{cases}$$
       *(Note: The exponent denominator is still the full `head_dim = 128`, NOT `64`, and rotation uses Llama split-half pairing $(x[i], x[i+64])$ for $i \in [0, 31]$ while leaving $x[32..63]$ and $x[96..127]$ unchanged.)*
3. **KV Cache Update, GQA Attention & Head-Wise Sigmoid Gating (Host CPU)**:
   - **If `sliding_attention`**: Write $(K_t, V_t)$ into circular ring slot $t \bmod 256$ of the 256-entry SWA cache, and attend over $\min(t + 1, 256)$ tokens.
   - **If `full_attention`**: Write $(K_t, V_t)$ into slot $t$ of the global KV cache, and attend over $0 \dots t$.
   - Multiply each head's attention output $\text{head}_h \in \mathbb{R}^{128}$ by its scalar gate $g_h = \sigma((W_{\text{gate}} u_l)_h)$:
     $$\widetilde{\text{head}}_h = g_h \cdot \text{Attention}(Q_h, K_{\lfloor h/4 \rfloor}, V_{\lfloor h/4 \rfloor}) \quad \forall h \in [0, 15]$$
4. **Attention Output Projection (NPU Subgraph `attn_out_l{N}.model`)**:
   - $a_l = W_o \widetilde{\text{head}} \in \mathbb{R}^{2048}$, lowered as $1\times 1$ Conv2D $[1, 2048, 1, 1] \to [1, 2048, 1, 1]$.

### 3.3 Stage B: SwiGLU MLP Block (`Lfm3MLP`, All 24 Layers)

- **Fused Multi-Op NPU Subgraph (`mlp_l{N}.model`)**:
  - Graph: `w1` (`Conv 2048→7680`) + `Sigmoid` + `Mul` (SiLU) + `w3` (`Conv 2048→7680`) + `Mul` + `w2` (`Conv 7680→2048`).
  - Compiled directly by `EsAAC` as a single multi-op `.model` loadable per layer (proven bit-exact in `FINDINGS.md` §22.2 for `Qwen2.5-Coder`'s `1536→8960→1536` MLP and `FINDINGS.md` §29–§30 for `DeepSeek-V2-Lite`).

---

## 4. Dual-Die Memory Layout & Subgraph Residency Plan

```text
+---------------------------------------------------------------------------------------------------+
|                        LFM3-1.7B DUAL-DIE MEMORY TOPOLOGY (DEFAULT 22 GiB MMZ)                    |
|                                                                                                   |
|  [ NUMA Node 0 / Die 0 (Cores 0..3) ]                [ NUMA Node 1 / Die 1 (Cores 4..7) ]         |
|  +----------------------------------------------+    +------------------------------------------+ |
|  | MMZ Pool 0: 11.00 GiB Total                  |    | MMZ Pool 1: 11.00 GiB Total              | |
|  |                                              |    |                                          | |
|  | • Layers 0..11 Loadables (36 files):         |    | • Layers 12..23 Loadables (36 files):    | |
|  |   - 7x Conv In/Out:        0.12 GiB (INT8)   |    |   - 6x Conv In/Out:      0.10 GiB (INT8) | |
|  |   - 5x Attn QKV/Out:       0.05 GiB (INT8)   |    |   - 6x Attn QKV/Out:     0.06 GiB (INT8) | |
|  |   - 12x SwiGLU MLP:        0.55 GiB (INT8)   |    |   - 12x SwiGLU MLP:      0.55 GiB (INT8) | |
|  | • LM Head Slice 0 (0..63,999): 0.13 GiB      |    | • LM Head Slice 1 (64k..128k): 0.13 GiB  | |
|  | • DMA Activation & Logits Buffers: 0.02 GiB  |    | • DMA Activation & Logits:     0.02 GiB  | |
|  | -------------------------------------------- |    | ---------------------------------------- | |
|  | TOTAL NPU MMZ (INT8):  0.87 GiB / 11.00 GiB  |    | TOTAL NPU MMZ (INT8): 0.86 / 11.00 GiB   | |
|  | TOTAL NPU MMZ (FP16):  1.73 GiB / 11.00 GiB  |    | TOTAL NPU MMZ (FP16): 1.72 / 11.00 GiB   | |
|  | RESIDENT MODELS:       37 / 62 ceiling (SAFE)|    | RESIDENT MODELS:      37 / 62 (SAFE)     | |
|  +----------------------------------------------+    +------------------------------------------+ |
|                                                                                                   |
|  [ Host Linux System RAM (NUMA 0 + NUMA 1) ]                                                      |
|  • Tied Embedding Table (128,000 × 2048 FP16 = 500.0 MiB) -> Zero-copy mmap via EmbeddingTable    |
|  • 13x ShortConv Rolling States (13 × 2048 × 7 × 2B = 358 KiB) -> L1/L2 cache-resident           |
|  • 5x SWA 256-Token Circular KV Caches (5 × 512 KiB = 2.50 MiB) -> L2/L3 cache-resident          |
|  • 6x Global Attention KV Caches (16K ctx = 192 MiB; 32K ctx = 384 MiB) -> NUMA-pinned RAM       |
|  • 11x Attention Gate Weights (11 × 16 × 2048 FP16 = 704 KiB) + 97x RMSNorm Weights (388 KiB)    |
+---------------------------------------------------------------------------------------------------+
```

### Guarding Against All Known `deep/BUGS.md` & `FINDINGS.md` Pitfalls

1. **62-Model/Die Kernel Driver Limit (`FINDINGS.md` §33.3–§33.4)**:
   - Each die loads exactly **37 `.model` files** (`12 layers × 3 subgraphs + 1 LM head slice`). This leaves a 25-model safety margin below the 62-model driver limit where `eic7700_npu` crashes in `destroy_frame`.
2. **DMA Buffer Size Floor (`FINDINGS.md` §31 item 1.1.3)**:
   - `conv_in_l{N}.model` outputs `[1, 6144, 1, 1]` (`12,288 B` in FP16) and `lm_head_d{0,1}.model` outputs `[1, 64000, 1, 1]` (`128,000 B` in FP16). Buffer allocation must size `gemm_out` to $\ge \max(6144, 7680) \times 4 = 30,720\text{ B}$ (page-aligned to `32,768 B`) and `logits_out` to `131,072 B`.
3. **Cache Coherency Directionality (`FINDINGS.md` §31 item 1.1)**:
   - Keep `CacheMode::NoCache` as the default baseline until the standalone empirical directionality probe verifies whether `ES_SYS_MemFlushCache` invalidates CPU cache lines on NPU-written buffers.
4. **INT8 Boundary Scaling & Input Calibration (`FINDINGS.md` §30, §31 item 1.6, §33.5)**:
   - When generating `table.json` via `make_quant_table.py`, always keep subgraph input/output tensors in **`fp16`** (or explicitly multiply/divide by `int8_step = threshold / 127.0` at every subgraph boundary) and calibrate `input` thresholds using **real post-RMSNorm activations** captured from `LiquidAI/LFM3-1.7B-ONNX`, never unit-normal noise or the `1.0` default fallback.

---

## 5. Step-by-Step Implementation Plan (Flash-Ready Milestones)

Each milestone below is self-contained, specifies exact file paths, inputs, outputs, and CLI commands, and includes an explicit pass/fail verification gate so a fast coding model (Flash) can execute them sequentially without guesswork.

**Which milestones are actually flash-safe.** "Self-contained with a pass/fail gate" is not the same
as "low judgement". Assign by what can go wrong silently:

| Milestone | Tier | Why |
| :--- | :--- | :--- |
| 1 — download, inspect, golden oracle | **flash** | **COMPLETED (2026-11-03)**: cos 0.999999 vs ORT, 175 arrays in `golden_ref.npz` |
| 2 — static 4D ONNX export + sidecar weights | **flash**, max for review | **COMPLETED (2026-11-03)**: all 80 ONNX subgraphs (72 layer + 8×16K LM head) + CPU sidecars exported, cos 1.00000000 across all gates |
| 3 — `EsAAC` batch compile + verify | **flash** | **COMPLETED (2026-11-03)**: all 80 `.model` files compiled & byte-size audited (40/die < 62); verified in `eswin-sim` (cos ≥ 0.9999994), `EsSimulator` & on `roma` silicon |
| 3 (fused-conv experiment) | **max** | **COMPLETED (2026-11-03)**: verdict = REJECTED by `EsAAC` (`SdpEmitter.cpp:243` `Check failed: i_precision <= PRECISION_FP16`, exit 134); isolated depthwise `k=7` compiles in 70.4 ms; keep split `conv_in` + `conv_out` + CPU 7-tap ring |
| **3.5 — batched T-wide prefill** | **max** | The residency arithmetic sits one model below a kernel-Oops ceiling, chunked CPU state updates must stay sequential, and numerics must match the T=1 path. Highest-consequence milestone in the plan |
| 4 — C++ tri-hybrid engine | **max** | Four new primitives, two KV cache disciplines, partial RoPE with a split-half pairing rule, and a CPU-attention kernel whose cost has never been measured here |
| 4 (attention microbenchmark) | **flash** | A timed loop; the *interpretation* belongs to whoever owns Milestone 4 |
| 5 — INT8 calibration | **max** for the table design, **flash** for running it | Threshold policy (absmax vs percentile, FP16 boundaries, per-channel mandatory) decides accuracy; the compile loop is mechanical |
| 6 — TP=2 + `Cached` DMA | **max** | Depends on the still-unresolved `MemFlushCache` direction question (audit 1.1) and adds a per-layer cross-die sync |

---

### Milestone 1: Checkpoint Acquisition, Architecture Inspector & Golden Reference Oracle [COMPLETED 2026-11-03]

- **Implementation & Verification Status** (2026-11-03):
  - **Artifacts Acquired**:
    - `LiquidAI/LFM3-1.7B` in `deep/models/LFM3-1.7B/`: `model.safetensors` (3,457,635,448 B, 1,728,801,536 parameters bit-exact), `config.json`, `configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `chat_template.jinja`.
    - `LiquidAI/LFM3-1.7B-ONNX` in `deep/models/LFM3-1.7B-ONNX/`: `onnx/model_fp16.onnx` (270,733 B), `onnx/model_fp16.onnx_data` (2,131,447,808 B), `onnx/model_fp16.onnx_data_1` (1,460,367,360 B) — all 39 dynamic inputs (4 sequence controls, 13 `past_conv.N` states, 22 `past_key_values.N` states) verified.
  - **Inspector & Oracle Tool**: `src/LFM3/inspect_and_golden.py` committed (`49107af`).
  - **Configuration Assertions**: Validated `config.json` against §2.1 (hidden 2048, intermediate 7680, 24 layers, 13 conv, 5 sliding_attention, 6 full_attention, 16 Q heads, 4 KV heads, vocab 128000, conv_L_cache 7, sliding_window 256, sandwich norm enabled) — all assertions PASSED.
  - **Differential Parity Gate**: Prompt `"Hello, world! Today is"` (token IDs `[35808, 20, 1530, 9, 10767, 355]`, `seq_len = 6`) executed on PyTorch (FP32) vs. ONNX Runtime (FP16):
    - **Cosine Similarity: 0.999999** (acceptance gate $\ge 0.9995$ PASSED).
    - **Max Absolute Difference: 0.022583**.
  - **Greedy Continuation**: 4 autoregressive steps generated: `[17479, 20, 309, 358]` (`" Monday"`, `","`, `" and"`, `" I"`).
  - **Oracle Archive**: Saved to `/home/chri/aixigua/deep/artifacts/lfm3_golden/golden_ref.npz` (175 arrays: `input_ids`, `prompt_text`, `pt_logits`, `ort_logits`, `generated_token_ids`, `embed_tokens`, `embedding_norm`, plus per-layer `op_norm`, `conv_out`/`attn_out`, `post_op_norm`, `ffn_norm`, `mlp_out`, `post_ffn_norm`, `block_out`).
- **Goal**: Download `LiquidAI/LFM3-1.7B` (SafeTensors) and `LiquidAI/LFM3-1.7B-ONNX` (`fp16` variant), verify all tensor shapes against §2, and generate deterministic golden reference tensors for 1-token prefill and 4-step autoregressive decode.
- **Files to Create**:
  - `src/LFM3/inspect_and_golden.py`
- **Step-by-Step Instructions**:
  1. Download `LiquidAI/LFM3-1.7B` (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors`) into `/home/chri/aixigua/deep/models/LFM3-1.7B/` and `LiquidAI/LFM3-1.7B-ONNX` (`onnx/model_fp16.onnx`, `onnx/model_fp16.onnx_data`) into `/home/chri/aixigua/deep/models/LFM3-1.7B-ONNX/`.
  2. In `src/LFM3/inspect_and_golden.py`:
     - Parse `config.json` and assert: `hidden_size == 2048`, `intermediate_size == 7680`, `num_hidden_layers == 24`, `num_attention_heads == 16`, `num_key_value_heads == 4`, `vocab_size == 128000`, `conv_L_cache == 7`, `sliding_window == 256`, `use_sandwich_norm == True`.
     - Run a 5-token prompt (`[1, 15043, 29892, 920, 526]`) through the Hugging Face PyTorch reference (`modeling_lfm3.py` in FP32/BF16) and/or `onnxruntime` on `model_fp16.onnx`.
     - Hook every layer $l \in [0, 23]$ and save a compressed golden archive `/home/chri/aixigua/deep/artifacts/lfm3_golden/golden_ref.npz` containing:
       - `input_ids`, `embed_out`
       - Per-layer `op_norm_in`, `op_out`, `post_op_norm_out`, `ffn_norm_in`, `mlp_out`, `post_ffn_norm_out`, `layer_out`
       - Per-layer `conv_state` (for the 13 `conv` layers) and `k_cache`, `v_cache` (for the 11 attention layers)
       - Final `logits` (`[1, 5, 128000]`) and greedy next-token IDs for 4 autoregressive steps.
- **Acceptance Gate**:
  - `/home/chri/aixigua/venv/bin/python src/LFM3/inspect_and_golden.py` exits with `0`, prints the 24-layer type table matching §2.2, and writes `golden_ref.npz` with cosine similarity $\ge 0.9995$ between PyTorch BF16 and ONNX Runtime FP16 outputs.

---

### Milestone 2: Static 4D Opset-13 ONNX Subgraph Exporter & Sidecar Weight Extractor [COMPLETED 2026-11-03]

- **Implementation & Verification Status** (2026-11-03):
  - **Script**: `src/LFM3/export_lfm3.py` created and verified.
  - **Artifact Locations & Structure (Reference for Milestones 3, 3.5 & 4)**:
    1. **CPU Sidecar Files** (`/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/`):
       - `embedding.bin`: Raw FP16 row-major table (`[128000, 2048]`, exactly `524,288,000` bytes = `500.0 MB`) for zero-copy CPU mmap via `EmbeddingTable` (Milestone 4).
       - `cpu_weights.bin`: Dense binary of all 143 CPU weight tensors (`2,993,152` bytes ≈ `2,923.0 KB`).
       - `cpu_weights.json`: JSON index mapping tensor name to `{offset, length, shape, dtype}` inside `cpu_weights.bin`. Contains:
         - 24 layers of `operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm` (all `[2048]` FP32).
         - 1 final `model.embedding_norm.weight` (`[2048]` FP32).
         - 13 depthwise Conv1D kernels `layer_{L}.conv.conv.weight` (`[2048, 7]` FP32).
         - 11 attention head QK norms `layer_{L}.self_attn.q_norm.weight` (`[128]` FP32), `k_norm.weight` (`[128]` FP32).
         - 11 attention head gates `layer_{L}.self_attn.gate_proj.weight` (`[16, 2048]` FP32).
       - `cpu_weights.npz`: Compressed NumPy archive of all 143 tensors for testing.
    2. **74 Static 4D Opset-13 NPU ONNX Subgraphs** (`/home/chri/aixigua/deep/artifacts/lfm3_onnx/`):
       - **26 ShortConv subgraphs** (13 layers: 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23):
         - `lfm3_l{00..23}_conv_in.onnx` (`48.0 MB` each, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 6144, 1, 1]`).
         - `lfm3_l{00..23}_conv_out.onnx` (`16.0 MB` each, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 2048, 1, 1]`).
       - **22 GQA Attention subgraphs** (11 layers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22):
         - `lfm3_l{02..22}_attn_qkv.onnx` (`24.0 MB` each, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 3072, 1, 1]`).
         - `lfm3_l{02..22}_attn_out.onnx` (`16.0 MB` each, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 2048, 1, 1]`).
       - **24 SwiGLU MLP subgraphs** (all layers 0..23):
         - `lfm3_l{00..23}_mlp.onnx` (`180.0 MB` each, full SwiGLU: `w1(2048->7680)` + `Sigmoid` + `Mul` + `w3(2048->7680)` + `Mul` + `w2(7680->2048)`).
       - **2 Dual-Die Tied LM Head slices**:
         - `lfm3_lm_head_d0.onnx` (`500.0 MB`, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 64000, 1, 1]`, Die 0).
         - `lfm3_lm_head_d1.onnx` (`500.0 MB`, `1x1 Conv2D [1, 2048, 1, 1] -> [1, 64000, 1, 1]`, Die 1).
  - **Differential Parity Gate Results in ONNX Runtime** (evaluated against `golden_ref.npz`):
    - Gate 1 (Layer 0 Conv In/Out & MLP): MLP `cos = 1.00000000`, `max_diff = 4.2e-7` (PASSED).
    - Gate 2 (Layer 2 Global Attn QKV & MLP): MLP `cos = 1.00000000`, `max_diff = 4.2e-7` (PASSED).
    - Gate 3 (Layer 4 SWA MLP): MLP `cos = 1.00000000`, `max_diff = 1.37e-6` (PASSED).
    - Gate 4 (Dual-Die LM Head merged logits): `cos = 1.00000000`, `max_diff = 3.81e-6` (PASSED).
  - **Downstream Coupling**:
    - Milestone 3 (`batch_compile_lfm3.py`) compiles the 74 `.onnx` files from `/home/chri/aixigua/deep/artifacts/lfm3_onnx/` into `.model` loadables in `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/models/`.
    - Milestone 4 (`lfm3_v2.cpp`) reads `embedding.bin` and `cpu_weights.bin` (indexed by `cpu_weights.json`) directly from `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/`.
- **Goal**: Export all NPU-targeted linear/MLP subgraphs as static 4D `[1, C, 1, 1]` Opset-13 ONNX files (compatible with `EsAAC`), and package all CPU-side weights (embeddings, RMSNorms, depthwise conv kernels, attention gates) into fast binary files.
- **Files to Create**:
  - `src/LFM3/export_lfm3.py`
- **Step-by-Step Instructions**:
  1. Model `src/LFM3/export_lfm3.py` after `src/qwen2_5_coder/export_qwen2_5_coder.py` and `src/DeepSeek-V2-Lite/export_expert_ffn.py`:
     - Always use `torch.onnx.export(..., opset_version=13, dynamo=False, do_constant_folding=True)` with `nn.Conv2d(..., kernel_size=1, bias=False)` and 4D tensors `[1, C, 1, 1]`.
  2. Export the **CPU binary files** to `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/`:
     - `embedding.bin`: `model.embed_tokens.weight` cast to FP16, shape `[128000, 2048]` (`524,288,000` bytes).
     - `cpu_weights.bin` (with a fixed header or JSON index `cpu_weights.json`):
       - For all 24 layers: `operator_norm.weight` (`[2048]`), `post_operator_norm.weight` (`[2048]`), `ffn_norm.weight` (`[2048]`), `post_ffn_norm.weight` (`[2048]`) in FP32.
       - Final norm: `embedding_norm.weight` (`[2048]`) in FP32.
       - For the 13 `conv` layers (`0,1,3,5,7,9,11,13,15,17,19,21,23`): `conv.conv.weight` (`[2048, 7]`) in FP32.
       - For the 11 attention layers (`2,4,6,8,10,12,14,16,18,20,22`): `self_attn.q_norm.weight` (`[128]`), `self_attn.k_norm.weight` (`[128]`), and `self_attn.gate_proj.weight` (`[16, 2048]`) in FP32/FP16.
  3. Export the **74 NPU ONNX subgraphs** to `/home/chri/aixigua/deep/artifacts/lfm3_onnx/`:
     - **For each of the 13 `conv` layers**:
       - `lfm3_l{L:02d}_conv_in.onnx`: `1x1 Conv2d(2048, 6144, bias=False)` with `conv.in_proj.weight`. Input `x: [1, 2048, 1, 1]`, Output `y: [1, 6144, 1, 1]`.
       - `lfm3_l{L:02d}_conv_out.onnx`: `1x1 Conv2d(2048, 2048, bias=False)` with `conv.out_proj.weight`. Input `x: [1, 2048, 1, 1]`, Output `y: [1, 2048, 1, 1]`.
     - **For each of the 11 `attention` layers**:
       - `lfm3_l{L:02d}_attn_qkv.onnx`: `1x1 Conv2d(2048, 3072, bias=False)` with `cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)`. Input `x: [1, 2048, 1, 1]`, Output `y: [1, 3072, 1, 1]`.
       - `lfm3_l{L:02d}_attn_out.onnx`: `1x1 Conv2d(2048, 2048, bias=False)` with `self_attn.o_proj.weight`. Input `x: [1, 2048, 1, 1]`, Output `y: [1, 2048, 1, 1]`.
     - **For all 24 layers**:
       - `lfm3_l{L:02d}_mlp.onnx`: SwiGLU module (`w1: Conv2d(2048, 7680)`, `w3: Conv2d(2048, 7680)`, `w2: Conv2d(7680, 2048)` using `w2(w1(x) * torch.sigmoid(w1(x)) * w3(x))`). Input `x: [1, 2048, 1, 1]`, Output `y: [1, 2048, 1, 1]`.
     - **Dual-Die Tied LM Head (2 files)**:
       - `lfm3_lm_head_d0.onnx`: `1x1 Conv2d(2048, 64000, bias=False)` using `embed_tokens.weight[0:64000]`.
       - `lfm3_lm_head_d1.onnx`: `1x1 Conv2d(2048, 64000, bias=False)` using `embed_tokens.weight[64000:128000]`.
- **Acceptance Gate**:
  - Run `onnxruntime` on Layer 0 (`conv` layer: `lfm3_l00_conv_in.onnx`, `lfm3_l00_conv_out.onnx`, `lfm3_l00_mlp.onnx`), Layer 2 (`full_attention` layer), and Layer 4 (`sliding_attention` layer) fed with inputs from `golden_ref.npz`, and assert max absolute difference $< 10^{-4}$ and cosine similarity $\ge 0.99999$ against `golden_ref.npz`.

---

### Milestone 3: AOT Compilation (`EsAAC`) & Simulator / Single-Block Silicon Verification [COMPLETED 2026-11-03]

- **Implementation & Verification Status** (2026-11-03, full write-up in `FINDINGS.md` §36):
  - **Batch Compiler & Manifest Tool**: `src/LFM3/batch_compile_lfm3.py` created and executed using the zero-copy `/run/host-wayland/simulator-ipc/lfm3_work` ↔ `/ipc/lfm3_work` RAM mount (`> 5 GB/s`). Also fixed the `ssh simulator` upload bottleneck (`0.56 MB/s` → `140.22 MB/s`, `250×`) by passing `-W 1048576` to `dropbear` and `-b 65536` + `nodelay` to `socat` in `deep/pdoman/esaac_essimulator/run-simulator.sh`.
  - **2 MiB `/dmabuf:npu` KMD Command-Stream Limit Discovered & Resolved**:
    - Compiling a half-vocab LM head (`2048 → 64,000`) as a single `.model` (`255.74 MiB`, `250` M-splits, `502` relocations) produced `3.73 MiB` of KMD command streams (`1.56 MiB` `dep_graph` + `1.42 MiB` `op_list` + `0.75 MiB` `surf_list`), overflowing the driver's fixed **2 MiB `/dmabuf:npu` buffer** (`FINDINGS.md` §34.2) and hanging both `EsSimulator` and `roma`.
    - Splitting the LM head into **8 slices of `16,000` channels** (`4` slices on Die 0: `lfm3_lm_head_d0_s{0..3}.model`, `4` slices on Die 1: `lfm3_lm_head_d1_s{0..3}.model`, `64.89 MiB` each) keeps each slice at `125` M-splits (`252 <= 256` relocations) and **`1.86 MiB < 2.00 MiB`** of KMD streams.
  - **Exact Byte-Size Audit of All 80 Compiled `.model` Loadables** in `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/models/` (`3.41 GiB` total, `40` models/die `< 62` ceiling):
    - `13 × conv_in` (`2048 → 6144`): `25,763,520 B` (`24.57 MiB` each, 100% uniform)
    - `13 × conv_out` (`2048 → 2048`): `8,602,312 B` (`8.20 MiB` each, 100% uniform)
    - `11 × attn_qkv` (`2048 → 3072`): `12,892,608 B` (`12.30 MiB` each, 100% uniform)
    - `11 × attn_out` (`2048 → 2048`): `8,602,312 B` (`8.20 MiB` each, 100% uniform)
    - `24 × mlp` (`2048 → 7680 → 2048`): `96,346,744 B` (`91.88 MiB` each, 100% uniform)
    - `8 × lm_head` (`2048 → 16000`): `68,045,904 B` (`64.89 MiB` each, 100% uniform)
    - Dual-die manifest generated at `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/manifest.json` (`40` models on Die 0, `40` models on Die 1).
  - **Simulator (`eswin-sim` & Vendor `EsSimulator`) + Physical Silicon (`roma` `/dev/npu0`) Verification**:
    - Patched `src/eswin_simulator/eswin_simulator/cli.py` to support `w1splitM_`/`w3splitM_`/`w2splitM_` SwiGLU naming and 8-channel-aligned `c_out_split` (`88/88` pytest passing).
    - `lfm3_l00_conv_in.model`: `eswin-sim` cos = **`0.99999997`**, `EsSimulator` = **PASS**, `roma` `/dev/npu0` = **`avg 0.7710 ms`** (`min 0.726 ms`, `1,297.0 FPS`).
    - `lfm3_l02_attn_qkv.model`: `eswin-sim` cos = **`0.99999996`**, `EsSimulator` = **PASS**, `roma` `/dev/npu0` = **`avg 0.4701 ms`** (`min 0.426 ms`, `2,127.2 FPS`).
    - `lfm3_l00_mlp.model`: `eswin-sim` cos = **`0.99999941`**, `EsSimulator` = **PASS**, `roma` `/dev/npu0` = **`avg 2.5382 ms`** (`min 2.485 ms`, `394.0 FPS`).
    - `lfm3_lm_head_d0_s0.model`: `eswin-sim` cos = **`0.99999996`**, `EsSimulator` = **PASS**, `roma` `/dev/npu0` = **`avg 1.9056 ms`** (`min 1.849 ms`, `524.8 FPS`).
  - **Fused-Conv Gate Verdict (§3.1)**:
    - Isolated depthwise `Conv` (`[1, 2048, 7, 1] → [1, 2048, 1, 1]`, `groups=2048`) compiles in **70.4 ms**.
    - Fused `in_proj → Concat(conv_state, x_tilde) → depthwise Conv → out_proj` (both with `Slice` and with 3 separate $2048\to 2048$ Convs) is **REJECTED by `EsAAC`** with `F20261103 SdpEmitter.cpp:243] Check failed: i_precision <= PRECISION_FP16` in `eaac::SdpEmitter::setPostDRPConfig()` (`SIGABRT`, exit 134).
    - **Decision**: Keep the split `conv_in` + `conv_out` NPU loadables (`40` models/die total) and execute the 7-tap depthwise convolution + gating on the host CPU. In Milestone 3.5, this selects **Route B** (parallel per-turn prefill/decode loadable swap) or **Route C** (co-resident 12 `T=64` MLPs/die = `40 + 12 = 52/die < 62`).
- **Goal**: Compile all 74 ONNX subgraphs to `.model` loadables using `EsAAC --gen-fp16-model=yes` inside the `simulator` container, verify numerical accuracy in `EsSimulator` / `eswin-sim`, and benchmark representative blocks on `roma`.
- **Files to Create**:
  - `src/LFM3/batch_compile_lfm3.py`
- **Step-by-Step Instructions**:
  1. Model `src/LFM3/batch_compile_lfm3.py` after `src/qwen2_5_coder/batch_compile_all_layers.py`.
  2. For each of the 74 ONNX files, invoke `EsAAC` inside the container:
     ```bash
     EsAAC --input-model=<name>.onnx --gen-fp16-model=yes
     ```
     *Critical gotchas, all three measured on this project:*
     - **`-o` is ignored.** `EsAAC` writes to `<cwd>/model_files/<name>_npu_b1.model` when the working directory is writable, and to `/tmp/model_files/<name>_npu_b1.model` otherwise (`FINDINGS.md` §29, §35.6). Locate outputs with `find <cwd> /tmp -name '<name>_npu_b1.model' -newermt '-10 minutes'`, not by assuming a path.
     - **Exit status 0 does not mean it compiled anything.** A missing or truncated input produces `Model <name>.onnx has no graph` / `getBatchFromInputShape failed` and still returns 0. The batch script must grep every log for `has no graph` and count output files at the far end, then fail loudly on a mismatch — this silently cost two experts in the §35 sweep.
     - **The container's `/tmp` does not persist between sessions**, and the `ssh simulator` relay (socat → dropbear) runs at **0.56 MB/s per stream but scales ~8× with parallel connections** (§35.6). Transfer the ONNX files 8-way parallel; a serial upload of this model set is ~40 min versus ~5 min.
  3. Generate `manifest.json` in `/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/` mapping:
     - **Die 0 (`device_id: 0`)**: Layers `0..11` (`36` `.model` files) + `lfm3_lm_head_d0.model` = **37 files**.
     - **Die 1 (`device_id: 1`)**: Layers `12..23` (`36` `.model` files) + `lfm3_lm_head_d1.model` = **37 files**.
  4. Verify representative loadables (`lfm3_l00_conv_in.model`, `lfm3_l02_attn_qkv.model`, `lfm3_l00_mlp.model`, `lfm3_lm_head_d0.model`) against `golden_ref.npz` using `EsSimulator` (or `eswin-sim`) and `/opt/eswin/bin/es_run_model` on `roma`.
  5. **Fused-conv experiment (§3.1 gate)**: compile one `lfm3_l00_conv_fused.onnx` containing `in_proj → depthwise k=(7,1) over a 7-slot state → out_proj`, with the state passed as a second graph input *only if* it can be an initializer-free Class B buffer. Note the hard constraint: `EsAAC` **SIGSEGVs on a second dynamic graph input**, so the state must enter as a Class B memory entry produced by `compiler.py --convert-weights`-style post-processing rather than as an ONNX input. Record whether it compiles and whether it is bit-exact; this decides 13 loadables and 13 submits per token.
- **Acceptance Gate**:
  - All 74 `.model` files exist, are non-empty, and **every compile log is free of `has no graph`**.
  - `EsSimulator` and `es_run_model -v` on `roma` report cosine similarity $\ge 0.9999$ against `golden_ref.npz` for `conv_in`, `attn_qkv`, `mlp`, and `lm_head_d0`.
  - The fused-conv experiment has a recorded verdict (compiles bit-exact / rejected), not an open question.

---

### Milestone 3.5: Batched (T-wide) Prefill Graphs — the throughput fix for prompt-heavy use

- **Goal**: make prompt ingestion bandwidth-bound-once instead of bandwidth-bound-per-token. Without this, LFM3-1.7B prefills at `~230 ms/token` (§1.1 item 1) — **~15 minutes for a 4K-token agentic prompt** — which disqualifies the model for the workload it was chosen for.
- **Why it works (and why batching only MLPs is not enough for the `< 20 ms/tok` gate)**: `conv_in`, `conv_out`, `attn_qkv`, `attn_out` and `mlp` are all pointwise in the sequence dimension, so a `[1, C, T, 1]` graph computes `T` tokens for one weight stream. While the 24 MLPs represent 77% of layer weights (`1.13 GB` of `1.47 GB`), the remaining 48 projections (`26` `conv_in`/`conv_out` + `22` `attn_qkv`/`attn_out`) still total `334.0M` params (`0.668 GB` FP16 / `0.334 GB` INT8) and `48` submits/token. If those 48 projections stay at `T=1`, prefill cannot go below `~108.4 ms/tok` in FP16 (`~60.7 ms/tok` in INT8) even if MLPs took `0 ms`. All pointwise layer subgraphs must therefore be batched at `T = 64` to reach the `< 20 ms/tok` (INT8) / `< 35 ms/tok` (FP16) target.
- **Files to Create**:
  - `src/LFM3/export_lfm3_prefill.py` (or a `--tokens T` flag on `export_lfm3.py`)
- **Step-by-Step Instructions**:
  1. Re-export all pointwise layer subgraphs (`mlp`, `conv_in`, `conv_out` — or `conv_fused` — and `attn_qkv`, `attn_out`) as `[1, C_in, T, 1] → [1, C_out, T, 1]` with `T = 64` (start here; `T = 128/256` later if SPRAM allows). Same weights, same opset 13, same `1×1` Conv2D lowering — only the spatial dim changes.
  2. Compile them as a *separate* set (`*_t64.model`) and add them to `manifest.json` under a `prefill` role.
  3. **Residency arithmetic — this is the constraint, respect it**: `eic7700_npu` Oopses past 62 resident models/die (§33.4). Choose the residency layout based on Milestone 3's fused-conv gate:
     - **Route A (if Milestone 3 fused ShortConv succeeds — preferred, zero swapping)**: each die has `24` `T=1` layer loadables + `1` LM head = `25` decode loadables/die. Adding all `24` `T=64` prefill layer loadables/die gives **`25 + 24 = 49` loadables/die** (`13` models of headroom below 62), keeping both full sets resident simultaneously with zero per-turn reload.
     - **Route B (if ShortConv stays split — 36 layer loadables/die)**: keeping both full 36-loadable sets resident would require `37 + 36 = 73/die > 62` (impossible). Instead, for prompts `> 32` tokens, swap the 36-loadable `T=64` prefill set in once at the start of `prefill()` and swap the 36-loadable `T=1` decode set back in before decoding (`36 × 34.25 ms ≈ 1.23 s` per swap, executed in parallel by Worker 0 and Worker 1 across both dies — acceptable once per turn for a multi-thousand-token prompt).
     - **Route C (lightweight fallback for short/medium prompts)**: keep only the owning die's 12 `T=64` MLPs co-resident with the 37 decode loadables (**`37 + 12 = 49` loadables/die**, `13` models of headroom). Zero swap overhead, but per-token prefill is floored by the 48 `T=1` projections at `~108 ms/tok` (FP16) / `~61 ms/tok` (INT8).
  4. Prefill loop: for each chunk of `T` tokens, submit the T-wide projections and T-wide MLP, and run the CPU-side work (conv rings, RoPE, QK-norm, gates, KV append, attention) over the chunk. Conv state and KV must be updated **in order within the chunk** — the rings are sequential, so this is a `T`-step loop on the CPU even though the GEMMs are batched.
  5. Only the **last** token of the final chunk needs the LM head; do not submit the head per chunk.
- **Acceptance Gate**:
  - Measured prefill on `roma` (using Route A or Route B full-layer `T=64` batching):
    - **At Milestone 3.5 (FP16 weights, `2.93 GB`/chunk)**: **≤ 35 ms per prompt token** (≥ 28 prompt tokens/s) for a 512-token prompt, and **≤ 30 s** end-to-end for a 4K-token prompt (which streams `64 × 2.93 GB = 187.5 GB` total, i.e. a ~26.8 s floor at 7 GB/s effective).
    - **At Milestone 5 (INT8 weights, `1.21 GB`/chunk)**: **≤ 20 ms per prompt token** (≥ 50 prompt tokens/s) for a 512-token prompt, and **≤ 15 s** end-to-end for a 4K-token prompt.
    - Record both, plus the decode rate immediately after prefill to prove the KV/conv state survived the chunked path.
  - Teacher-forced logits for a 512-token prompt match the T=1 path with cosine ≥ 0.9999 (batching must not change numerics).
  - `dmesg` clean, and resident model count printed at session start and asserted `< 55`.

---

### Milestone 4: Native C++17 Tri-Hybrid Inference Engine (`lfm3_v2`)

- **Goal**: Build a standalone C++17 dual-die inference executable (`src/LFM3/lfm3_v2.cpp`) on top of `libes_runtime.so` (modeled directly on `src/qwen2_v2/qwen2_v2.cpp`) that executes the full 24-layer LFM3 tri-hybrid pipeline across Die 0 and Die 1.
- **Files to Create**:
  - `src/LFM3/lfm3_v2.cpp`
  - `src/LFM3/Makefile`
  - `src/LFM3/CMakeLists.txt`
- **Step-by-Step Instructions**:
  1. Clone the structure of `src/qwen2_v2/qwen2_v2.cpp` (which uses `WorkerPool`, `DmaBuffer`, `EmbeddingTable`, and `Tokenizer`).
  2. Implement the **4 LFM3-specific architectural primitives** in `lfm3_v2.cpp`:
     - **Primitive 1 — Sandwich RMSNorm**:
       - Pre-norm (`operator_norm`) before calling the layer operator; post-norm (`post_operator_norm`) on the operator output *before* adding to the residual stream $h$; pre-norm (`ffn_norm`) before `mlp_l{N}.model`; post-norm (`post_ffn_norm`) on the MLP output *before* adding to $h$. Use `eps = 1e-5f`.
     - **Primitive 2 — Gated ShortConv State Engine (`Lfm3ShortConvState`)**:
       - Allocate `float conv_state[13][2048][6]` (zero-initialized on session start/reset).
       - On `conv` layers (`0,1,3,5,7,9,11,13,15,17,19,21,23`):
         1. Submit `lfm3_l{L:02d}_conv_in.model` on the owning die (`[1, 2048] -> [1, 6144]` FP16).
         2. Unpack $B = \text{out}[0..2047]$, $C = \text{out}[2048..4095]$, $x' = \text{out}[4096..6143]$.
         3. Compute $\tilde{x}[c] = B[c] \cdot x'[c]$.
         4. Compute 7-tap depthwise dot product over `[conv_state[idx][c][0..5], x_tilde[c]]` with `conv_weight[idx][c][0..6]`, shift `conv_state` left by 1 tap and store $\tilde{x}[c]$ at tap 5, and multiply by $C[c]$ to produce $y[c]$.
         5. Submit `lfm3_l{L:02d}_conv_out.model` on the owning die (`[1, 2048] -> [1, 2048]` FP16).
     - **Primitive 3 — SWA vs. Global GQA Attention with QK-Norm, Zero-Tail Partial RoPE & Head Gate**:
       - Maintain two distinct KV caches:
         - **5 SWA Layers (`4,8,12,16,20`)**: Fixed circular buffer `swa_k[5][256][4][128]` and `swa_v[5][256][4][128]` (write at `pos % 256`, attend over `min(pos + 1, 256)` slots).
         - **6 Global Layers (`2,6,10,14,18,22`)**: Linear buffer `glob_k[6][max_ctx][4][128]` and `glob_v[6][max_ctx][4][128]` (write at `pos`, attend over `0..pos`).
       - On attention layers:
         1. Submit `lfm3_l{L:02d}_attn_qkv.model` (`[1, 2048] -> [1, 3072]` FP16). While NPU executes (or immediately after), compute the 16-head gate $g_h = \sigma(\sum_{d=0}^{2047} W_{\text{gate}}[h, d] u_l[d])$ for $h \in [0, 15]$ on CPU.
         2. Apply `q_norm` (`head_dim=128, eps=1e-5f`) to each of the 16 Q heads and `k_norm` to each of the 4 K heads.
         3. Apply RoPE in FP32:
            - For SWA layers: rotate all 64 pairs $(d, d+64)$ for $d \in [0, 63]$ with $\theta_d = 10000^{-2d/128}$.
            - For Global layers: rotate **only the first 32 pairs** $(d, d+64)$ for $d \in [0, 31]$ with $\theta_d = 1000000^{-2d/128}$; leave indices $d \in [32, 63]$ and $d+64 \in [96, 127]$ untouched!
         4. Compute GQA attention (`16 Q : 4 KV`, scale $1/\sqrt{128}$), multiply each head $h$'s 128-dim output by $g_h$, and submit `lfm3_l{L:02d}_attn_out.model` (`[1, 2048] -> [1, 2048]` FP16).
     - **Primitive 4 — Dual-Die Parallel LM Head & Chat Template**:
       - After Layer 23 on Die 1 and final `embedding_norm`, broadcast the normalized hidden state `[1, 2048]` to Die 0 and Die 1, submit `lfm3_lm_head_d0.model` and `lfm3_lm_head_d1.model` concurrently (`64,000` FP16 logits per die), and merge top-K logits across all `128,000` tokens.
       - **Chat template — read it from the checkpoint, do not transcribe it.** Verified from `LiquidAI/LFM3-1.7B/tokenizer_config.json`: `bos_token = <|startoftext|>`, `eos_token = <|im_end|>`, `pad_token = <|pad|>`, and **`additional_special_tokens` is empty**. The `<|think|>…<|etik|>` pair that appeared in an earlier draft of this plan is not in the tokenizer config; load the repo's `chat_template` string at runtime (or copy it verbatim into `manifest.json` during Milestone 1) and derive the special-token list from `tokenizer.json`. Because EOS is `<|im_end|>` and the template is ChatML-shaped, the existing atomic-delimiter rule applies: **parse ChatML delimiters before BPE subword segmentation** (QWEN.md §7, Qwen2 specifics), and reuse the Issue #47 sampler guards (sliding-window repetition penalty, n-gram cycle detector, EOS `151643`/`151645` handling generalized to LFM3's ids).
     - Default sampling `temperature=0.1, top_k=50, repetition_penalty=1.1`, with `--no-think` only if the retrieved `chat_template` actually defines a thinking mode.
- **Acceptance Gate**:
  - Cross-compile with `riscv64-linux-gnu-g++`, deploy to `roma`, and run a 5-token teacher-forced prefill + 4-token greedy decode test (`--verify-golden golden_ref.npz`).
  - Assert that the top-1 predicted token IDs match `golden_ref.npz` across all 4 greedy steps and sustained FP16 generation speed reaches **$\ge 1.8\text{ tokens/sec}$** (Mode B calibrated band, §1.2) with zero kernel errors in `dmesg`.
  - **Report a per-stage timing breakdown, not just the total**: NPU submit time, CPU conv-ring time, CPU attention time (SWA and global separately), CPU norms/gates, and D2D transfer. A single aggregate number cannot distinguish "working" from "working but with the wrong thing dominating", and §1.1 item 3 predicts CPU attention grows with context — that prediction needs a baseline at 512 tokens to be checkable later.
  - **Attention-kernel microbenchmark on `roma`**: measure the global-attention CPU kernel at 4K and 32K context and compare against the §1.1 item 3 budget (~8 ms and ~65 ms per token). If measured cost exceeds the budget by more than 2×, stop and fix the kernel (OpenMP over heads, NUMA pinning, INT8 KV) before continuing to Milestone 5 — do not discover it at 128K context.

---

### Milestone 5: Real-Activation INT8 Quantization (`make_quant_table.py`) & 2x Speedup

- **Goal**: Quantize the 74 subgraphs (or at minimum the 24 SwiGLU MLPs + 2 LM Head slices, which represent **80.7% of all model weights**) to INT8 using `make_quant_table.py` calibrated on real LFM3 activations, doubling throughput to **3.9–5.6 tokens/sec** while preserving FP16 subgraph I/O boundaries.
- **Files to Create**:
  - `src/LFM3/calibrate_lfm3_int8.py`
- **Step-by-Step Instructions**:
  1. **Collect Real Calibration Activations (`FINDINGS.md` §30 & §33.5)**:
     - Run 32 diverse calibration sentences (512 tokens total) through `LiquidAI/LFM3-1.7B` on host and record the exact input tensor distributions for every `conv_in`, `conv_out`, `attn_qkv`, `attn_out`, `mlp`, and `lm_head` subgraph.
     - **The graph input must be calibrated, not defaulted.** `make_quant_table.py` originally never observed graph *inputs* (`expose_intermediates()` only exposes tensors produced by a node), so they silently fell back to a `1.0` threshold and would have clipped real post-RMSNorm activations — measured absmax **4.02** on DeepSeek expert FFN input (`FINDINGS.md` §33.5). That is fixed; verify the emitted table's `input` threshold is ~the observed absmax and not `1.0` before compiling anything.
     - `absmax` versus `99.99th-percentile` is a real decision, not a detail: `make_quant_table.py` uses **absmax** (matching the vendor table's `int8_step == threshold/127` symmetric scheme), while this milestone's earlier text said percentile. Percentile reduces range but **clips outliers silently**. Pick one deliberately, record which, and check the resulting perplexity — do not mix them between subgraphs.
  2. **Synthesize `table.json` with FP16 Boundaries (`FINDINGS.md` §31 item 1.6)**:
     - In `src/LFM3/calibrate_lfm3_int8.py`, generate per-subgraph `table.json` files where:
       - `model_info.input_data_type = ["fp16"]` and `model_info.output_data_type = ["fp16"]` (so `EsAAC` inserts hardware `CvtPrecision` nodes at subgraph boundaries, keeping all `DmaBuffer` interfaces in FP16 and avoiding manual host INT8 rescaling; if `EsAAC` emits `dtype=4` INT8 descriptors for any node, record `int8_step = threshold / 127.0` in `manifest.json` so `lfm3_v2` scales at the boundary).
       - Every internal node and blob (`input`, weights, intermediate activations) uses the calibrated threshold and `int8_step = threshold / 127.0`.
       - **`node_info[*].quantized_method` must stay `per_channel`.** `per_tensor` makes `EsAAC` SIGSEGV (exit 139) — measured on the DeepSeek expert FFN with an otherwise valid table (`FINDINGS.md` §33.2). This is a vendor-frontend limit, not a tuning knob.
     - **Size the DMA buffers from the measured surface, not the tensor.** For the INT8 DeepSeek expert FFN the graph declares `output (1,2048,1,1) INT8` = 2048 bytes but the runtime reports a **4096-byte** output surface, page-padded with zeros (`FINDINGS.md` §33.1). Allocate to the surface size and read only the tensor's bytes; decoding the padding as data produces garbage (that same padding is why an fp16 decode of the output returned NaN).
  3. Compile all 74 INT8 subgraphs with `EsAAC --input-model=<name>.onnx --quant-stats=<name>_table.json` and deploy to `roma`. Grep every log for `has no graph` — `EsAAC` exits 0 even when it compiled nothing (Milestone 3, step 2).
- **Acceptance Gate**:
  - Verify on `roma` using `tools/e2e_regression.sh`:
    - Perplexity degradation vs. FP16 baseline is $< 3\%$.
    - Autoregressive decode throughput reaches **$\ge 3.9\text{ tokens/sec}$** (`3.9–5.6 tok/s` expected).

---

### Milestone 6 (Optional Performance Milestone): 2-Way Tensor Parallelism (`TP=2`) & `Cached` DMA Upgrade

- **Goal**: Unlock both memory controllers simultaneously (`28 GB/s` aggregate) during layer execution and eliminate `NoCache` CPU read penalties (`FINDINGS.md` §21.1, §24.2, §31 item 1.1).
- **Step-by-Step Instructions**:
  1. **Empirical `ES_SYS_MemFlushCache` Directionality Probe (`FINDINGS.md` §31 item 1.1)**:
     - Run a standalone test on `roma` where the CPU reads a `Cached` `DmaBuffer` (`0xAAAA`), the NPU overwrites it via DMA (`0x5555`), and the CPU calls `ES_SYS_MemFlushCache(memFd)` before reading again. Once Clean+Invalidate semantics are verified, switch `lfm3_v2` buffers from `CacheMode::NoCache` to `CacheMode::Cached` with explicit `flush_cache()` calls before and after `submit_task()`.
  2. **2-Way Tensor Parallelism (`TP=2`) within the 62-Model/Die Limit**:
     - To keep resident loadables per die $\le 62$ when running all 24 layers on both dies, fuse each layer's operator and MLP into **2 loadables per layer per die** (`24 × 2 + 1 LM head = 49 loadables/die < 62`):
       - Split MLP `w1`/`w3` column-wise (`2048 → 3840` per die) and `w2` row-wise (`3840 → 2048` per die).
       - Split Attention by GQA groups (`8 Q : 2 KV` heads per die) and ShortConv by channels (`1024` channels per die).
- **Acceptance Gate**:
  - Sustained generation throughput on `roma` reaches **`5.5 – 8.0 tokens/sec`**, measured against the `PP=2` Mode A baseline on the same prompt, with the per-layer cross-die reduction cost reported separately. If `TP=2` does not beat `PP=2` by at least 25%, keep `PP=2` and record why — do not carry the more complex layout for a marginal gain.

---

## 6. Feasibility Analysis of Multimodal & Larger Models in the LFM3 Family

One of Liquid AI's core architectural decisions in the LFM3 generation is **"One Backbone, Six Specialists"**: every model in the LFM3 release branches directly from `LFM3-1.7B-Base` and shares the **exact same 24-layer tri-hybrid language backbone** (`hidden_size=2048`, `13 conv + 5 SWA + 6 global`).

Once Milestones 1–5 above are completed for `LFM3-1.7B`, **85–90% of the code and NPU pipeline for the entire LFM3 multimodal family is already built.** Below is the detailed feasibility analysis for each sibling model and larger MoE variants on EIC7700X.

### 6.1 Summary Matrix: LFM3 Family & Larger Liquid Models on EIC7700X

| Model | Modality | Total Params | Architecture Delta vs. `LFM3-1.7B` | INT8 MMZ Footprint | Resident Loadables / Die | Expected EIC7700X Performance | Feasibility Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[`LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B)** | Text $\to$ Text (Agent/Tools) | **1.73B** | Base 24-layer tri-hybrid backbone | `~1.73 GiB` (`0.87 GiB/die`) | **37 / 62** (49 with T-wide prefill MLPs, or 49 with full T-wide prefill if ShortConv fuses) | **3.9 – 5.6 tok/s** (`PP=2`) / **5.5 – 8.0 tok/s** (`TP=2`, unmeasured — §1.2 caveat) | **VERY HIGH (Start Here)** |
| **[`d1-2B`](https://huggingface.co/LiquidAI/d1-2B)** | Text + Image + Video $\to$ Calibrated Decision | **~1.92B** | + 164M hybrid vision encoder + 25M projector; **1-step forward pass (no decode loop!)** | `~1.95 GiB` (`0.98 GiB/die`) | **39 / 62** | **~230 – 330 ms per decision** (no *per-token* decode loop, but one full weight stream — §6.2) | **HIGH (pending the vision-encode measurement)** |
| **[`LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B)** | Text + Image + Video $\to$ Text | **1.92B** | + 164M hybrid vision encoder (14 layers: 8 conv2d + 6 attn) + 25M projector | `~1.95 GiB` (`0.98 GiB/die`) | **39 / 62** | `< 45 ms` vision encode + **3.9 – 5.6 tok/s** text decode | **VERY HIGH** |
| **[`LFM3-Audio-2B`](https://huggingface.co/LiquidAI/LFM3-Audio-2B)** | Text + Audio $\leftrightarrow$ Text + Speech | **~2.0B** | + 17-layer causal audio encoder (`d=768`) + 3-layer audio decoder + `LiquidCodec` | `~2.05 GiB` (`1.03 GiB/die`) | **40 / 62** | **High** for ASR / Audio-to-Text; **Medium** for real-time TTS | **MEDIUM-HIGH** |
| **[`LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B)** & **`d1-omni-2B`** | Text + Image + Video + Audio $\leftrightarrow$ Text + Speech | **2.20B** | Union of `LFM3-VL-2B` and `LFM3-Audio-2B` on the shared 1.7B backbone (32K ctx) | `~2.20 GiB` (`1.10 GiB/die`) | **42 / 62** | Same as VL + Audio; fits in **10%** of 22 GiB MMZ | **MEDIUM-HIGH** |
| **`LFM2.5-8B-A1B`** (MoE — **not** an LFM3 model; there is no `LFM3-8B-A1B` in the Oct 27 2026 release) | Text MoE (8.3B total / **1.2B active**) | **8.3B** | Different generation from LFM3: verify tokenizer/backbone before assuming any Milestone 1–5 code transfers | `~8.3 GiB` (INT8) or **`~16.6 GiB` (FP16)** | **unknown — must be measured** (INT8 needs one loadable per *stream class*, §35) | **7.5 – 11.0 tok/s** (INT8, expert-parallel) / **4.0 – 5.8 tok/s** (FP16) | **HIGH, but gated on the class count and on the FP16 hypothesis below** |
| **`LFM2.5-24B-A2B`** (MoE — likewise not LFM3) | Text MoE (24B total / **2.45B active**) | **24.0B** | 40 layers (30 gated short conv + 10 GQA), 64 experts top-4 | `~24.0 GiB` (requires **27 GiB U-Boot MMZ**) | Blocked if INT8 stream classes $> 62$ | `5.0 – 7.0 tok/s` *only if* stream classes $\le 60$/die | **LOW-MEDIUM (Hardware-Gated)** |

---

### 6.2 Deep-Dive on Multimodal Sibling Models (`LFM3-VL-2B`, `d1-2B`, `LFM3-Audio-2B`, `LFM3-Omni-2B`)

#### 1. `LiquidAI/LFM3-VL-2B` (1.92B) & `LiquidAI/d1-2B` (~1.92B) — **The Immediate Next Ports**
- **Architecture**: Pairs the exact 1.73B LFM3 language backbone with Liquid AI's new **164M-parameter in-house hybrid vision encoder** (`model_type: lfm3_vl_vision`, 14 layers consisting of **8 gated depthwise 2D convolution layers** and **6 bidirectional attention layers**, `num_attention_heads=16`) plus a **25M-parameter MLP projector** (`projector_hidden_size=4096`, `projector_hidden_act=gelu`) that merges every $2\times 2$ block of image/video patches into a single 2048-dim language token.
- **Why it maps cleanly to EIC7700X**:
  - Unlike heavy ViT encoders (`SigLIP-400M` or `Qwen2-VL`'s `675M` 3D-patch ViT), Liquid's 164M vision encoder uses $3\times$ less compute and relies on **2D depthwise convolutions + standard Conv2D/attention**, which are the native fixed-function sweet spot of the NVDLA CMAC/SDP hardware (`FINDINGS.md` §4 & §13.1).
  - **Staying under the 62-Model/Die Limit**: Because the vision encoder operates on fixed image tile resolutions (or can be compiled as 1–2 multi-layer `.model` loadables on Die 0 and Die 1), it adds only **1–2 resident `.model` files per die** (`37 + 2 = 39 < 62`).
  - **Special Callout for `d1-2B`**: `d1-2B` replaces autoregressive token generation with a **single forward pass** that outputs calibrated probability distributions over typed decisions (classification, OCR span extraction, object pointing, interval estimation). On `roma`, where prefill is compute-dense and autoregressive decode is memory-bandwidth-bound, `d1-2B` avoids the *per-generated-token* memory wall — but it does not avoid the memory wall. One forward pass still streams every weight once: `~1.9 GB ÷ 14 GB/s ≈ 135 ms` floor, i.e. **~230–330 ms** inside the measured 41–59% band, which is where the `~150–250 ms` estimate in §6.1 comes from and why its lower end is optimistic. Two further costs are not yet quantified and should be before this is called "EXTREMELY HIGH": the vision encoder runs over *many* patches (an image at `2×2` merge is ~1024 tokens), which is compute rather than a single-token pass; and if the decision pass is run with T=1 subgraphs it inherits the Milestone 3.5 prefill problem wholesale. `< 45 ms` vision encode is asserted without arithmetic — measure it.

#### 2. `LiquidAI/LFM3-Audio-2B` (~2.0B) & `LiquidAI/LFM3-Omni-2B` (2.2B)
- **Architecture** (verified from `LiquidAI/LFM3-Omni-2B/config.json`):
  - **Shared Language Backbone**: Identical 24-layer `lfm3` (`hidden_size=2048`, `conv_L_cache=7`, `sliding_window=256`, `use_sandwich_norm=true`).
  - **Causal Audio Encoder (`lfm3_omni_encoder`)**: 17 layers, `hidden_size=768`, `conv_channels=256`, `num_mel_bins=128`, `num_key_value_heads=4`, `sliding_window=32`, `conv_L_cache=3`, `use_sandwich_norm=false`.
  - **Audio Decoder & `LiquidCodec`**: A lightweight 3-layer transformer/conv module (`num_hidden_layers=3`, `num_attention_heads=32`, `head_dim=32`, `conv_L_cache=7`) paired with `LiquidCodec` for 24 kHz streaming audio synthesis.
- **Hardware Mapping & Tradeoffs on EIC7700X**:
  - **Audio Input (ASR, Voice Command, Spoken Tool Calling, `d1-omni-2B`)**: Highly feasible. The 128-bin Mel-spectrogram frontend runs on CPU/DSP in $< 5\text{ ms}$, and the 17-layer `d=768` causal audio encoder (`~120M` params total) uses local `k=3` convolutions and `window=32` SWA—requiring only a few milliseconds per second of audio on NPU/CPU before handing embeddings to the LFM3-1.7B backbone.
  - **Audio Output (Real-Time TTS / Interleaved Speech)**: Requires running the 3-layer audio decoder and `LiquidCodec` detokenizer at every speech frame step alongside backbone generation. While `LiquidCodec` runs at $87\times$ real-time on an Apple M4 Max thread (~12–15× real-time estimated on a 1.4 GHz SiFive P550 core), interleaved text+audio generation multiplies the number of backbone steps per second of speech. **Recommendation**: Port **Audio-In / Text-Out (ASR, Audio QA, Spoken Tool Calling, and `d1-omni-2B`) first**, and benchmark the standalone `LiquidCodec` frame rate on `roma` before committing to real-time streaming TTS.

---

### 6.3 What About Bigger Models? (The MoE tier is **LFM2.5**, not LFM3)

**Naming matters here:** the October 27, 2026 LFM3 launch is six *dense* ~2B models
(`LFM3-1.7B`, `LFM3-VL-2B`, `LFM3-Audio-2B`, `LFM3-Omni-2B`, `d1-2B`, `d1-Omni-2B`). **There is no
`LFM3-8B-A1B`.** Liquid's MoE tier is the previous generation — **[`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)** (8.3B total / 1.2B active) and **[`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)** (24B total / 2.45B active, 40 layers = 30 gated short conv + 10 GQA, 64 routed experts top-4). A "port LFM3 first, then 8B-A1B" roadmap therefore crosses a generation boundary: **verify the LFM2.5 tokenizer, layer schedule and expert geometry from its own `config.json` before assuming any Milestone 1–5 artifact transfers** (the "same 128K tokenizer as LFM3" claim below is unverified). Larger LFM3-MoE checkpoints may follow the same pattern, but plan against what is published today.

Applying our silicon-verified findings from `FINDINGS.md` §24–§35 yields a striking conclusion:

1. **Why an `8B-A1B` Liquid MoE Is Uniquely Advantaged over `DeepSeek-V2-Lite` (15.7B) on `roma`**:
   - In `FINDINGS.md` §29–§30, `DeepSeek-V2-Lite` (15.7B total) required **30.1 GiB in FP16**, forcing INT8 expert quantization (`14.2 GiB`).
   - In `FINDINGS.md` §33.2 and §34.3, INT8 per-channel quantization in `EsAAC` was found to bake expert-specific requantization shift bytes into `kmd-task-op_list` (which cannot be patched from user space), requiring one resident loadable per "stream class" against the hard **62-model/die limit** (`FINDINGS.md` §34.4–§34.5).
   - **An `8B-A1B` Liquid MoE (`~8.3B` params) completely sidesteps the INT8 stream-class problem if needed**:
     - Even in **pure FP16**, all routed experts of an 8.3B MoE occupy only **~14.0 GiB total (~7.0 GiB per die)**—which **fits inside the default 22 GiB MMZ (`11 GiB/die`) without U-Boot reconfiguration**!
     - And in **FP16** a single Class B loadable per layer (`--gen-fp16-model=yes` + `--convert-weights`) is *expected* to be byte-identical across all experts, because FP16 has no requantization step for `EsAAC` to bake per-expert shifts into. **This is a hypothesis, not a proven result.** What §27, §28.1, §29 and §32 actually proved is narrower: FP16 runtime-bound weights are bit-exact against their own baked blob, and `DmaBufferSlice` offset selection works. Nobody has compiled two *different* FP16 experts and diffed their Class B files. The INT8 result (§33.2, §34.4) shows `EsAAC` does embed weight-derived bytes in `kmd-task-op_list`, so the FP16 case must be checked rather than assumed. **Test: compile two FP16 experts of one layer, `cmp` the Class B outputs, and if they differ, bind one expert's region through the other's loadable on `roma` (`probe_two_experts --same-class` already does exactly this).** Two compiles and a diff; do it before building anything on top of the claim.
     - Furthermore, if INT8 stream classes across layers stay $\le 60$/die — measured at **15 classes for 104 real DeepSeek experts across all 26 MoE layers, extrapolating to ~25 for 1664** (`FINDINGS.md` §35) — an `8B-A1B` MoE streams only **~1.2 GB/token** across both dies simultaneously (`28 GB/s` expert-parallel bandwidth), achieving a calibrated **`9.5 – 13.7 tokens/sec`**. Note that class counts were measured on DeepSeek's expert geometry, not Liquid's; an `8B-A1B` port must re-run `count_stream_classes.py` on its own experts. Also budget the submit count: top-$k$ experts × layers per token is what made DeepSeek's ~318 submits/token worth measuring (§28), and the same arithmetic applies here.

2. **Why `24B-A2B` (`24B` Total / `2.45B` Active) Is High-Risk on a Single 32 GiB Board**:
   - At 24B parameters, INT8 weights alone require **~24.0 GiB**, which exceeds the default 22 GiB MMZ and leaves almost zero headroom even under the 28 GiB U-Boot MMZ carveout (`FINDINGS.md` §21.2) once 40 layers of loadables, KV caches, and Linux OS overhead are included.
   - Unless `EsAAC` / `eswin_compiler` adds native **INT4 (`W4A16` / `W4A8`) weight decompression** on the NPU or GPU, models in the 24B class should be deferred in favor of **`LFM3-1.7B`**, **`LFM3-VL-2B`**, **`d1-2B`**, and the **`8B-A1B` MoE** tier.
