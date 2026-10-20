# Gemma-4-12B INT8 Hardware Acceleration Plan on Dual-Die ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **FEASIBLE (with Pipelined Dual-Die Offload)**
Running [`Rin247/gemma-4-12B-it-Uncensored-Aquarion-INT8`](https://huggingface.co/Rin247/gemma-4-12B-it-Uncensored-Aquarion-INT8) on the ESWIN EIC7700X hardware is **technically feasible** and physically fits within the board's dedicated video memory (MMZ).

The model's INT8 quantized weights occupy **12.13 GiB** (`model.safetensors` = 13,019,639,056 bytes). The board features **32 GiB LPDDR5-6400 physical memory**, with the Linux kernel configured to allocate **22 GiB of contiguous hardware MMZ VRAM** (11 GiB on NUMA Node 0 / Die 0 and 11 GiB on NUMA Node 1 / Die 1), alongside ~9.4 GiB of Linux host OS RAM.

By splitting the 48 transformer layers symmetrically across the two dies (24 layers on Die 0 and 24 layers on Die 1) and partitioning the massive 262,144-token LM Head projection across both dies, the model runs entirely in hardware NPU memory without thrashing host RAM or swapping over PCIe.

### 1.2 Performance & Throughput Projections

| Metric | Target Specification |
| :--- | :--- |
| **Model Size** | 12.0 Billion Parameters (Dense Decoder-Only) |
| **Weight Format** | INT8 per-channel / per-tensor (`safetensors`) |
| **Total Weight Footprint** | **12.13 GiB** (MMZ hardware VRAM) |
| **Die 0 Weight Allocation** | **6.07 GiB** (Layers 0–23 + 131,072-token LM Head slice) |
| **Die 1 Weight Allocation** | **6.07 GiB** (Layers 24–47 + 131,072-token LM Head slice) |
| **MMZ VRAM Headroom** | **~4.93 GiB free per die** (for KV-cache & scratch buffers) |
| **Theoretical Peak Compute**| 19.99 TOPS INT8 (Dual-Die aggregate) |
| **Memory Bandwidth Limit** | LPDDR5-6400 dual-channel (~51.2 GB/s per die, ~102.4 GB/s peak) |
| **Achieved Read Bandwidth** | ~20–30 GB/s sustained DMA streaming per die |
| **Expected Token Rate** | **1.2 – 2.5 tokens/second** (Full NPU streaming pipeline) |
| **Fallback Token Rate** | **0.4 – 0.8 tokens/second** (if RMSNorm/Softmax runs on Host CPU) |

> [!NOTE]
> In autoregressive generation (batch size = 1), throughput is strictly **memory bandwidth bound**. Streaming 12.13 GB of weights from LPDDR5 to the NPU systolic arrays at an effective memory read speed of ~25 GB/s yields a mathematical ceiling of $\approx 2.06 \text{ tokens/second}$. This speed is viable for offline document analysis, deep uncensored reasoning, and agentic workflows.

---

## 2. Hardware Architecture & Memory Allocation

The EIC7700X SoC consists of two symmetric compute dies interconnected via a high-speed die-to-die (D2D) bus:

```
+---------------------------------------------------------------------------------------------------+
|                                 ESWIN EIC7700X DUAL-DIE ARCHITECTURE                             |
|                                                                                                   |
|  [ NUMA Node 0 / Die 0 ]                                           [ NUMA Node 1 / Die 1 ]        |
|  +--------------------------------------------+    D2D Interconnect   +---------------------------+
|  | SiFive P550 4-Core CPU @ 1.4 GHz          |<======================>| SiFive P550 4-Core CPU    |
|  | MMZ Pool 0: 11 GiB (VRAM)                  |   (Hardware Pipeline) | MMZ Pool 1: 11 GiB (VRAM) |
|  |                                            |                       |                           |
|  | - Layers 0 to 23 (.model blocks): 5.56 GiB |                       | - Layers 24 to 47: 5.56GB |
|  | - LM Head Slice 0 (0..131071):       0.51GB|                       | - LM Head Slice 1: 0.51GB |
|  | - Ring KV Cache (1024 window):       0.38GB|                       | - Ring KV Cache:   0.38GB |
|  | - Global KV Cache (2048 ctx):        0.19GB|                       | - Global KV Cache: 0.19GB |
|  | - Activation Ping-Pong Scratch:      0.08GB|                       | - Activation Ping-Pong:   |
|  | ------------------------------------------ |                       | ------------------------- |
|  | TOTAL MMZ COMMITTED: 6.72 GiB / 11.00 GiB  |                       | TOTAL MMZ: 6.72 / 11.00GB |
|  | FREE MMZ HEADROOM:   4.28 GiB              |                       | FREE MMZ:   4.28 GiB      |
|  +--------------------------------------------+                       +---------------------------+
|                                                                                                   |
|  [ Host Linux OS System RAM (9.4 GiB Total across NUMA 0 & 1) ]                                   |
|  - Embedding Table (262,144 x 3840 @ FP16 = 2.01 GiB) -> Memory mapped zero-copy (mmap)          |
|  - SentencePiece / TikToken Vocabulary Table (262k entries) -> ~18 MiB RAM                        |
|  - Host Coordination Engine & Thread-Isolated Die Workers -> ~150 MiB RAM                         |
|  - Linux Kernel & OS buffers -> ~1.50 GiB RAM                                                     |
|  - Available System RAM Reserve: ~5.7 GiB                                                         |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Gemma-4 Architecture Analysis & NPU Lowering Challenges

Gemma-4 presents several distinctive architectural features that differ from standard LLaMA/Qwen architectures:

### 3.1 Architectural Parameters

| Parameter | Value | Hardware Mapping Consideration |
| :--- | :--- | :--- |
| **Layers** | 48 | Partitioned $24 + 24$ across Die 0 and Die 1. |
| **Hidden Dimension ($d_{model}$)** | 3840 | Multiples of 64; optimal for 64-channel NPU systolic MAC tiles. |
| **Intermediate Size ($d_{ffn}$)** | 15360 | $4\times d_{model}$. Gate & Up projections require $3840 \to 15360$. |
| **Attention Query Heads** | 16 | $16 \times 256 = 4096$ total query dim. |
| **Key/Value Heads (GQA)** | 8 | $8 \times 256 = 2048$ total KV dim (Grouped Query Attention). |
| **Head Dimension** | 256 | Standard layers have $head\_dim = 256$. |
| **Hybrid Attention Layers** | 40 Sliding / 8 Global | Every 6th layer (5, 11, 17, 23, 29, 35, 41, 47) is a **Global Attention** layer with $head\_dim = 512$ and $num\_kv\_heads = 1$! |
| **Activation Function** | `gelu_pytorch_tanh` | GeLU approximation: $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$. |
| **Normalization** | Gemma RMSNorm | Normalized with $(1.0 + \text{weight}) \times \frac{x}{\text{RMS}(x)}$. |
| **Logit Softcapping** | 30.0 | $\text{logits} = 30.0 \times \tanh(\text{logits} / 30.0)$ before softmax. |
| **Vocabulary Size** | 262,144 | Massive vocabulary: LM Head weights = $262144 \times 3840 = 1.00 \text{ GiB}$. |

---

### 3.2 Key Technical Hurdles & Solutions

#### Challenge A: Massive Vocabulary & Split LM Head (262,144 tokens)
- **Problem**: Projecting the final hidden state ($1 \times 3840$) to $262,144$ logits requires a $1.00 \text{ GiB}$ INT8 matrix multiplication. A single FP16 output tensor is $524 \text{ KB}$. A single NPU die submodel loading 1 GB for one operator creates memory imbalance.
- **Solution**: Symmetrically partition the LM Head matrix into two submodels:
  - `lm_head_die0.model`: Computes logits $[0 \dots 131071]$.
  - `lm_head_die1.model`: Computes logits $[131072 \dots 262143]$.
  - Both dies execute their half in parallel using their respective CMAC arrays. The host CPU receives two $262 \text{ KB}$ FP16 vectors, applies the $30.0 \times \tanh(\text{logits} / 30.0)$ softcapping in vector registers, and performs Top-P/Top-K sampling.

#### Challenge B: Hybrid Attention & Alternating Layer Topologies
- **Problem**: Gemma-4 does not have uniform layers. 40 layers use **Sliding Window Attention** (local window of 1024 tokens) with 8 KV heads ($head\_dim=256$). Meanwhile, 8 layers (indices 5, 11, 17, 23, 29, 35, 41, 47) use **Full Global Attention** with a single shared KV head ($head\_dim=512$, $attention\_k\_eq\_v = true$).
- **Solution**:
  - Maintain two distinct KV-cache ring buffers:
    1. `SlidingKVCache`: 40 layers $\times$ 1024 positions $\times$ 8 heads $\times$ 256 dim $\times$ 2 bytes (FP16) = $320 \text{ MiB}$ per die. Fixed circular ring buffer; past tokens beyond 1024 roll over automatically.
    2. `GlobalKVCache`: 8 layers $\times$ max 2048 positions $\times$ 1 head $\times$ 512 dim $\times$ 2 bytes (FP16) = $32 \text{ MiB}$ per die. Linear monotonically growing cache.
  - Compile two separate submodel archetypes: `gemma_sliding_block.model` and `gemma_global_block.model`.

#### Challenge C: GeLU Pytorch Tanh & Gemma RMSNorm
- **Problem**: Standard LLaMA uses SwiGLU ($\text{swish}(x) \times y$) and standard RMSNorm ($\gamma \cdot \frac{x}{\text{RMS}(x)}$). Gemma adds $+1.0$ to norm weights and uses GeLU Tanh in the MLP.
- **Solution**:
  - **RMSNorm**: Gemma's scale factor is $(1.0 + \gamma)$. During compilation, fold the $+1.0$ offset directly into the static weight tensor before quantization: $\hat{\gamma} = 1.0 + \gamma$. This makes Gemma RMSNorm bit-compatible with the hardware SDP / DSP normalizer!
  - **GeLU Tanh**: The NPU's SDP (Single Data Point) engine contains a 256-entry piecewise-linear Look-Up Table (LUT). The compiler programs the SDP LUT with the sampled curve of $y = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ over the dynamic range $[-6.0, +6.0]$, executing the non-linear activation in hardware at full systolic throughput.

### 3.3 Context Window Scaling (4K–16K), Chunked Prefill & Output Size Decoupling
- **Sliding Window Capped Memory**: The 40 sliding-attention layers are strictly bounded to a 1024-token rolling buffer. As context grows to 4,096, 8,192, or 16,384 tokens, the sliding layers **never exceed 320 MiB** of KV cache.
- **Global Attention Scaling**: Only the 8 global attention layers grow linearly with context length ($16.38 \text{ KiB/token}$). At 16K context, global KV cache consumes only $256 \text{ MiB}$ total.
- **Chunked Prefill**: Long prompts (e.g. 8K or 16K documents) are ingested in **1024-token chunks**, preventing $O(N^2)$ SPRAM memory spikes and bounding peak intermediate activation buffers to $<32 \text{ MiB}$.
- **Decoupled Output Ceiling**: Replaces the legacy 1024-token limit with a **4096-token generation ceiling** (`--max-tokens 4096`), ensuring complete, untruncated narrative responses. Autoregressive token generation appends 1 token per step without requiring static preallocation.

---

## 4. Simulator-First Development Strategy

Compiling and deploying a 48-layer 12-Billion parameter model directly on physical hardware without a functional simulator is counterproductive:
1. Driver faults on the board corrupt kernel MMZ driver state, requiring 60-90 second physical reboots.
2. The proprietary vendor `EsSimulator` is an outdated 32-bit x86 binary that crashes when presented with multi-head hybrid attention or large-dimension submodels.
3. Quantization noise (INT8 clipping) on 48 stacked transformer layers can cause logits to diverge into gibberish if scales are not verified layer-by-layer.

### 4.1 Development Workflow with `eswin_simulator`

```
  [ Hugging Face safetensors & config.json ]
                     │
                     ▼
  [ eswin_compiler: Generate .model FlatBuffers & quant_stats.json ]
                     │
                     ▼
  [ STEP 1: Bit-Accurate Functional Simulation (eswin_simulator) ]
    ├── Execute .model submodel on x86_64 host (NumPy / C++ accelerator)
    ├── Run PyTorch Transformers golden forward pass side-by-side
    ├── Layer-by-layer differential check:
    │     Cosine Similarity > 0.995?  Max Abs Error < 0.05?
    │     ├── NO  ──> Auto-tune INT8 clipping scales in compiler
    │     └── YES ──> Proceed to next block
    └── Verify Dual-Die KV-Cache memory indexing & sliding window logic
                     │
                     ▼
  [ STEP 2: Hardware Deployment on Roma Board ]
    ├── Upload verified .model files to /home/chri/models/gemma4_12b/
    ├── Stream execution through dual-die pipeline
    └── Guaranteed first-time driver stability & coherent output!
```

---

## 5. Implementation Roadmap & Milestones

### Phase 1: Model Extraction & Weight Partitioning (Days 1–3)
- [ ] Implement `extract_gemma4_weights.py` to ingest `model.safetensors` from `Rin247/gemma-4-12B-it-Uncensored-Aquarion-INT8`.
- [ ] Extract per-tensor FP16 / INT8 weights and scales into structured chunks:
  - `embedding.bin`: Token embeddings ($262144 \times 3840$, FP16, $2.01 \text{ GiB}$).
  - `layers_0_to_23.bin`: Die 0 layer weights ($5.56 \text{ GiB}$).
  - `layers_24_to_47.bin`: Die 1 layer weights ($5.56 \text{ GiB}$).
  - `lm_head_split.bin`: Split into Die 0 and Die 1 halves ($503 \text{ MiB}$ each).
- [ ] Fold $(1.0 + \gamma)$ into RMSNorm weights.

### Phase 2: Compiler Lowering & SDP LUT Generation (Days 4–7)
- [ ] Extend `eswin_compiler` to support Gemma-4 hybrid attention blocks:
  - Generate `gemma_sliding_block.model` (40 layers).
  - Generate `gemma_global_block.model` (8 layers).
- [ ] Calibrate and generate the 256-entry SDP Look-Up Table for `gelu_pytorch_tanh`.
- [ ] Generate FlatBuffers loadables for `lm_head_die0.model` and `lm_head_die1.model`.

### Phase 3: Simulator Verification (`eswin_simulator`) (Days 8–11)
- [ ] Ingest Gemma-4 submodels into `eswin_simulator`.
- [ ] Compare simulated NPU layer outputs against PyTorch `transformers` golden output tensors on the host.
- [ ] Verify that cosine similarity across all 48 layers remains above $0.992$.
- [ ] Validate sliding window cyclic KV-cache indexing (1024 token eviction).

### Phase 4: Dual-Die Runtime Engine (`gemma4_chat`) (Days 12–16)
- [ ] Implement `gemma4_chat` runtime in C++17 under `romanaccio/src/gemma-4-12B-it-Uncensored-Aquarion-INT8/`:
  - `DieWorker` thread 0 pinned to NUMA Node 0 (NPU Die 0).
  - `DieWorker` thread 1 pinned to NUMA Node 1 (NPU Die 1).
  - Synchronous D2D pipeline queue: Die 0 passes layer 23 hidden state to Die 1 via shared DMA buffer.
- [ ] Integrate SentencePiece tokenizer with 262k vocabulary.
- [ ] Implement logit softcapping ($30.0 \times \tanh(\text{logits}/30.0)$) and sampling.

### Phase 5: Hardware Benchmark & Tuning (Days 17–19)
- [ ] Deploy compiled models and runtime to `roma`.
- [ ] Benchmark generation speed, memory stability, and thermal headroom.
- [ ] Document final performance numbers and user guide.
