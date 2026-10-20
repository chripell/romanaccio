# SmolLM-135M Interactive Chat Engine & Dual-Die NPU Acceleration

This directory contains the complete end-to-end standalone runtime for **SmolLM-135M** on the **ESWIN EIC7700X / EIC7702 (ROMA RISC-V AI PC / FML13V03)**.

It provides both a zero-overhead native **C++17 runtime** (`smollm_chat`) and a reference **Python runtime** (`smollm_chat.py`), featuring interactive streaming chat, Grouped-Query Attention (GQA), RoPE, SwiGLU FFN, KV caching, and Dual-Die NPU acceleration.

---

## 1. Directory Structure

```
src/smollm/
├── Makefile                          # High-performance C++ build script (OpenMP + dlopen)
├── README.md                         # This runtime guide
├── SMOLLM_GUIDE.md                   # Complete compilation & dual-die execution manual
├── export_smollm.py                  # Converts ONNX weights & tokenizer to binary format
├── smollm_chat.cpp                   # Native C++17 chat engine with dual-die NPU offload
├── smollm_chat.py                    # Python interactive chat engine reference
└── models/
    ├── smollm_135m.bin               # 514 MB FP32 weights (all 30 layers)
    ├── tokenizer.bin                 # 517 KB clean UTF-8 BPE vocabulary (49,152 tokens)
    ├── tokenizer.json                # Hugging Face tokenizer configuration
    ├── config.json                   # SmolLM-135M model architecture hyperparameters
    ├── smollm_lm_head_d0_npu_b1.model# Die 0 NPU LM Head binary (Tokens 0..24575)
    └── smollm_lm_head_d1_npu_b1.model# Die 1 NPU LM Head binary (Tokens 24576..49151)
```

---

## 2. Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Size** | 135 Million Parameters | Causal LM based on LLaMA architecture |
| **Hidden Dimension ($d_{model}$)** | 576 | Vector dimension per token |
| **Transformer Layers** | 30 | Full transformer blocks |
| **Attention Query Heads** | 9 | Head dimension = 64 |
| **Attention KV Heads** | 3 | Grouped-Query Attention (3:1 ratio) |
| **MLP Intermediate Dimension** | 1536 | SwiGLU activation ($\text{Down}(\text{SiLU}(\text{Gate}) \odot \text{Up})$) |
| **Vocabulary Size** | 49,152 | Byte-level BPE with tied word embeddings |
| **Context Window** | 2048 | Dynamic RoPE positional embeddings |

---

## 3. Fast Start: Building & Running

### 3.1 Native C++ Engine (Recommended)

#### Build
```bash
cd src/smollm
make
```
*Compiles with `-O3 -std=c++17 -fopenmp -ldl` with zero external dependencies.*

#### Interactive Chat
```bash
./smollm_chat
```

#### Custom Options
```bash
./smollm_chat \
  --model models/smollm_135m.bin \
  --tokenizer models/tokenizer.bin \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.15 \
  --max-tokens 256
```

#### Command-Line Arguments
- `--model <path>`: Path to binary weights (default: `models/smollm_135m.bin`).
- `--tokenizer <path>`: Path to binary tokenizer (default: `models/tokenizer.bin`).
- `--cpu`: Force CPU backend (disables probing for `/dev/npu0` and `/dev/npu1`).
- `--temp <float>`: Temperature scaling (default: `0.7`, `0` = greedy argmax).
- `--top-k <int>`: Top-K vocabulary filtering (default: `40`).
- `--top-p <float>`: Top-P nucleus sampling (default: `0.9`).
- `--repeat-penalty <float>`: Sliding-window repetition penalty (default: `1.15`).
- `--max-tokens <int>`: Maximum tokens generated per prompt (default: `256`).

---

### 3.2 Python Reference Runtime

```bash
python3 smollm_chat.py
```
*(Requires `numpy`. Supports both interactive REPL and single-prompt completion).*

---

## 4. Hardware Acceleration & Dual-Die NPU Offload

### 4.1 How Dual-Die Offload Works
The SmolLM-135M vocabulary has 49,152 tokens. In the LM Head projection ($576 \to 49152$), the computation is partitioned across both SoC dies:
- **Die 0 (`/dev/npu0`)**: Computes logits for vocabulary slice `0 .. 24575` using `smollm_lm_head_d0_npu_b1.model`.
- **Die 1 (`/dev/npu1`)**: Computes logits for vocabulary slice `24576 .. 49151` using `smollm_lm_head_d1_npu_b1.model`.

Both dies execute concurrently in under **1.1 ms**, offloading 49,152 matrix multiplications from the CPU cores on every generated token.

### 4.2 Engineering Architecture & Key Discoveries
1. **Thread-Level Device Isolation**: The vendor EDLA runtime (`libedla_runtime.so.1.0`) binds device state to the calling thread. Calling `ES_NPU_SetDevice` for both Die 0 and Die 1 on a single thread returns error `0xa00f6046` (`ES_NPU_ERROR_DEV_MULTI_SET`). `smollm_chat` spawns dedicated, NUMA-pinned worker threads (`DieWorker`): Die 0 is bound to CPU cores 0–3, and Die 1 to CPU cores 4–7.
2. **Half-Precision (FP16) DMA Contract**: The compiled `.model` kernels expect FP16 activations. The input tensor is shaped `[1, 576, 1, 1]` (1,152 bytes) and each output tensor is `[1, 24576, 1, 1]` (49,152 bytes). Native `_Float16` conversions are performed directly on the RISC-V cores.
3. **Cache Invalidation (`ES_SYS_MemFlushCache`)**: Memory allocated via `/lib/libes_memory.so.1.0` in cached mode requires flushing CPU L1/L2 lines before NPU DMA execution and invalidating caches before host CPU reading.
4. **Contiguous Row-Major SIMD Optimization**: The 30-layer Transformer weights are transposed once at load time to ensure stride-1 sequential memory access for all inner products. Combined with preallocated attention scratch buffers, generation speed reaches **~11–13.5 tokens/s** (a 12× speedup over unaligned column traversals).

### 4.3 Automatic Fallback
The C++ runtime uses `dlopen` to detect `/lib/libedla_runtime.so.1.0` and `/lib/libes_memory.so.1.0` at runtime:
- **On RISC-V board (`roma`)**: If dual NPUs are available, it automatically maps unified shared memory (`mmz_nid_0_part_0` and `mmz_nid_1_part_0`) and enables dual-die offload.
- **On host x86_64 or without NPUs**: It automatically falls back to an OpenMP-parallelized CPU projection without requiring separate builds or libraries.

---

## 5. Issue #47 Repetition Fix & Generation Quality

In earlier vendor runtimes (Issue #47), LLMs on the EIC7700X exhibited infinite repetition loops (e.g. repeated punctuation or looping phrases) followed by buffer overflows.

In `smollm_chat`, this is completely resolved through a two-tiered mitigation:
1. **Sliding-Window Repetition Penalty**: Penalizes tokens generated within the last 64 positions by a scaling factor ($1.15\times$).
2. **Consecutive Duplicate Break Detector**: Immediately aborts generation if 4 identical consecutive tokens are generated, preventing runaway loops.

---

## 6. Obtaining Models & Weights

Due to GitHub file size limits, large binary weights (`*.bin`) and compiled NPU loadables (`*.model`) are excluded from git.

### Download & Export from Upstream
1. Download the SmolLM-135M ONNX model from Hugging Face:
   - [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
2. Export weights and vocabulary into `models/`:
   ```bash
   python3 export_smollm.py
   ```
   This reads your downloaded ONNX model (e.g. `SmolLM-135M.onnx`) and produces:
   - `models/smollm_135m.bin` (514 MB)
   - `models/tokenizer.bin` (517 KB)
3. For NPU LM Head offloading, compile the dual-die models using `EsAAC` as described in [SMOLLM_GUIDE.md](SMOLLM_GUIDE.md).

