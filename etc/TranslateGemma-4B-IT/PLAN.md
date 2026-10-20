# TranslateGemma-4B-IT INT8 Dual-NPU Acceleration Plan on ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **VERY HIGH (Optimal Fit for Edge Translation)**
Deploying [`litert-community/TranslateGemma-4B-IT`](https://huggingface.co/litert-community/TranslateGemma-4B-IT) on the ESWIN EIC7700X hardware is **highly feasible, mathematically balanced, and directly actionable**.

TranslateGemma-4B-IT is Google's state-of-the-art open machine translation model built upon the Gemma foundation architecture. It is specifically instruction-tuned to deliver high-quality bi-directional translation across **55 natural languages** (including high-resource languages like Spanish, Chinese, French, German, Italian, as well as mid- and low-resource languages).

In INT8 quantization, the complete model occupies **~4.2 GiB**. With **22 GiB of contiguous hardware MMZ VRAM** available on the board (11 GiB on Die 0 and 11 GiB on Die 1), TranslateGemma-4B-IT fits with abundant headroom:
- **Die 0**: Layers 0–15 (~2.1 GiB) + 128,000-token LM Head slice (~0.33 GiB) = **~2.43 GiB committed** out of 11 GiB MMZ.
- **Die 1**: Layers 16–31 (~2.1 GiB) + 128,000-token LM Head slice (~0.33 GiB) = **~2.43 GiB committed** out of 11 GiB MMZ.
- **MMZ Free Margin**: **>8.5 GiB free per die**, easily supporting extensive KV-caches up to 4096-token context windows.

### 1.2 Performance & Latency Projections

| Metric | Specification |
| :--- | :--- |
| **Model Size** | 4.0 Billion Parameters (Dense Decoder-Only) |
| **Weight Quantization** | INT8 per-channel / symmetric |
| **Total Model Footprint**| **~4.2 GiB** (MMZ hardware VRAM) |
| **Hardware Offload Mode**| Concurrent Dual-Die Pipelined Streaming (Die 0 + Die 1) |
| **VRAM Consumption** | **~2.43 GiB on Die 0**, **~2.43 GiB on Die 1** |
| **Memory Bandwidth Demand**| ~4.2 GB of weights streamed per generated token |
| **Achieved DMA Bandwidth** | ~25–30 GB/s sustained per die |
| **Expected Generation Speed**| **5.0 – 9.0 tokens/second** |
| **Sentence Translation Latency**| **~3.5 – 6.0 seconds** for a standard 30-word sentence (~40 tokens) |
| **Supported Languages** | 55 languages with high BLEU / COMET translation quality |

---

## 2. Model Architecture & LiteRT Ingestion Pipeline

### 2.1 Model Architecture Parameters

| Parameter | Value | Hardware Implementation Notes |
| :--- | :--- | :--- |
| **Hidden Dimension ($d_{model}$)** | 2560 | Multiple of 64; 40 systolic channels on NPU. |
| **Intermediate Size ($d_{ffn}$)** | 10240 | $4\times d_{model}$; Gate & Up projections $2560 \to 10240$. |
| **Number of Layers** | 32 | Symmetrically partitioned: 16 on Die 0, 16 on Die 1. |
| **Attention Query Heads** | 16 | $16 \times 256 = 4096$ dimension. |
| **Key/Value Heads (GQA)** | 8 | Grouped Query Attention ($8 \times 256 = 2048$ dim). |
| **Head Dimension** | 256 | Standard modern head dimension. |
| **Vocabulary Size** | 256,000 | Multilingual SentencePiece vocabulary. |
| **Activation Function** | `gelu_pytorch_tanh` | Mapped to 256-entry hardware SDP Look-Up Table. |
| **Normalization** | Gemma RMSNorm | $(1.0 + \gamma) \cdot \frac{x}{\text{RMS}(x)}$. Scale $+1.0$ folded at compile time. |

---

### 2.2 Unpacking Google LiteRT `.task` Bundle

The Hugging Face repository provides `translategemma-4b-it-int8-web.task`. This archive is a Google MediaPipe / LiteRT zip container:

```
translategemma-4b-it-int8-web.task (ZIP Archive)
├── model.tflite              # FlatBuffers file containing INT8 tensor buffers & quantization scales
├── tokenizer.model           # Google SentencePiece binary vocabulary (256,000 tokens)
└── metadata.json             # Model card, prompt formatting templates & supported ISO codes
```

#### Extraction Workflow
1. **Unzip `.task`**:
   ```bash
   unzip -q translategemma-4b-it-int8-web.task -d ./extracted_task/
   ```
2. **Inspect TFLite Tensors with FlatBuffers**:
   Using `tflite.Model.GetRootAsModel()`, parse tensor buffers:
   - All projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are stored as signed 8-bit integers (`int8`) with associated float scale vectors.
   - Token embedding table ($256000 \times 2560$) is stored in FP16 or INT8 format.
3. **Alternative Safetensors Ingestion**:
   If license acceptance or Hugging Face access to `google/translategemma-4b-it` is configured, raw weights can also be ingested directly from standard `.safetensors` using `eswin_compiler`.

---

## 3. Dual-Die Hardware Memory Layout & Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                                TRANSLATEGEMMA-4B DUAL-DIE MEMORY LAYOUT                           |
|                                                                                                   |
|  [ NUMA Node 0 / Die 0 ]                                           [ NUMA Node 1 / Die 1 ]        |
|  +--------------------------------------------+    D2D Interconnect   +---------------------------+
|  | SiFive P550 4-Core CPU @ 1.4 GHz          |<======================>| SiFive P550 4-Core CPU    |
|  | MMZ Pool 0: 11 GiB (VRAM)                  |   (Hardware Pipeline) | MMZ Pool 1: 11 GiB (VRAM) |
|  |                                            |                       |                           |
|  | - Layers 0 to 15 (.model blocks):  2.10 GiB|                       | - Layers 16 to 31: 2.10GB |
|  | - LM Head Slice 0 (0..127999):     0.33 GiB|                       | - LM Head Slice 1: 0.33GB |
|  | - Static KV Cache (2048 ctx):      0.26 GiB|                       | - Static KV Cache: 0.26GB |
|  | - Ping-Pong DMA Activation Scratch:0.04 GiB|                       | - Ping-Pong DMA:   0.04GB |
|  | ------------------------------------------ |                       | ------------------------- |
|  | TOTAL MMZ COMMITTED: 2.73 GiB / 11.00 GiB  |                       | TOTAL MMZ: 2.73 / 11.00GB |
|  | FREE MMZ HEADROOM:   8.27 GiB              |                       | FREE MMZ:   8.27 GiB      |
|  +--------------------------------------------+                       +---------------------------+
|                                                                                                   |
|  [ Host Linux OS System RAM (9.4 GiB Total across NUMA 0 & 1) ]                                   |
|  - Multilingual Embedding Table (256,000 x 2560 @ FP16 = 1.31 GiB) -> Memory mapped zero-copy    |
|  - SentencePiece Multilingual Tokenizer (256k tokens) -> ~15 MiB RAM                              |
|  - Dual-Die Pipeline Coordinator & CLI Interface -> ~80 MiB RAM                                   |
|  - Free Linux Host RAM: ~7.8 GiB                                                                 |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Prompt Engineering & Chat Template for Machine Translation

TranslateGemma-4B-IT uses the standardized Gemma instruction format with explicit source and target language specifications:

```
<start_of_turn>user
Translate the following text from {source_language} to {target_language}:
"{input_text}"<end_of_turn>
<start_of_turn>model
```

### Supported Languages (55 Languages)
High-resource: English (`en`), Chinese (`zh`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Japanese (`ja`), Korean (`ko`), Portuguese (`pt`), Russian (`ru`), Arabic (`ar`).
Mid- and low-resource: Polish, Dutch, Turkish, Vietnamese, Hindi, Swedish, Ukrainian, Czech, Greek, Indonesian, Romanian, Hungarian, Danish, Finnish, Norwegian, and 25+ more.

---

## 5. End-to-End Implementation Roadmap & Milestones

### Phase 1: Artifact Extraction & Ingestion Tool (Days 1–3)
- [ ] Implement `extract_translategemma.py` to unpack `translategemma-4b-it-int8-web.task`.
- [ ] Parse `.tflite` FlatBuffers tensor definitions, extracting INT8 weights and per-channel scale factors.
- [ ] Extract and verify `tokenizer.model` using the SentencePiece C++ / Python API.
- [ ] Partition layers $0\dots 15$ for Die 0 and $16\dots 31$ for Die 1.
- [ ] Split LM Head into two $128,000 \times 2560$ halves.

### Phase 2: Compiler Lowering & SDP LUT Generation (Days 4–6)
- [ ] Ingest extracted tensor shapes into `eswin_compiler`.
- [ ] Program the 256-entry SDP Look-Up Table for `gelu_pytorch_tanh` ($[-6.0, +6.0]$).
- [ ] Fold $(1.0 + \gamma)$ into RMSNorm weights.
- [ ] Compile 32 layer submodels (`layer_00.model` to `layer_31.model`) and 2 LM head submodels (`lm_head_die0.model`, `lm_head_die1.model`).

### Phase 3: Simulator Functional Verification (Days 7–8)
- [ ] Run test translation sentences through `eswin_simulator` on x86_64 host.
- [ ] Verify that generated token IDs match Google's LiteRT reference output.
- [ ] Validate dual-die KV cache rolling state across multi-turn translation requests.

### Phase 4: RISC-V C++ Runtime Implementation (`translategemma_cli`) (Days 9–12)
- [ ] Implement `translategemma_cli` in C++17 under `romanaccio/src/TranslateGemma-4B-IT/`:
  - Native SentencePiece C++ tokenizer binding.
  - Zero-copy mmap embedding lookup.
  - Thread-isolated `DieWorker`s executing on NUMA nodes 0 and 1.
  - Interactive REPL and batch translation file mode (`--source en --target it --input input.txt`).
- [ ] Add lightweight REST / JSON API server (`--serve 8080`) for local network translation services.

### Phase 5: Hardware Benchmark & Performance Validation (Days 13–14)
- [ ] Deploy compiled models and binary to `roma`.
- [ ] Measure translation throughput (tokens/second) and latency across multiple language pairs.
- [ ] Verify translation quality against standard WMT benchmark sentences.
- [ ] Commit all code, documentation, and user guides to `romanaccio`.
