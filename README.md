# romanaccio

**romanaccio** (a Romanesco pun on *roma*) is an open-source development suite, reverse-engineering repository, and runtime library collection for the **ESWIN EIC7700X / EIC7702** SoC (found on the **DC-ROMA RISC-V AI PC** and **FML13V03 Mainboard II**).

It provides standalone runtimes, bug fixes for vendor software, comprehensive reverse-engineered hardware/firmware documentation, and architectural plans for open re-implementations of the compiler, simulator, and runtime toolchain.

---

## 1. Repository Layout

```
romanaccio/
├── README.md                         # Main repository architecture and component index
├── FINDINGS.md                       # Master summary of reverse engineering discoveries (§1–§49)
├── findings/                         # Detailed standalone technical reports
│   ├── NPU_OPERATIONS.md             # Hardware engine & 372 DSP kernels catalog, GPU profile
│   ├── MODEL_FORMAT.md               # Reverse-engineered .model (NVDA FlatBuffers) specification
│   ├── SYSTEM_CAPABILITIES_AND_MODELS.md # Compute envelope, latency benchmarks, model feasibility
│   ├── CONTEXT_WINDOW_AND_MEMORY.md  # 4K-32K context scaling, KV cache math & chunked prefill
│   ├── RUNTIME_API.md                # Userspace NPU runtime API specification & multi-die caveats
│   └── CPU_NPU_PROTOCOL.md           # Kernel ioctl catalog (eic7700_npu.ko) & SMMU DMA mapping
└── src/
    ├── LFM3/                         # Liquid AI LFM3 tri-hybrid dual-die engine, LoRA, d1-2B & vision
    ├── DeepSeek-V2-Lite/             # DeepSeek-V2-Lite MoE dual-die partitioning & expert loadables
    ├── eswin_compiler/               # Open EsAAC compiler equivalent (PLAN.md)
    ├── eswin_simulator/              # Open EsSimulator functional simulator equivalent (PLAN.md)
    ├── eswin_runtime/                # Open C11/C++ NPU runtime, CLI chat REPL & Python bindings
    ├── smollm/                       # SmolLM-135M chat engine (v1 prototype)
    ├── smollm_v2/                    # SmolLM-135M chat engine (v2 using Open Runtime & dual-die NPU)
    ├── qwen2_fix/                    # Root cause fixes for Issue #47 (v1 standalone prototype)
    ├── qwen2_v2/                     # Qwen2 / DeepSeek chat engine (v2 using Open Runtime & anti-looping)
    ├── gemma-4-12B-it-Uncensored-Aquarion-INT8/ # Gemma-4 12B INT8 dual-die inference (PLAN.md)
    ├── yolov11/                      # Ultralytics YOLO11 dual-NPU real-time vision (PLAN.md)
    ├── TranslateGemma-4B-IT/         # TranslateGemma 4B machine translation (PLAN.md)
    ├── whisper/                      # OpenAI Whisper edge speech-to-text (PLAN.md)
    ├── qwen2_vl/                     # Qwen2-VL-2B multimodal vision-language model (PLAN.md)
    ├── qwen2_5_coder/                # Qwen2.5-Coder-1.5B/7B high-speed copilot (PLAN.md)
    └── rpg_uncensored/               # Stheno, Mistral-Nemo & Qwen2.5 uncensored RPG engine (PLAN.md)
```

---

## 2. Components Overview

### 2.1 Liquid AI LFM3 Tri-Hybrid Engine & Multimodal Suite (`src/LFM3/`)
Full hardware-accelerated dual-die C++17 runtime (`lfm3_v2`) for Liquid AI's LFM3 tri-hybrid architecture (gated ShortConv, Sliding Window Attention, and Global Attention):
- **Silicon Performance**: Verified **`9.02 tok/s` sustained decode** (`112.6 ms/tok`) and **`55.5–72.1 tok/s` batched INT8 prefill** on the dual-die EIC7700X (`PP=2`, layers 0–11 on Die 0, 12–23 on Die 1).
- **Route D Class B Breakthrough**: Reuses 8 shared archetype loadables + 8 LM head slices = **only 12 resident models/die** (far below the 62-model kernel Oops ceiling, zero per-turn swapping).
- **Zero-Recompile INT8 Quantization**: Direct NVDLA float64 tiler + SmoothQuant channel balancing (`in_inv_smooth[c]`) + per-output-channel `CvtPrecision_QuantM` scaling (`out_step_chan[c]`), eliminating denominator-anchor sink collapse with zero `.model` recompiles.
- **Multimodal & Decision Ecosystem**:
  - `LFM3-1.7B-Heretic`: Uncensored fine-tuned LoRA merged in FP32 and deployed in INT8.
  - `d1-2B`: Single-forward-pass zero-decode decision foundation engine (711–865 ms latency).
  - `LFM3-VL-2B`: 164M hybrid vision encoder + Layer 0 visual embedding injection via `/image` CLI.
  - Resident daemon on `roma:8088` and OpenAI-compatible REST server (`src/LFM3/lfm3_openai_server.py`) on `host:8000` with streaming SSE.
- **Documentation & Instructions**: See [`src/LFM3/README.md`](src/LFM3/README.md) for full compilation, packaging, and execution instructions.

### 2.2 Open ESWIN Runtime (`src/eswin_runtime/`)
A production-ready C11 and Python userspace runtime library (`libes_runtime.so`):
- **Contiguous DMA Allocator**: Zero-copy MMZ physical memory manager (`/dev/es_mem`) with automatic SMMU address mapping and CPU data cache maintenance.
- **WorkerPool NUMA Isolation**: Spawns isolated worker daemons for Die 0 (cores 0–3) and Die 1 (cores 4–7), eliminating single-process driver lock stalls (`0xa00f6046`) and kernel teardown Oops.
- **Model Package Manifest & Downloader**: Manifest parser (`manifest.json`), zero-dependency NIST SHA-256 validator, local cache manager, and `eswin-fetch` CLI.
- **Embedded Tokenizer & Sampler**: Ported from `llama.cpp` supporting Hugging Face Byte-Level BPE, TikToken, and ChatML with Issue #47 anti-looping protections (sliding-window repetition penalty, consecutive duplicate suppression, cyclic loop detection).
- **TransformerSession & KV-Cache**: Circular multi-layer KV-cache and dual-die LM Head reduction engine.
- **CLI Chat REPL & Python SDK**: Interactive streaming terminal chat CLI (`eswin-chat`) and Pythonic ctypes bindings (`eswin_runtime`).
- **Documentation & Instructions**: See [`src/eswin_runtime/README.md`](src/eswin_runtime/README.md).

### 2.3 SmolLM-135M Runtimes (`src/smollm/`, `src/smollm_v2/`)
- **SmolLM-v2 Chat Engine (`src/smollm_v2/`)**:
  Modular C++17 and Python interactive chat engine for [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M) built on `libes_runtime.so`. Offloads the 49K-token vocabulary LM Head across both NPU dies simultaneously (~0.9 ms latency). See [`src/smollm_v2/README.md`](src/smollm_v2/README.md).
- **SmolLM-v1 Prototype (`src/smollm/`)**:
  First-generation standalone C++17 chat engine with dual-die LM head offload. See [`src/smollm/README.md`](src/smollm/README.md).

### 2.4 Qwen2 / DeepSeek Issue #47 Resolution (`src/qwen2_fix/`, `src/qwen2_v2/`)
- **Qwen2-v2 Chat Engine (`src/qwen2_v2/`)**:
  Modular C++17 and Python interactive chat engine for Qwen2-0.5B, 1.5B, and DeepSeek-R1-Distill-Qwen models. Integrates native anti-looping samplers that permanently resolve [Framework Issue #47](https://github.com/DC-DeepComputing/Framework/issues/47). See [`src/qwen2_v2/README.md`](src/qwen2_v2/README.md).
- **Qwen2-Fix Standalone Patch (`src/qwen2_fix/`)**:
  Standalone C++17 patch and interactive chat REPL resolving repetition traps, SIGSEGV faults, and multi-die deadlocks. See [`src/qwen2_fix/README.md`](src/qwen2_fix/README.md).

### 2.5 Re-Implementation Architecture Plans
- **Open Compiler (`src/eswin_compiler/PLAN.md`)**:
  Incremental plan for an open-source `EsAAC` alternative. Features a hybrid frontend (ONNX for vision; native `safetensors` + `config.json` for LLMs), 4D tensor IR, NVDLA hardware engine lowering, and DSP offloading.
- **Open Simulator (`src/eswin_simulator/PLAN.md`)**:
  Incremental plan for a 64-bit cross-platform `EsSimulator` equivalent for bit-accurate numerical verification against PyTorch and ONNX golden data.

### 2.6 Modern Neural Network Acceleration Plans
- **Gemma-4-12B-it INT8 (`src/gemma-4-12B-it-Uncensored-Aquarion-INT8/PLAN.md`)**:
  Comprehensive feasibility assessment and dual-die pipelined execution plan for the 12-Billion parameter Gemma-4 model. Partitions 48 transformer layers (24 per die) and the 262k LM head across 22 GiB of MMZ VRAM with hybrid sliding-window attention management.
- **YOLO11 Real-Time Vision (`src/yolov11/PLAN.md`)**:
  Ultralytics YOLO11 (n, s, m, l, x) acceleration roadmap utilizing the hardware CMAC, SDP (SiLU LUT), PDP (SPPF MaxPool), and RUBIK engines. Supports 130–180 FPS single-die inference and dual-die dual-camera concurrent 1080p60 streaming.
- **TranslateGemma-4B-IT (`src/TranslateGemma-4B-IT/PLAN.md`)**:
  High-accuracy offline translation engine covering 55 languages. Ingests Google LiteRT (`.task` / `.tflite`) and Hugging Face weights, achieving 5.0–9.0 tokens/second on the dual-die pipeline.
- **OpenAI Whisper Speech-to-Text (`src/whisper/PLAN.md`)**:
  Privacy-preserving, fully offline voice transcription and dictation. Maps 1D audio convolutions and encoder attention directly to hardware, reaching >15x real-time transcription speeds.
- **Qwen2-VL-2B Multimodal Assistant (`src/qwen2_vl/PLAN.md`)**:
  Vision-Language model pairing SigLIP / ViT patch convolutions on Die 0 with our proven Qwen2 dual-die LLM engine for image analysis, visual document OCR, and UI reasoning.
- **Qwen2.5-Coder-1.5B & DeepSeek-R1-Distill-1.5B (`src/qwen2_5_coder/PLAN.md`)**:
  High-speed terminal copilot and interactive coding engine generating at 35–45 tokens/second with 16K–32K context support and chunked prefill.
- **Uncensored RPG & Creative Writing Engine (`src/rpg_uncensored/PLAN.md`)**:
  Dedicated roleplay stack supporting `Sao10K/L3.1-8B-Stheno-v3.3`, `Mistral-Nemo-12B-HERETIC`, and `Qwen2.5-7B-Instruct-abliterated`. Features 16K context window buffers, 4096-token narrative outputs, and an embedded OpenAI-compatible API server for direct connection to SillyTavern.

---

> [!TIP]
> **Model-Specific Build & Execution Guides**:
> Detailed instructions on downloading weights, compiling loadables, and running live inference on hardware are maintained in each model's dedicated directory:
> - **LFM3 Family (`LFM3-1.7B`, `Heretic`, `d1-2B`, `LFM3-VL-2B`)**: [`src/LFM3/README.md`](src/LFM3/README.md)
> - **SmolLM-135M**: [`src/smollm/README.md`](src/smollm/README.md) and [`src/smollm_v2/README.md`](src/smollm_v2/README.md)
> - **Qwen2 / DeepSeek**: [`src/qwen2_fix/README.md`](src/qwen2_fix/README.md) and [`src/qwen2_v2/README.md`](src/qwen2_v2/README.md)
> - **Open ESWIN Runtime (`libes_runtime.so`)**: [`src/eswin_runtime/README.md`](src/eswin_runtime/README.md)

