# romanaccio

**romanaccio** (a Romanesco pun on *roma*) is an open-source development suite, reverse-engineering repository, and runtime library collection for the **ESWIN EIC7700X / EIC7702** SoC (found on the **DC-ROMA RISC-V AI PC** and **FML13V03 Mainboard II**).

It provides standalone runtimes, bug fixes for vendor software, comprehensive reverse-engineered hardware/firmware documentation, and architectural plans for open re-implementations of the compiler, simulator, and runtime toolchain.

---

## 1. Repository Layout

```
romanaccio/
├── README.md                         # Main repository guide and model download instructions
├── FINDINGS.md                       # Master summary of reverse engineering discoveries
├── findings/                         # Detailed standalone technical reports
│   ├── NPU_OPERATIONS.md             # Hardware engine & 372 DSP kernels catalog, GPU profile
│   ├── MODEL_FORMAT.md               # Reverse-engineered .model (NVDA FlatBuffers) specification
│   ├── SYSTEM_CAPABILITIES_AND_MODELS.md # Compute envelope, latency benchmarks, model feasibility
│   ├── CONTEXT_WINDOW_AND_MEMORY.md  # 4K-32K context scaling, KV cache math & chunked prefill
│   ├── RUNTIME_API.md                # Userspace NPU runtime API specification & multi-die caveats
│   └── CPU_NPU_PROTOCOL.md           # Kernel ioctl catalog (eic7700_npu.ko) & SMMU DMA mapping
├── etc/
│   └── TranslateGemma-4B-IT/         # Google TranslateGemma 4B translation engine (PLAN.md)
└── src/
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

### 2.1 Open ESWIN Runtime (`src/eswin_runtime/`)
A production-ready C11 and Python userspace runtime library (`libes_runtime.so`):
- **Contiguous DMA Allocator**: Zero-copy MMZ physical memory manager (`/dev/es_mem`) with automatic SMMU address mapping and CPU data cache maintenance.
- **WorkerPool NUMA Isolation**: Spawns isolated worker daemons for Die 0 (cores 0–3) and Die 1 (cores 4–7), eliminating single-process driver lock stalls (`0xa00f6046`) and kernel teardown Oops.
- **Model Package Manifest & Downloader**: Manifest parser (`manifest.json`), zero-dependency NIST SHA-256 validator, local cache manager, and `eswin-fetch` CLI.
- **Embedded Tokenizer & Sampler**: Ported from `llama.cpp` supporting Hugging Face Byte-Level BPE, TikToken, and ChatML with Issue #47 anti-looping protections (sliding-window repetition penalty, consecutive duplicate suppression, cyclic loop detection).
- **TransformerSession & KV-Cache**: Circular multi-layer KV-cache and dual-die LM Head reduction engine.
- **CLI Chat REPL & Python SDK**: Interactive streaming terminal chat CLI (`eswin-chat`) and Pythonic ctypes bindings (`eswin_runtime`).

### 2.2 Applications (v2 Powered by Open Runtime)
- **SmolLM-v2 Chat Engine (`src/smollm_v2/`)**:
  Modular C++17 and Python interactive chat engine for [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M) built on `libes_runtime.so`. Offloads the 49K-token vocabulary LM Head across both NPU dies simultaneously (~0.9 ms latency).
- **Qwen2-v2 Chat Engine (`src/qwen2_v2/`)**:
  Modular C++17 and Python interactive chat engine for Qwen2-0.5B, 1.5B, and DeepSeek-R1-Distill-Qwen models. Integrates native anti-looping samplers that permanently resolve [Framework Issue #47](https://github.com/DC-DeepComputing/Framework/issues/47).

### 2.3 Legacy Prototype Baselines (Preserved)
- `src/smollm/`: First-generation standalone C++17 prototype.
- `src/qwen2_fix/`: First-generation standalone C++17 patch.

### 2.4 Re-Implementation Architecture Plans
- **Open Compiler (`src/eswin_compiler/PLAN.md`)**:
  Incremental plan for an open-source `EsAAC` alternative. Features a hybrid frontend (ONNX for vision; native `safetensors` + `config.json` for LLMs), 4D tensor IR, NVDLA hardware engine lowering, and DSP offloading.
- **Open Simulator (`src/eswin_simulator/PLAN.md`)**:
  Incremental plan for a 64-bit cross-platform `EsSimulator` equivalent for bit-accurate numerical verification against PyTorch and ONNX golden data.

### 2.5 Modern Neural Network Acceleration Plans
- **Gemma-4-12B-it INT8 (`src/gemma-4-12B-it-Uncensored-Aquarion-INT8/PLAN.md`)**:
  Comprehensive feasibility assessment and dual-die pipelined execution plan for the 12-Billion parameter Gemma-4 model. Partitions 48 transformer layers (24 per die) and the 262k LM head across 22 GiB of MMZ VRAM with hybrid sliding-window attention management.
- **YOLO11 Real-Time Vision (`src/yolov11/PLAN.md`)**:
  Ultralytics YOLO11 (n, s, m, l, x) acceleration roadmap utilizing the hardware CMAC, SDP (SiLU LUT), PDP (SPPF MaxPool), and RUBIK engines. Supports 130–180 FPS single-die inference and dual-die dual-camera concurrent 1080p60 streaming.
- **TranslateGemma-4B-IT (`etc/TranslateGemma-4B-IT/PLAN.md`)**:
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

## 3. Large Binary Files & Model Download Instructions

To keep the git repository lightweight and compatible with GitHub hosting, large binary files (raw model weights `*.bin`, compiled NPU loadables `*.model`, and ONNX graphs) are **not stored directly in git**.

### 3.1 SmolLM-135M Model Data
The SmolLM-135M runtime requires the following model files under `src/smollm/models/`:
- `smollm_135m.bin` (514 MB): Raw FP32 weights for all 30 transformer layers.
- `tokenizer.bin` (517 KB): Clean UTF-8 BPE token vocabulary.
- `smollm_lm_head_d0_npu_b1.model` (28 MB): Die 0 NPU offline loadable (tokens 0..24575).
- `smollm_lm_head_d1_npu_b1.model` (28 MB): Die 1 NPU offline loadable (tokens 24576..49151).

#### How to Obtain / Generate Model Files:
1. **Download Upstream Model**:
   Download the original SmolLM-135M model from Hugging Face:
   - Base weights: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
   - Or ONNX model: [HuggingFaceTB/SmolLM-135M ONNX](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
2. **Export Binary Weights**:
   Place the ONNX model (e.g. `SmolLM-135M.onnx`) in your working directory and run:
   ```bash
   cd src/smollm
   python3 export_smollm.py
   ```
   This automatically generates `models/smollm_135m.bin` and `models/tokenizer.bin`.
3. **Compile Dual-Die LM Head NPU Models**:
   Compile the partitioned LM Head ONNX slices for Die 0 and Die 1 using `EsAAC`:
   ```bash
   # In your ESWIN compilation environment (or container):
   EsAAC --model=smollm_lm_head_d0.onnx --gen-fp16-model=yes --target-device=EIC7700X
   EsAAC --model=smollm_lm_head_d1.onnx --gen-fp16-model=yes --target-device=EIC7700X
   ```
   Place the resulting `.model` files into `src/smollm/models/`.

### 3.2 Qwen2-0.5B and DeepSeek-7B Models
For testing `src/qwen2_fix/`, download the pre-quantized vendor NPU packages or place the vendor models into:
- `/opt/eswin/data/npu/` (on the target board)
- `~/models/Qwen2-0.5B_ENNP/` or `~/models/deepseek-r1-distill-qwen-7b_ENNP/`
The runtime flags in `es_qwen2_demo` accept custom model paths via `--model-dir`.

---

## 4. Building the Projects

### SmolLM-135M C++ Engine
```bash
cd src/smollm
make
./smollm_chat
```

### Qwen2 / DeepSeek Fix
```bash
cd src/qwen2_fix
./build.sh
./build/es_qwen2 --system "You are a helpful coding assistant."
```
