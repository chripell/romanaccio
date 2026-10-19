# SmolLM-v2: Dual-Die NPU Interactive Chat Engine

## 1. Overview

`smollm_v2` is the second-generation interactive chat engine for **SmolLM-135M-Instruct**, built entirely on top of the modular **Open ESWIN NPU Runtime** (`libes_runtime.so`).

Unlike the first-generation prototype (`src/smollm`), which relied on monolithic memory mapping and manual driver orchestration, `smollm_v2`:
- Uses the standard **Model Package Manifest** (`manifest.json`) specifying model architecture, vocab size, hidden dimensions, and NPU offload targets.
- Leverages **WorkerPool Process Isolation** across NUMA nodes, executing Die 0 on cores 0–3 and Die 1 on cores 4–7 with zero lock contention.
- Uses the shared C11 runtime API (`es_model_*`, `es_session_*`), embedding the high-performance Tokenizer and anti-looping Sampler ported from `llama.cpp`.
- Provides both modern C++17 (`smollm_v2.cpp`) and Python 3 (`smollm_v2.py`) interactive streaming chat REPLs.

---

## 2. Model Architecture & Dual-Die Splitting

| Parameter | Specification |
| :--- | :--- |
| **Model** | HuggingFaceTB/SmolLM-135M-Instruct |
| **Architecture** | LLaMA / Transformer (30 Layers, 9 Heads, 3 KV Heads) |
| **Hidden Size / Intermediate** | 576 / 1536 |
| **Vocabulary Size** | 49,152 tokens |
| **Die 0 Offload** | LM Head vocabulary slice $[0 \dots 24,575]$ |
| **Die 1 Offload** | LM Head vocabulary slice $[24,576 \dots 49,151]$ |
| **Context Window** | 2,048 tokens (Circular KV-cache in contiguous memory) |

---

## 3. Building

### Host Build (Simulation / Fallback)
```bash
# Build runtime first
cd ../eswin_runtime && make && cd ../smollm_v2

# Build smollm_v2
make
```

### Cross-Compiling for RISC-V (Board Roma)
```bash
# Cross-compile runtime
cd ../eswin_runtime && make clean && make CROSS_COMPILE=riscv64-linux-gnu- && cd ../smollm_v2

# Cross-compile smollm_v2
make clean && make CROSS_COMPILE=riscv64-linux-gnu-
```

---

## 4. Running

### C++ Interactive REPL
```bash
# Run with default dual-die acceleration
./smollm_v2

# Custom sampling options
./smollm_v2 --temp 0.7 --top-p 0.9 --repeat-penalty 1.15 --max-tokens 256
```

### Python Interactive REPL
```bash
python3 smollm_v2.py --temp 0.7
```

### Interactive Commands
- `/reset`: Flushes conversation history and clears the KV-cache.
- `/exit` or `/quit`: Cleanly exits and shuts down NPU worker daemons.
- `Ctrl+C`: Halts active token generation gracefully without killing the REPL.
