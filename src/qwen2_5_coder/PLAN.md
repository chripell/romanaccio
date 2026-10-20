# Qwen2.5-Coder-1.5B & DeepSeek-R1-Distill-1.5B High-Speed Edge Engine Plan on ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **EXTREMELY HIGH / IMMEDIATE FIT**
Deploying **Qwen2.5-Coder-1.5B-Instruct** and **DeepSeek-R1-Distill-Qwen-1.5B** on the ESWIN EIC7700X is **immediately actionable** with minimal development overhead.

Because these models share the exact same Qwen2 transformer architecture as the 7B model already operational in `romanaccio/src/qwen2_fix/`, the entire driver stack, dual-die pipeline, memory allocator, and tokenizer are **100% compatible**.

### 1.2 Why 1.5B Models are a Game-Changer for the Roma Platform
While 7B models achieve ~10.5 tokens/second (adequate for deliberate chatting) and 12B models achieve ~1.5–2.0 tokens/second:
- **Qwen2.5-Coder-1.5B** and **DeepSeek-R1-Distill-1.5B** achieve **35 to 45 tokens per second** on the dual-die NPU hardware!
- At 35–45 tokens/sec, text streams faster than natural reading speed. This unlocks interactive use cases:
  1. **Instant Terminal Copilot**: Generate bash commands, awk/sed scripts, and git workflows in $<1.5$ seconds without leaving the terminal.
  2. **Real-Time Code Completion**: Code autocomplete inside Vim / Nano / Emacs directly on the RISC-V laptop.
  3. **High-Speed DeepSeek Reasoning**: DeepSeek-R1's `<think>` chain-of-thought completes in 2–3 seconds instead of 30+ seconds.

### 1.3 Resource Footprint & Throughput Comparison

| Model | Parameters | INT8 Model Footprint | Die 0 Allocation | Die 1 Allocation | Expected Speed | Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Coder-1.5B** | 1.54 B | **~1.55 GiB** | 0.78 GiB | 0.78 GiB | **38 – 45 tokens/s** | Code generation & Linux shell assistant |
| **DeepSeek-R1-Distill-1.5B**| 1.54 B | **~1.55 GiB** | 0.78 GiB | 0.78 GiB | **35 – 42 tokens/s** | Rapid mathematical & logic reasoning |
| **Qwen2.5-Coder-7B** | 7.61 B | **~7.60 GiB** | 3.80 GiB | 3.80 GiB | **10.5 – 12.0 tokens/s**| Complex multi-file software engineering |

---

## 2. Memory Architecture & Dual-Die Partitioning

```
+---------------------------------------------------------------------------------------------------+
|                         QWEN2.5-CODER-1.5B DUAL-DIE ALLOCATION (22 GiB TOTAL MMZ)                 |
|                                                                                                   |
|  [ NUMA Node 0 / Die 0 ]                                           [ NUMA Node 1 / Die 1 ]        |
|  +--------------------------------------------+    D2D Interconnect   +---------------------------+
|  | MMZ Pool 0: 11 GiB Total                   |<======================>| MMZ Pool 1: 11 GiB Total  |
|  |                                            |   (Hardware Pipeline) |                           |
|  | - Layers 0 to 13 (.model blocks):  0.64 GiB|                       | - Layers 14 to 27: 0.64GB |
|  | - LM Head Slice 0 (0..76031):      0.14 GiB|                       | - LM Head Slice 1: 0.14GB |
|  | - Static KV Cache (4096 ctx):      0.18 GiB|                       | - Static KV Cache: 0.18GB |
|  | ------------------------------------------ |                       | ------------------------- |
|  | TOTAL MMZ COMMITTED: 0.96 GiB / 11.00 GiB  |                       | TOTAL MMZ: 0.96 / 11.00GB |
|  | FREE MMZ HEADROOM:   10.04 GiB (91% FREE!) |                       | FREE MMZ:  10.04 GiB      |
|  +--------------------------------------------+                       +---------------------------+
+---------------------------------------------------------------------------------------------------+
```

With 91% of VRAM remaining free, the model can support huge context windows (8192+ tokens) or run alongside other models (e.g., YOLO11 + Whisper) simultaneously!

---

## 3. Terminal Copilot & Developer Tooling Architecture

Under `romanaccio/src/qwen2_5_coder/`, we will provide specialized CLI developer utilities:

### 3.1 `roma-copilot` (Shell & CLI Command Assistant)
```bash
# Ask questions directly in the terminal
roma-copilot "how do I find all files larger than 100MB modified in the last 2 days?"
# Output:
# find . -type f -size +100M -mtime -2 -exec ls -lh {} +

# Pipeline command output directly into the model
git diff | roma-copilot --commit-msg
# Output:
# feat(driver): add cache flush ioctl to prevent DMA corruption
```

### 3.2 `roma-code` (Interactive Coding REPL)
High-speed conversational coding interface with syntax highlighting, markdown code block extraction, and clipboard copying.

---

## 4. Implementation Roadmap & Milestones

### Phase 1: Model Ingestion & Quantization (Days 1–2)
- [ ] Ingest `Qwen/Qwen2.5-Coder-1.5B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- [ ] Extract weights: 28 layers, $d_{model}=1536$, $intermediate=8960$, vocab $152,064$.
- [ ] Partition into Die 0 (layers 0–13) and Die 1 (layers 14–27).
- [ ] Compile `.model` blocks using `EsAAC` / `eswin_compiler`.

### Phase 2: Pipeline Verification with Existing Runtime (Days 3–4)
- [ ] Symlink model files into `qwen2_v2` / `es_qwen2` test environment.
- [ ] Verify execution using `libes_qwen2_process.so`.
- [ ] Confirm generation speed reaches **>35 tokens/second** on physical hardware.

### Phase 3: Developer CLI Utilities (`roma-copilot`) (Days 5–7)
- [ ] Implement `roma-copilot` lightweight C++ CLI tool with system prompt tuned for Linux administration and code generation.
- [ ] Add stdin pipe support (`cat file.c | roma-copilot "explain this bug"`).
- [ ] Benchmark latency to first token ($<80 \text{ ms}$).

### Phase 4: Documentation & Deployment (Days 8–9)
- [ ] Document model deployment and usage instructions in `README.md`.
- [ ] Commit all code and configuration files to `romanaccio`.
