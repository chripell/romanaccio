# Uncensored RPG & Creative Writing Engine (Stheno, Mistral-Nemo, Qwen2.5) Dual-NPU Plan

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **EXTREMELY HIGH (Matches 22 GiB MMZ Hardware Profile)**
Running uncensored, abliterated, and Heretic-processed Large Language Models for **interactive roleplay (RPG), storytelling, and creative fiction** on the ESWIN EIC7700X is **fully supported and uniquely suited** to the dual-die 22 GiB hardware configuration.

Roleplay and creative writing have distinct technical requirements compared to basic question-answering:
1. **Uncensored / Abliterated Weights**: Guardrails and safety refusals are mathematically ablated via directional vector removal (Heretic / abliteration), ensuring the model never breaks character, refuses dark fantasy themes, or outputs generic moralizing lectures.
2. **Extended Context Window (8K – 16K Tokens)**: Roleplay sessions require maintaining complex **Character Cards**, world lorebooks, personality traits, and dozens of turns of conversational memory.
3. **Large Output Budgets (2048 – 4096 Tokens)**: Story responses require narrative room for dialogue, descriptive action, and world-building without arbitrary cutoff.
4. **Standard Frontend Compatibility**: Native integration with the premier open-source roleplay frontend, **SillyTavern**, via an embedded local OpenAI-compatible API (`/v1/chat/completions`).

### 1.2 Model Evaluation & Comparison for EIC7700X Dual-Die

| Model Candidate | Parameter Count | Weight Format | INT8 Size | Dual-Die Offload | 16K Context KV Cache | Generation Speed | RPG Style & Strengths |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Sao10K/L3.1-8B-Stheno-v3.3`**<br>(or `Pranavz/L3-8B-Stheno-v3.2-heretic`) | 8.03 B | INT8 Safetensors | **~8.03 GiB** | 4.01 GiB / die | **2.10 GiB** | **~9.5 – 11.5 tokens/s** | **The Gold Standard for RPG**. Incredible character fidelity, organic human dialogue, nuanced emotional tracking. |
| **`huihui-ai/Qwen2.5-7B-Instruct-abliterated`** | 7.61 B | INT8 Safetensors | **~7.60 GiB** | 3.80 GiB / die | **0.92 GiB** | **~10.5 – 12.0 tokens/s** | **Drop-in Compatible** with our existing `qwen2_fix` engine. Razor-sharp instruction adherence, high speed. |
| **`DavidAU/Mistral-Nemo-12B-HERETIC`**<br>(or `TheDrummer/Rocionante-12B-v1.1`) | 12.2 B | INT8 Safetensors | **~12.20 GiB**| 6.10 GiB / die | **2.62 GiB** | **~1.5 – 2.2 tokens/s** | **Masterpiece Literary Prose**. Unsurpassed atmospheric writing, multi-character tracking, complex plotting. |

---

## 2. Hardware Memory Budget with 16K Context Window

With **22 GiB of contiguous hardware MMZ VRAM** (11 GiB on Die 0 + 11 GiB on Die 1), all three models run with full 16,384-token context buffers completely inside NPU hardware memory:

```
+---------------------------------------------------------------------------------------------------+
|                     16K CONTEXT VRAM CONSUMPTION BREAKDOWN (22 GiB MMZ TOTAL)                     |
|                                                                                                   |
|  [ Model A: L3.1-8B-Stheno-v3.3 (8B) ]                                                            |
|  • Weights (INT8 split 50/50): 8.03 GiB  (4.01 GiB per die)                                      |
|  • 16K KV Cache (FP16 split):  2.10 GiB  (1.05 GiB per die)                                      |
|  • Scratch / Ping-Pong:        0.10 GiB  (0.05 GiB per die)                                      |
|  • Total MMZ Used:            10.23 GiB / 22.00 GiB (46.5% Committed — 11.77 GiB Free Headroom)   |
|                                                                                                   |
|  [ Model B: Qwen2.5-7B-Instruct-Abliterated (7.6B) ]                                              |
|  • Weights (INT8 split 50/50): 7.60 GiB  (3.80 GiB per die)                                      |
|  • 16K KV Cache (FP16 split):  0.92 GiB  (0.46 GiB per die)                                      |
|  • Scratch / Ping-Pong:        0.10 GiB  (0.05 GiB per die)                                      |
|  • Total MMZ Used:             8.62 GiB / 22.00 GiB (39.2% Committed — 13.38 GiB Free Headroom)   |
|                                                                                                   |
|  [ Model C: Mistral-Nemo-12B-HERETIC (12.2B) ]                                                    |
|  • Weights (INT8 split 50/50): 12.20 GiB (6.10 GiB per die)                                      |
|  • 16K KV Cache (FP16 split):  2.62 GiB  (1.31 GiB per die)                                      |
|  • Scratch / Ping-Pong:        0.12 GiB  (0.06 GiB per die)                                      |
|  • Total MMZ Used:            14.94 GiB / 22.00 GiB (67.9% Committed — 7.06 GiB Free Headroom)    |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. SillyTavern & RPG Frontend Architecture

To make RPG interaction seamless, the runtime provides an embedded C++ HTTP server supporting the standard OpenAI API specification:

```
[ User Device (Browser / Tablet / PC / Laptop) ]
                    │
                    ▼ (Local Network WiFi / LAN)
[ SillyTavern Frontend (http://localhost:8000 or http://roma.local:8000) ]
  • Character Card Parsing (Tavern v2 PNG metadata)
  • Dynamic World Lorebook / Memory Book Ingestion
  • Author's Note & Scenario Framing
                    │
                    ▼ (HTTP POST /v1/chat/completions)
[ Roma EIC7700X: Embedded C++ RPG API Server (roma-rpg-server) ]
  ├── Chunked Prefill (1024-token tiles) for lorebooks & scenario cards
  ├── Dual-Die NPU Streaming Pipeline (Die 0 + Die 1)
  ├── Advanced RPG Samplers:
  │     • Min-P Sampling (min_p = 0.05 - 0.10)
  │     • Dynamic Temperature (temp = 0.85 - 1.10)
  │     • Repetition Penalty (rep_pen = 1.12, win = 128)
  │     • Anti-Stutter & Loop Suppressor (Framework Issue #47 fix)
  └── SSE (Server-Sent Events) Real-Time Token Streaming
                    │
                    ▼
[ Streaming Prose: "The tavern was dimly lit by flickering tallow candles..." ]
```

---

## 4. Implementation Roadmap & Milestones

### Phase 1: Model Acquisition & Ingestion (Days 1–3)
- [ ] Acquire `Sao10K/L3.1-8B-Stheno-v3.3` (or `Pranavz/L3-8B-Stheno-v3.2-heretic`) and `huihui-ai/Qwen2.5-7B-Instruct-abliterated`.
- [ ] Verify refusal rate: Run benchmark probes ("Write a fictional heist plan", "Describe a gruesome combat scene") to confirm ablation is clean and zero refusals occur.
- [ ] Extract weights and partition into Die 0 and Die 1 blocks.

### Phase 2: Compiler Lowering & 16K KV Cache Configuration (Days 4–6)
- [ ] Ingest Llama-3.1 and Qwen2.5 graphs into `eswin_compiler` / `EsAAC`.
- [ ] Configure RoPE parameters:
  - Llama 3.1: $\theta = 500,000.0$, scale factor $8.0$ (supporting up to 128k).
  - Qwen 2.5: $\theta = 1,000,000.0$.
- [ ] Compile `.model` blocks with support for dynamic KV cache strides up to $16,384$ tokens.

### Phase 3: Simulator Functional Validation (Days 7–8)
- [ ] Verify 16K attention accuracy and chunked prefill logic in `eswin_simulator`.
- [ ] Validate Min-P sampling and repetition penalties.

### Phase 4: Embedded API Server & SillyTavern Integration (Days 9–12)
- [ ] Implement `roma-rpg-server` in C++17 under `romanaccio/src/rpg_uncensored/`:
  - Embedded micro-HTTP server (`httplib`).
  - Endpoint `/v1/models` and `/v1/chat/completions`.
  - SSE streaming chunks (`data: {"choices":[{"delta":{"content":"..."}}]}`).
- [ ] Connect SillyTavern and test with rich Character Cards and 8,000-token multi-turn chats.

### Phase 5: Hardware Benchmark & Tuning (Days 13–14)
- [ ] Deploy to `roma` and benchmark generation throughput at 1K, 4K, 8K, and 16K context lengths.
- [ ] Verify that long context generation maintains consistency without looping or memory leakage.
- [ ] Document setup instructions in `README.md`.
