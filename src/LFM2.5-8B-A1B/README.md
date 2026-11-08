# LiquidAI LFM2.5-8B-A1B Sparse MoE on the ESWIN EIC7700X (Dual-Die)

Native C++17 engine ([`lfm2_moe`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp)) for `LiquidAI/LFM2.5-8B-A1B` (**8.47B** total parameters, **~1.42B** active parameters/token): 18 gated short convolutions (`K=3`), 6 global GQA layers (`32` Q / `8` KV heads, `d_head=64`, `θ=5,000,000`), 2 dense stem SwiGLU layers (L0–L1, `TP=2` via 4 quarter-experts), and 22 sparse MoE layers (L2–L23, **32 experts, Top-4 routing**, `EP=2` with 16 even experts on Die 0 and 16 odd experts on Die 1).

Design details live in [`PLAN.md`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/PLAN.md) and silicon measurements live in [`../../FINDINGS.md`](file:///home/chri/aixigua/romanaccio/FINDINGS.md#L2878-L2954) (§52).

---

## 1. Silicon Verification & Performance Summary on `roma`

| Metric / Workload | Measured Result on `roma` | Notes |
|---|---|---|
| **MMZ Weight Allocation** | **`5.166 GiB × 2` (`3` chunks/die `< 2.0 GiB`)** | Works around `eswin_rsvmem_heap_mmap` 4.0 GiB kernel truncation bug; **`8.78 GiB` Linux OS RAM free** |
| **Resident Loadables / Die** | **`12 / 62`** (`16 / 62` with `--speculative-pld`) | `[PASS]` (`< 55` safety gate; kernel ceiling `62`) |
| **Min Layer Cosine (All 24 Layers)** | **`0.95180935`** | `[PASS]` (`>= 0.9500`) |
| **128K Final Logits Cosine vs. FP32** | **`0.99521100`** | `[PASS]` (`>= 0.9950`) |
| **Greedy 4-Token Match** | **`4 / 4 exact`** (`' Paris'`, `'.'`, `' city'`) | `[PASS]` (`DETERMINISM` + `SENSITIVITY` + `RESTORE`) |
| **Short-Context Decode (`15`–`64` tok)** | **`11.25 – 11.30 tok/s` (`88.5 – 90.3 ms/tok`)** | `exp_mlp`: `35.92 GB/s` driver; `LM Head`: `55.44 GB/s` driver |
| **332-Token C-Refactoring (`160` tok)** | **Prefill: `33.46 – 34.15 tok/s` (`29.3 ms/tok`)**, **Decode (`T=1`): `10.39 tok/s` (`96.9 ms/tok`)** | `--no-think` masks `<think>` (`124901`) for instant code generation |

---

## 2. Common Invocations on `roma`

```bash
ssh roma
cd /home/chri/models/LFM2.5-8B-A1B_ENNP

# 1. One-shot CLI prompt (with --no-think for immediate response)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --no-think -n 64 --temp 0 \
  --prompt "What is the capital of France and what is it famous for?"

# 2. Prompt from file (e.g., C code refactoring)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --no-think -n 160 --temp 0 --prompt-file /tmp/prompt_c.txt

# 3. Interactive terminal chat REPL (type '/exit' to quit, '/nothink' to toggle thinking)
sudo LD_LIBRARY_PATH=. ./lfm2_moe

# 4. Golden Oracle verification & acceptance gate (determinism, sensitivity, restore, 4/4 greedy, 15-tok speed)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --verify-golden golden_ref.bin

# 5. Optional Prompt-Lookup Speculative Decoding (T=8 batch verification)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --speculative-pld --no-think -n 160 --temp 0 --prompt-file /tmp/prompt_c.txt
```

---

## 3. CLI Flags Reference

| Flag / Option | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | `""` | One-shot user prompt text (`@FILE` loads prompt from file). |
| `--prompt-file FILE` | `-f` | `""` | Path to text/source file to use as prompt. |
| `--no-think` | | `false` | Disable `<think>` reasoning block and mask token `124901` (`<think>`). |
| `--max-tokens N` | `-n` | `128` | Maximum new tokens to decode autoregressively. |
| `--max-ctx N` | | `4096` | Context window size in tokens. |
| `--model-dir DIR` | `-m` | `.` | Package directory containing `cpu_weights.bin`, `weights_die*.bin`, `embedding.bin`. |
| `--system TEXT` | | `"You are a..."` | Custom system prompt for ChatML template. |
| `--temp F` | | `0.3` | Sampling temperature (`0` for greedy argmax). |
| `--top-k K` | | `50` | Top-k vocabulary sampling filter. |
| `--repeat-penalty F` | | `1.05` | Sliding-window repetition penalty. |
| `--speculative-pld` | `--pld` | `false` | Enable `T=8` batch verification with Prompt-Lookup Drafting (`--pld-k K`, default `K=3`). |
| `--verify-golden FILE` | | `""` | Run golden numerical verification, gate-integrity checks, and 15-token decode benchmark. |

---

## 4. Building & Packaging from Scratch (Host + `simulator` Container)

```bash
# 1. Inspect safetensors & generate golden reference
/home/chri/aixigua/venv/bin/python /home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/inspect_lfm2_moe.py
/home/chri/aixigua/venv/bin/python /home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/inspect_and_golden.py

# 2. Export ONNX archetypes, compile with EsAAC, calibrate SmoothQuant, and pack dual-die MMZ blobs
/home/chri/aixigua/venv/bin/python /home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/build_lfm2_moe_pkg.py

# 3. Cross-compile libes_runtime.so and lfm2_moe for RISC-V 64
make -C /home/chri/aixigua/romanaccio/src/eswin_runtime CROSS_COMPILE=riscv64-linux-gnu- -j8
make -C /home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B CXX=riscv64-linux-gnu-g++ -j8
```
