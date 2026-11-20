# LiquidAI LFM2.5-8B-A1B Sparse MoE on the ESWIN EIC7700X (Dual-Die)

Native C++17 engine ([`lfm2_moe`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp)) for `LiquidAI/LFM2.5-8B-A1B` (**8.47B** total parameters, **~1.42B** active parameters/token): 18 gated short convolutions (`K=3`), 6 global GQA layers (`32` Q / `8` KV heads, `d_head=64`, `θ=5,000,000`), 2 dense stem SwiGLU layers (L0–L1, `TP=2` via 4 quarter-experts), and 22 sparse MoE layers (L2–L23, **32 experts, Top-4 routing**, `EP=2` with 16 even experts on Die 0 and 16 odd experts on Die 1).

Design details live in [`PLAN.md`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/PLAN.md) and silicon measurements live in [`../../FINDINGS.md`](file:///home/chri/aixigua/romanaccio/FINDINGS.md#L2878-L3100) (§52).

---

## 1. Silicon Verification & Performance Summary on `roma`

| Metric / Workload | Measured Result on `roma` | Notes |
|---|---|---|
| **MMZ Weight Allocation** | **`5.166 GiB × 2` (`3` chunks/die `< 2.0 GiB`)** | Works around `eswin_rsvmem_heap_mmap` 4.0 GiB kernel truncation bug; **`8.78 GiB` Linux OS RAM free** |
| **Resident Loadables / Die** | **`12 / 62`** (`16 / 62` with `--speculative-pld`) | `[PASS]` (`< 55` safety gate; kernel ceiling `62`) |
| **Boundary Precision (`Ops / Exp / Head`)** | **`INT16 / INT16 / INT16` outputs (`INT8` weights)** | Eliminates all `dyntok` INT8 output & 128K LM Head vocab clipping (Milestone 16) |
| **Min Layer Cosine (All 24 Layers)** | **`0.97924192`** (`L00=0.999548`, `L23=0.986258`) | `[PASS]` (`>= 0.9490`) |
| **128K Final Logits Cosine vs. FP32** | **`0.99837211`** (`embedding_norm = 0.975337`) | `[PASS]` (`>= 0.9900`) |
| **Greedy 4-Token Match** | **`4 / 4 exact`** (`' Paris'`, `'.'`, `' city'`) | `[PASS]` (`DETERMINISM` + `SENSITIVITY` + `RESTORE`) |
| **Golden Decode (`15` tok, `--verify-golden`)** | **`11.38 tok/s` (`87.91 ms/tok`)** (`12.67` pre-`kT8SafeSlots`) | Gate `>= 8.00 tok/s` `[PASS]`; `exp_mlp`: `40.07 GB/s` drv; `LM Head`: `62.00 GB/s` drv |
| **Phase F Prefix Cache (`--verify-prefix-cache`, `3,818` tok)** | **`5 / 5 PASS`, `max_abs=0.000000`, `cos=1.00000000`, `4/4` greedy** | `restore=19.14 ms` (`8K=41.53 ms <= 45 ms`), `71.7x` 4K TTFT (`101.03s -> 1.41s`), Tier-0 `0.00 ms` (§52.8) |
| **Unseen Python Code Prompts (`roma`)** | **`11.02–11.94 tok/s`** (`10.64 tok/s` `--speculative-pld`) | `--no-think` & `--think-level low` generate verified executable Python (`ZERO setDevice errors`) |
| **332-Token C-Refactoring (`160` tok)** | **Prefill: `33.46 – 34.15 tok/s` (`29.3 ms/tok`)**, **Decode (`T=1`): `10.39 tok/s` (`96.9 ms/tok`)** | Pre-M16 332-token benchmark; `--no-think` masks `<think>` (`124901`) for instant code generation |

---

## 2. Common Invocations on `roma`

```bash
ssh roma
cd /home/chri/models/LFM2.5-8B-A1B_ENNP

# 1. One-shot CLI prompt (defaults to --think-level medium = 256 thinking tokens; pass --think-level low or --no-think)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --think-level low -n 128 --temp 0 \
  --prompt "What is the capital of France and what is it famous for?"

# 2. Prompt from file (e.g., C code refactoring)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --no-think -n 160 --temp 0 --prompt-file /tmp/prompt_c.txt

# 3. Interactive terminal chat REPL (supports '/think <none|low|medium|high|xhigh|N>', '/nothink', '/cache', '/cache-clear', '/image <path>', '/exit')
sudo LD_LIBRARY_PATH=. ./lfm2_moe --speculative-pld

# 4. Golden Oracle & Phase F Prefix Cache verification gates
sudo LD_LIBRARY_PATH=. ./lfm2_moe --verify-golden golden_ref.bin
sudo LD_LIBRARY_PATH=. ./lfm2_moe --verify-prefix-cache ../LFM3-1.7B_ENNP/bench_c_code_4k.c

# 5. Resident TCP daemon (port 8089) + Unified OpenAI HTTP gateway (host or board)
sudo LD_LIBRARY_PATH=. ./lfm2_moe --speculative-pld --daemon 8089
# On host: /home/chri/aixigua/venv/bin/python ../LFM3/lfm3_openai_server.py --backend-host roma --moe-port 8089

# 6. Optional Prompt-Lookup Speculative Decoding (T=8 batch verification) with thinking budget
sudo LD_LIBRARY_PATH=. ./lfm2_moe --speculative-pld --think-level low -n 384 --temp 0 \
  --prompt "Can you write a Python function that returns all the palindrome substrings of a given string."
```

---

## 3. CLI Flags Reference

| Flag / Option | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | `""` | One-shot user prompt text (`@FILE` loads prompt from file). |
| `--prompt-file FILE` | `-f` | `""` | Path to text/source file to use as prompt. |
| `--no-think` | | `false` | Disable `<think>` reasoning block (`think_budget=0`) and mask token `124901` (`<think>`). |
| `--think-level LEVEL` | | `medium` | Set thinking token budget preset: `none` (`0`), `low` (`64`), `medium` (`256`, default), `high` (`512`), `xhigh` (`1536`), or `unlimited`. |
| `--think-budget N` | | `256` | Set maximum thinking tokens before closing `</think>` (`0` = off; uses soft `</think>` logit ramp at $0.75 N$ + hard cap). Thinking tokens do not count against `--max-tokens N`. |
| `--max-tokens N` | `-n` | `128` | Maximum new answer tokens to decode autoregressively. |
| `--max-ctx N` | | `4096` | Context window size in tokens. |
| `--daemon [PORT]` | `--server` | `8089` | Run as resident TCP daemon server for the unified OpenAI REST gateway. |
| `--cache-dir DIR` | | `~/.cache/romanaccio/kvcache` | Root directory for persistent NVMe `.kvb` prefix cache (`LFM3_CACHE_DIR` env override). |
| `--cache-max-gb N` | | `256` | Prefix cache NVMe disk quota in GiB (leaf-first GDSF eviction with block pinning). |
| `--no-prefix-cache` | | `false` | Disable Tier-0 RAM and Tier-1 NVMe prefix caching for uncached benchmarks. |
| `--verify-prefix-cache` | | `off` | Run Phase F prefix-cache bit-exactness (`0.000000` diff) & latency gate (`[FILE]` optional). |
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
