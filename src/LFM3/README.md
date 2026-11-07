# LFM3-1.7B on the ESWIN EIC7700X (dual-die)

Native C++17 engine (`lfm3_v2`) for LiquidAI's LFM3-1.7B tri-hybrid backbone: 13 gated short
convolutions, 5 sliding-window attentions (256) and 6 global attentions across 24 layers, run
layer-pipelined over both NPU dies (`PP=2`, layers 0-11 on die 0, 12-23 on die 1) with a
tensor-parallel LM head split 8 x 16,000.

Design and milestones live in `PLAN.md`. Measurements live in `../../FINDINGS.md` (§36-§47).
This file is only "how to run what is already built".

## 1. Which build to run today

| package | accuracy gate | decode | prefill (41 / 286 tok) | use it for |
|---|---|---|---|---|
| **FP16** (`LFM3-1.7B_FP16`) | **PASSES** -- logits cosine 0.99999111 vs PyTorch FP32, min-layer 0.99998251, 4/4 greedy (§43.1, §47) | **6.19 tok/s** (`161.5 ms/tok`) | ~30 tok/s | highest-precision reference |
| **INT8** (`LFM3-1.7B_ENNP`) | **PASSES** -- logits cosine **0.99594256** vs PyTorch FP32, min-layer **0.98892771**, **4/4 greedy** (`[17479, 20, 309, 358]`, §46.5, §47.4) | **8.96 tok/s** (`111.6 ms/tok`) | **55.5 / 72.1 tok/s** (`18.0 / 13.9 ms/tok`) | fast, reduced-memory (`0.88 GiB/die`) INT8 deployment |
| mixed, FP16 `proj2048` (`LFM3-1.7B_MIXED`) | FAILS -- min-layer 0.857, logits 0.816 (pre-§46.5 ablation artifact, §43.2) | ~5.1 tok/s | -- | historical ablation only |

Both **FP16** (`LFM3-1.7B_FP16`) and **INT8** (`LFM3-1.7B_ENNP`, after the §46.5 SmoothQuant +
per-channel `CvtPrecision_QuantM` + static LM head fix and §47 decode/prefill optimizations) pass all
golden accuracy, gate-integrity (determinism, sensitivity, restore), and 4/4 greedy decode checks on `roma`.

## 2. Run it

Everything runs natively on `roma` (`ssh roma`), and the NPU character devices (`/dev/eic7700_npu*`, `/dev/es_mem`) require root privileges (`sudo`).

### 2.1 Common Invocations by Use Case

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP   # Or LFM3-1.7B_FP16 (omit --int8 for FP16)

# 1. One-shot command line prompt
sudo ./lfm3_v2 --int8 --prompt "Explain depthwise convolution in one sentence." -n 64

# 2. Prompt from file (codebase analysis / benchmarking)
sudo ./lfm3_v2 --int8 -f bench_c_code_32k.c --no-think -n 4096 --max-ctx 36864

# 3. Interactive terminal chat REPL (supports /image <path>, /nothink, /exit)
sudo ./lfm3_v2 --int8

# 4. Resident background daemon (listening on port 8088 for OpenAI server or d1_runner)
sudo ./lfm3_v2 --int8 --daemon 8088

# 5. Full acceptance gate (golden numerical verification & determinism tests)
sudo ./lfm3_v2 --verify-golden golden_ref.bin

# 6. Global attention kernel micro-benchmark
sudo ./lfm3_v2 --bench-attn
```

**`--model-dir` matters, and silently.** The engine resolves its package from the **current working directory** (falling back to well-known board paths, and failing loudly if it finds no `cpu_weights.bin`). It prints the resolved absolute path and, per die, the weight blob's path with an FNV checksum of its first and last 64 KiB. Always check those lines before recording performance numbers.

---

### 2.2 CLI Flags & Environment Variables Reference

| Flag / Option | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | `""` | One-shot user prompt text (starts with `@FILE` to load from file). |
| `--prompt-file FILE` | `-f` | `""` | Path to text/source file to use as prompt (ideal for multi-KB code/benchmarks). |
| `--int8` | | `false` | Enable INT8/INT16 boundary scaling for quantized packages (`LFM3-1.7B_ENNP`, `Heretic`). **Do not use with FP16.** |
| `--no-think` | | `false` | Disable internal `<think>` loops, forcing instant generation of the direct answer. |
| `--max-tokens N` | `-n` | `128` | Maximum new tokens to decode autoregressively. |
| `--max-ctx N` | | `4096` | Context window size in tokens (supports up to `32768`–`40960` for 32K prompts). |
| `--daemon [PORT]` | `--server` | `8088` | Run as resident socket daemon for the OpenAI REST server or `d1_runner.py`. |
| `--model-dir DIR` | `-m` | `.` | Directory containing `cpu_weights.bin`, `weights_die*.bin`, `embedding.bin`. |
| `--system TEXT` | | `"You are a..."` | Custom system prompt for ChatML template. |
| `--temp F` | | `0.1` | Sampling temperature ($F \ge 0.0$). |
| `--top-k K` | | `50` | Top-k vocabulary sampling filter. |
| `--repeat-penalty F`| | `1.1` | Sliding-window repetition penalty. |
| `--verify-golden FILE`| | `""` | Run golden verification acceptance gate against golden tensor file. |
| `--bench-attn` | | `false` | Benchmark CPU global attention kernel throughput. |
| `--per-subgraph-scales`| | `false` | Use per-subgraph scale entries instead of shared archetype scales. |
| `--weights-layer0` | | `false` | Bandwidth timing experiment only: reuses layer 0 weights across all layers. **Outputs garbage text.** |
| `LFM3_PREFILL_T1=1` | env | `0` | Force sequential unbatched $T=1$ prefill (diagnostic only; batched $T=64$ is default). |
| `LFM3_DUMP_SURFACES=1`| env | `0` | Print detailed NPU input/output surface descriptors at load time. |
| `LFM3_MEASURE_EXTENT=1`| env | `0` | Sentinel-fill buffers to probe actual NPU byte write extents (§45). Slow. |
| `LFM3_WIDE_SURFACES=1`| env | `0` | Force legacy 2 MiB driver-bucket output buffers instead of compact geometry surfaces. |

### Gate-integrity checks (built in, not optional)

`--verify-golden` first proves the gate can fail before letting it pass anything:

1. **DETERMINISM** -- the same prefill re-run in the same process must be bit-identical.
2. **SENSITIVITY** -- `0xFF` XORed over 1 MiB at offset 0 of the resident weight blob must move
   the metrics (any difference counts; a non-finite metric counts loudest).
3. **RESTORE** -- XOR it back and the result must reproduce run 2.

If any fails, the run aborts before the decode test. This is what caught the staging-buffer race
(§42.5), and it is the reason a `FAIL` from these tests is more informative than a `PASS`.

## 3. Rebuilding a package

Host side, in this order. `deep/` holds the checkpoints and all generated artifacts.

```bash
V=/home/chri/aixigua/venv/bin/python
cd /home/chri/aixigua/romanaccio/src/LFM3

$V inspect_and_golden.py                      # config audit + golden oracle  (M1)
$V export_lfm3.py                             # 74 static 4D opset-13 ONNX + CPU weight sidecars (M2)
$V batch_compile_lfm3.py                      # EsAAC -> .model loadables + manifest (M3, container)
$V export_lfm3_prefill.py                     # T=64 prefill graphs               (M3.5)
$V calibrate_lfm3_int8.py                     # real-activation INT8 tables      (M5)
$V build_lfm3_int8_pkg.py                     # INT8 package via the NVDLA tiler (M5)
$V build_mixed_proj2048_fp16.py               # mixed FP16-proj2048 package
```

Then deploy to `roma:/home/chri/models/<name>/` and `cd` there to run. `embedding.bin` is 500 MB
and byte-identical across packages -- `md5sum` before re-sending it, the link to the board is
~5 MB/s and the blobs are ~1-3 GB.

The EsAAC container (`ssh simulator`) has two traps that cost real time: `-o` is ignored (output
lands in `<cwd>/model_files/`, or `/tmp/model_files/` if the cwd is not writable), and **it exits 0
even when it compiled nothing** -- a truncated upload reports `Model <x>.onnx has no graph` and
still returns success. Grep the logs and count output files. Uploads run at ~0.56 MB/s per stream
but scale ~8x in parallel; pulls run at ~134 MB/s.

## 4. Known limits and resolved bugs

- **INT8 accuracy root-caused and resolved on silicon (§46).** `attribute_int8_error.py` (fixed and
  run in `FINDINGS.md` §46.1) proved that per-channel INT8/INT16 weight quantization alone gives
  **min-layer `0.9973` and logits `0.9989`** (zero layers below `0.99`), whereas activation/boundary
  quantization alone dropped min-layer to **`0.7139` and logits to `0.8390`**. Specifically, three
  mechanisms caused the failure:
  1. **`conv_out` / `proj2048` Sandwich-RMSNorm sink channels + cubed input kurtosis (`min-layer
     0.6727, logits 0.5979` alone, §46.2)**: In LFM3's Sandwich-normed architecture, `conv.out_proj`
     and `self_attn.o_proj` learn 3–5 giant "denominator-anchor / sink" output channels (`10x` the
     median row norm) whose subsequent `post_operator_norm.weight` ($\gamma_k$) is `~0.0` to `1e-5`
     (`150x–2,300x` smaller than the median $\gamma_i \approx 0.024$). Per-tensor INT8 output
     quantization spent all 7 bits on those discarded sink channels while crushing the 2,043
     signal-carrying channels into `{-3..+3}` (~50% rounded to `0`), which `post_operator_norm` then
     amplified by `150x–2,300x`. Meanwhile, `conv_out`'s input $C \odot \text{conv}(B \odot x')$ has
     cubed channel outliers (`amax / rms = 30..40`), zeroing 84%–90% of input channels under INT8.
  2. **`mlp` static INT8 input & position $t=0$ (`skip_bos=True`) clipping (`min-layer 0.8801,
     logits 0.7988` alone, §46.3)**: Because `Sigmoid` is fused inside the NPU MLP graph, `mlp`
     cannot use `dyntok` and instead used a static `in_thresh = 6.88` calibrated with `skip_bos=True`
     (and `calibrate_lfm3_int8.py`'s simulator kept position $t=0$ in FP32). On hardware, position
     $t=0$ reached `|x| = 13.60` at `L05..L10` and `L23`, clipping hard by `2x` and poisoning all
     subsequent positions via the `K=7` ShortConv window and attention sink.
  3. **LM Head `dyntok` saturation across static `EsAAC` slices (§46.5)**: Passing `dyntok`
     (`lm_tok_ratio`) into the 8 static `EsAAC`-compiled INT8 LM head slices scaled `h_final` (`amax
     = 9.90` vs `input_threshold = 20.82`) up by `2.10x`, saturating all logits `> 12.8` at `+127` so
     slice `d0_s0` (`out_step = 0.21785`, token `20`) systematically beat slice `d0_s1` (`out_step =
     0.21118`, token `17479`).
  - **Zero-recompile fix verified on `roma` (§46.5, §47)**: `build_lfm3_int8_pkg.py` now computes
    SmoothQuant input channel scales (`in_inv_smooth[c]`) and per-output-channel `CvtPrecision_QuantM`
    steps (`out_step_chan[c]`, stored in `quant_chan_scales.bin`) across all 72 layer subgraphs, and
    `lfm3_v2.cpp` uses static input scaling on the INT8 LM head. On `roma` (`LFM3-1.7B_ENNP`), this
    achieves **`min-layer = 0.98892771`, `logits = 0.99594256`, `4/4` exact greedy token match
    (`[17479, 20, 309, 358]`), `8.96 tok/s` sustained decode (`55.5–72.1 tok/s` prefill)**, and passes
    all gate-integrity checks.
- **Calibration is thin and in-sample.** 32 sentences / **512 tokens** of real text, captured with
  PyTorch forward hooks (not synthetic noise), supplies every threshold -- and the same 512 tokens
  are then used to score perplexity. Needs a larger, held-out set.
- **Output-buffer sizing & direct `CacheMode::Cached` pack/unpack resolved (§47.1).** `ES_SYS_MemFlushCache(fd)`
  takes an fd and no range, so maintenance covers the whole allocation (`2 MiB` driver buckets previously
  cost `~58.7 ms/tok`). Because write-extent probing (§45.1) proved every `T=1` archetype writes `<= 8,192 B`,
  sizing `T=1` output surfaces to `32 KiB` (and `T=64` to `256/768 KiB`), zero-filling + flushing each
  DMA buffer at `init()` and `reset_state()`, and packing/unpacking directly into `CacheMode::Cached` DMA
  buffers (eliminating the `g_stage_*` staging buffers and `g_stage_mutex`) cuts host CPU decode overhead
  to `8.66 ms/tok` (`7.7%` of decode time) and boosts decode throughput by **+60.9% in INT8 (`5.57` $\to$
  `8.96 tok/s`)** and **+33.7% in FP16 (`4.63` $\to$ `6.19 tok/s`)**.
- **Full `W=64` (`kPrefillBatch = 64`) INT8 Prefill (`55.5–72.1 tok/s`, `9.4x–12.2x` faster, §47.2 & §47.3).**
  `T=64` batching is enabled for INT8 after fixing NVDLA's 32-channel (`[C/32, T=64, 32]`) `FEATURE` input
  packing and aligning `mlp_t1`'s `w2` `CvtPrecision_QuantM` surfaces up to `mlp_t64`'s 256-byte (`2 × 64`
  INT16 entries, `+4,096 B/layer = +96 KiB` total) SDP planar-transpose layout so both `T=1` and `T=64`
  share a single weight blob and all 64 token slots are active (`41` tokens in `739.3 ms` = `55.46 tok/s`;
  `286` tokens in `3,968.0 ms` = `72.08 tok/s`, vs `5.90 tok/s` originally).
- **Long context & `> 256`-token SWA ring-wrap fix (§47.3).** Expanding `SwaKVCache` ring capacity from
  `kSwaWin = 256` to `kSwaCap = kSwaWin + kChunk = 320` and iterating over exact causal positions
  `p in [max(0, abs_pos - 255), abs_pos]` (`slot = p % 320`) prevents batched `T <= 64` prefill from
  overwriting active sliding-window KV slots or attending to future tokens in the same batch when crossing
  `pos = 256` (verified end-to-end on a 286-token prompt + 64-token decode = 350 tokens). The global-attention
  KV lives on the host (`--max-ctx` defaults to `4096`, configurable up to `32768`); measured cost crosses
  the NPU's weight-stream cost somewhere between 32K and 128K tokens (§1.1 item 3).

---

## 5. Resident Socket Daemon & OpenAI-Compatible REST Server

To avoid the ~20s cold-start initialization time between queries and integrate with standard LLM clients, `lfm3_v2` can run as a resident background daemon on the board, bridged to an OpenAI-compatible REST server.

### 5.1 Start the Resident Engine on `roma`

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP
sudo ./lfm3_v2 --int8 --daemon 8088
```

The daemon keeps all weights resident in dual-die MMZ and listens on port `8088` for JSON-lines queries.

### 5.2 Start the OpenAI API Server (on host or board)

```bash
cd /home/chri/aixigua/romanaccio/src/LFM3
/home/chri/aixigua/venv/bin/python lfm3_openai_server.py --host 0.0.0.0 --port 8000 --backend-host roma --backend-port 8088
```

Endpoints available:
- `GET  /v1/models` - Lists available models (`lfm3-1.7b`, `lfm3-1.7b-heretic`, `lfm3-vl-2b`, `d1-2b`).
- `POST /v1/chat/completions` - Full chat completion spec, streaming Server-Sent Events (SSE) at ~9 tok/s, and multimodal `image_url` payloads.
- `POST /v1/completions` - Raw prompt completions.
- `POST /v1/decisions` - Single-pass zero-decode typed decisions (`d1-2b`).
- `GET  /healthz` - Health probe.

---

## 6. Zero-Decode Typed Decisions (`d1-2B`)

Liquid AI's `d1-2B` model answers questions directly from candidate logits at designated answer slots from a **single forward prefill pass**. Nothing is decoded autoregressively, ensuring zero schema violations, deterministic outputs, and sub-1000ms decision latency.

`d1_runner.py` is a client application that constructs the structured ChatML prompt, maps candidate answers to target vocabulary tokens, sends the prefill request to the resident NPU engine daemon over TCP, and scales logits using the official piecewise-linear calibration tables (`calibration.json`).

### 6.1 Step 1: Start the Resident Engine Daemon on `roma`

Because `d1_runner.py` offloads the forward prefill pass to the dual-die NPU, `lfm3_v2` must be running as a resident daemon on the board:

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP
sudo ./lfm3_v2 --int8 --daemon 8088
```

The engine keeps the quantized weights resident in dual-die MMZ and listens on port `8088` for single-pass logit requests.

### 6.2 Step 2: Execute Typed Decisions via `d1_runner.py`

From your host environment (or directly on the board), execute decision queries against the daemon:

```bash
cd /home/chri/aixigua/romanaccio/src/LFM3

# Binary yes/no question (noul)
/home/chri/aixigua/venv/bin/python d1_runner.py \
  --state "Customer ordered a Dell XPS 15 laptop on November 3rd with tracking number 987654321." \
  --question "Did the customer purchase electronic hardware?" \
  --type noul --host roma --port 8088

# Multi-choice categorization (choice)
/home/chri/aixigua/venv/bin/python d1_runner.py \
  --state "I need to return this sweater because it is too small and get my money back." \
  --question "What is the customer's intent?" \
  --type choice \
  --criteria '{"billing_refund": "Refund request", "tech_support": "Technical help", "general_inquiry": "General question"}' \
  --host roma --port 8088
```

### 6.3 Step 3 (Optional): Query via OpenAI-Compatible REST API

When the REST server is running (see [Section 5.2](#52-start-the-openai-api-server-on-host-or-board)), decisions can also be submitted over HTTP:

```bash
curl -s http://127.0.0.1:8000/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d1-2b",
    "state": "Customer ordered a Dell XPS 15 laptop on November 3rd with tracking number 987654321.",
    "question": "Did the customer purchase electronic hardware?",
    "type": "noul"
  }'
```

---

## 7. Multimodal Vision Understanding (`LFM3-VL-2B`)

Liquid AI's `LFM3-VL-2B` pairs a 164M hybrid vision encoder (8 Conv2D + 6 bidirectional attention layers) with the 24-layer LFM3 backbone. Input images are preprocessed and encoded into `[N, 2048]` visual embeddings (e.g., 64 visual tokens for standard 256x256 patches) and injected directly into Layer 0 hidden states at `<|image|>` placeholder positions.

You can run multimodal vision either via the **interactive REPL** on `roma` or automatically via the **OpenAI REST API**.

---

### Workflow A: Standalone Interactive REPL on `roma` with `/image`

In this workflow, you encode images to a `.bin` visual embedding file on your host, transfer it to `roma`, and chat interactively.

#### A.1 Extract Visual Embeddings on the Host
Use `vision_encoder.py` (which runs the ONNX vision encoder) to convert an image into binary float32 embeddings:

```bash
cd /home/chri/aixigua/romanaccio/src/LFM3

# Generate 64 visual tokens (512 KiB) from any PNG or JPEG
/home/chri/aixigua/venv/bin/python vision_encoder.py /path/to/my_image.png /tmp/image_embeds.bin

# Copy the binary embeddings to the board
scp /tmp/image_embeds.bin roma:/tmp/
```

#### A.2 Start `lfm3_v2` in Interactive Mode on `roma`
Launch the engine on `roma` without the `--prompt` or `--daemon` flags:

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP
sudo ./lfm3_v2 --int8
```

#### A.3 Load the Image and Chat
At the interactive prompt, use `/image` to attach the embeddings, then ask your question:

```text
=================================================================
 LFM3-1.7B Interactive Dual-Die Chat REPL
 (type '/exit' to quit, '/nothink' to toggle thinking, '/image <path>' to load image)
=================================================================

User> /image /tmp/image_embeds.bin
[Loaded 64 visual tokens from /tmp/image_embeds.bin]
User> Describe what you see in this image in detail.
Assistant> <think>
...
</think>
The image depicts a ...
```

---

### Workflow B: Automated Multimodal Service via OpenAI REST API

This is the recommended workflow for web apps, clients, and automated testing. You do **not** need to manually convert images to `.bin`; the REST server handles image fetching, ONNX encoding, and streaming automatically.

#### B.1 Start the Resident Engine Daemon on `roma`
```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP
sudo ./lfm3_v2 --int8 --daemon 8088
```

#### B.2 Start the OpenAI REST Server on the Host
```bash
cd /home/chri/aixigua/romanaccio/src/LFM3
/home/chri/aixigua/venv/bin/python lfm3_openai_server.py --host 0.0.0.0 --port 8000 --backend-host roma --backend-port 8088
```
*(The server automatically loads `Lfm3VisionEncoder` on startup).*

#### B.3 Send Multimodal Chat Completion Requests
Use standard OpenAI multimodal payload syntax with local filepaths, base64 data URIs, or HTTP URLs:

```bash
curl -N -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm3-vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the geometry and color of this image in one sentence."},
          {"type": "image_url", "image_url": {"url": "/path/to/my_image.png"}}
        ]
      }
    ],
    "stream": true
  }'
```

---

## 8. PEFT LoRA Fine-Tuning Fusion & Heretic Deployment (`MihaiPopa-1/LFM3-1.7B-Heretic-LoRA`)

To apply LoRA fine-tuning adapters without modifying `.model` topology or adding runtime overhead:

```bash
# 1. Fuse LoRA matrices into base weights in FP32
python apply_lora.py \
  --base-dir /home/chri/aixigua/deep/models/LFM3-1.7B \
  --lora-dir /home/chri/aixigua/deep/models/LFM3-1.7B-Heretic-LoRA \
  --out-dir /home/chri/aixigua/deep/models/LFM3-1.7B-Heretic

# 2. Build the zero-recompile INT8 package (reuses Class B archetypes & static embedding)
python build_lfm3_int8_pkg.py \
  --model-dir /home/chri/aixigua/deep/models/LFM3-1.7B-Heretic \
  --pkg-out-dir /home/chri/aixigua/deep/models/LFM3-1.7B-Heretic_ENNP \
  --base-pkg-dir /home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg_int8

# 3. Deploy to roma and run
ssh roma
cd /home/chri/models/LFM3-1.7B-Heretic_ENNP
sudo ./lfm3_v2 --int8 --prompt "Write a short rebellious haiku about breaking boundaries." -n 64
```

Measured on `roma`:
- **Prefill Speed**: **`54.95 tokens/sec`** (`18.2 ms/token`).
- **Decode Speed**: **`9.02 tokens/sec`** (`112.6 ms/token`).
- Fully accessible via the resident daemon and OpenAI endpoint (`"model": "lfm3-1.7b-heretic"`).

---

## 9. Prefill Throughput & Large Context Window Benchmarks (4K–32K)

Benchmarking LLM prefill speed requires a representative workload. Rather than artificial repetitive sequences (e.g. repeated words or synthetic patterns that compress trivially or skew attention distributions), real C source code provides the gold standard:
- **Realistic BPE Distribution**: Tests tokens spanning identifiers, control flow keywords, punctuation, indentation, and function declarations.
- **Hierarchical Dependency Chains**: Stresses KV-cache preservation across nested blocks, header declarations, and implementation scopes.
- **Authentic Engineering Workload**: Measures how quickly the model digests a multi-thousand-line codebase before answering developer queries.

Two self-contained benchmark source files are included:
1. `bench_c_code_4k.c` (~3,781 ChatML prefill tokens): A fast validation benchmark completing in ~55 seconds.
2. `bench_c_code_32k.c` (~31,762 ChatML prefill tokens): A full 32K context stress test encompassing the complete ESWIN contiguous DMA allocator, NUMA worker pool, multi-die mailbox driver, and hardware KV-cache.

### 9.1 Fast 4K Token Prefill Benchmark (~55 seconds)

Run a single-turn prefill and generation with thinking disabled (`--no-think`) and a 1024-token response budget:

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP

# Prefill 3,781 tokens, generate up to 1024 tokens
sudo ./lfm3_v2 --int8 -f bench_c_code_4k.c --no-think -n 1024 --max-ctx 8192
```

### 9.2 Full 32K Token Context Window Benchmark (~7–8 minutes)

For the full 32K context window evaluation:
- Use `-f bench_c_code_32k.c` (or `--prompt-file bench_c_code_32k.c`) as the prompt.
- Set `--max-ctx 36864` (or `40960`) so the engine allocates sufficient host KV-cache headroom for both the 31,762 prefill tokens and the 4,096 response tokens.
- Set `-n 4096` to allow a comprehensive technical explanation.
- Pass `--no-think` to prevent the model from spending generation cycles inside internal `<think>` loops.

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_ENNP

# Full 32K context benchmark: 31,762 prefill tokens + up to 4,096 generated tokens
sudo ./lfm3_v2 --int8 -f bench_c_code_32k.c --no-think -n 4096 --max-ctx 36864
```

### 9.3 Live Progress & Measured Output Metrics

During execution:
- **Live Prefill Progress**: When running prompts over 128 tokens, `lfm3_v2` dynamically prints chunked progress updates:
  ```text
  [Prefill progress: 768/3781 tokens (20.3%) | 74.4 tok/s]
  ```
- **Performance Report**: Upon completing generation, the engine displays the exact prefill and decode metrics:
  ```text
  [Prefill: 3781 tok in 59168.8 ms (63.90 tok/s, 15.6 ms/tok) | Decode: 128 tok at 7.84 tok/s (128.6 ms/tok)]
  ```
  - **`Prefill tok/s`**: Rate at which the dual-die NPU batches and streams weights across the 64-token chunked graphs while the 8 P550 CPU cores execute L2-tiled, NUMA-pinned (`mmap` + `mbind(MPOL_BIND)` + `MADV_HUGEPAGE` 2 MiB HugePages) batched INT8 64-bit SWAR attention (`63.90 tok/s` sustained over 3,781 tokens, peaking at `74.4 tok/s`).
  - **`Decode tok/s`**: Sustained memory-bandwidth-bound autoregressive token generation rate (`9.44 tok/s` at short context, `7.96 tok/s` at `~4,000` tokens context with fused SmoothQuant RMSNorm).


