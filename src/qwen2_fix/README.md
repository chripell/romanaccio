# Issue #47 Fix: DeepSeek / Qwen2 Repetition Loop & SIGSEGV Resolution

This directory contains the fixed and enhanced implementation of the ESWIN Qwen2/DeepSeek inference client, resolving [Issue #47](https://github.com/DC-DeepComputing/Framework/issues/47).

---

## 1. Problem Description (Issue #47)

When running DeepSeek-R1-Distill-Qwen or Qwen2 models using `es_qwen2` (both in single-die and dual-die modes), users observed two fatal failure modes:
1. **Repetition Trap / "Wait... Wait... Wait..." Loop**:
   During long generation (such as during `<think>` chains or complex prompts), the model gets trapped in an infinite cycle repeating `"Wait."`, `"Wait—"`, or similar tokens until reaching `response_max_len`.
2. **Segmentation Fault (`SIGSEGV`) / Crashes**:
   Out-of-bounds memory accesses when repetition penalty is enabled, memory leaks/unhandled exceptions on missing embedding files, or KV cache buffer overflows.

---

## 2. Root Cause Analysis & Fixes

### Bug 1: Off-by-One Out-of-Bounds Memory Corruption in `repetitionPenalty`
- **Location**: `es_qwen2_client.cpp:1427` and `1458` (original lines 1301 and 1331).
- **Cause**: The boundary check used `>` instead of `>=`:
  ```cpp
  // BUG: When ids == vocabOffset + vocabSizePerDie, condition evaluates to false!
  if (ids > (mConfig.devPrivAttr[idx].vocabOffset + mConfig.vocabSizePerDie)) {
      continue;
  }
  // Accesses logitData[vocabSizePerDie], which is 1 element past the allocated buffer!
  logitData[ids - mConfig.devPrivAttr[idx].vocabOffset] = ...;
  ```
- **Fix**: Changed the boundary check to strict range checking:
  ```cpp
  int32_t vocabOffset = mConfig.devPrivAttr[idx].vocabOffset;
  int32_t vocabEnd = vocabOffset + mConfig.vocabSizePerDie;
  if (ids < vocabOffset || ids >= vocabEnd) {
      continue;
  }
  ```

---

### Bug 2: Unbounded Repetition History & Greedy Repetition Trap
- **Location**: `repetitionPenalty()` and `config.json`.
- **Cause**:
  1. The original implementation passed `outputIds` directly into an `unordered_set` without any sliding window limit. After 200–300 generated tokens, almost *all* common English conjunctions, pronouns, and verbs were penalized. This distorted the logit distribution so severely that the model was forced to choose unpenalized or higher-scoring reasoning tokens like `"Wait"`.
  2. The default `config.json` configured `"repetition_penalty": 1.0` (disabled) and `"do_sample": 0` (greedy argmax). Greedy decoding with zero temperature has no escape mechanism once a token self-reinforces.
- **Fix**:
  1. Implemented a **sliding window** (`repeat_window`, default: 64 tokens) for repetition penalty. Only recent tokens within the window receive penalty, preserving the model's vocabulary for natural sentence formation.
  2. Updated `config.json` defaults:
     - `"repetition_penalty": 1.1`
     - `"repeat_window": 64`
     - `"do_sample": 1`
     - `"temperature": 0.6`
     - `"top_k": 20`
     - `"top_p": 0.85`

---

### Bug 3: KV Cache Buffer Overrun
- **Location**: `generate()` loop and `calcKCacheOutputAddrOffset()`.
- **Cause**: `mConfig.tokenNum` specifies the compiled maximum KV cache dimension (e.g. 1024 tokens). The loop ran `for (i = 0; i < mConfig.responseMaxLen; ++i)`. When prompt tokens + response tokens approached or exceeded `tokenNum`, `calcKCacheOutputAddrOffset(i)` computed offsets beyond the DMA buffer allocation, corrupting subsequent memory regions.
- **Fix**:
  1. Clamped `maxAllowedLen = std::min(mConfig.responseMaxLen, mConfig.tokenNum - 1)`.
  2. Added an explicit capacity guard before model execution:
     ```cpp
     if (i >= mConfig.tokenNum - 1) {
         ES_LOG_WARN("Reached KV cache capacity limit (%d), stopping.\n", mConfig.tokenNum);
         Qwen2Client::mStopped.store(true);
         // Wake up condition variables and break
         break;
     }
     ```

---

### Bug 4: Incomplete / Commented-out Duplicate Loop Detector
- **Location**: `generate()` line 1219.
- **Cause**: The vendor had commented out duplicate checking (`// if (checkDuplicateWords(...)) goto gen_done;`) because the label `gen_done` did not exist in the file.
- **Fix**:
  Implemented `checkTokenRepetition(const std::vector<int32_t>& outputIds)`:
  - **Single token repetition**: Detects if the identical token repeats 4+ times consecutively.
  - **N-gram cycle repetition**: Detects repeating periodic cycles (periods 2 to 16 repeated 3 times, such as `"Wait, "` cycles).
  - On detection, cleanly triggers EOS early stopping and halts generation.

---

### Bug 5: Multi-Die Deadlock on Early Exit
- **Location**: Synchronization in `generate()` and `getTopK()`.
- **Cause**: When Die 0 encountered EOS or stopped early, Die 1 would continue to the next iteration and block inside `mPostSyncCv.wait(...)` or `mTokenIdSyncCv.wait(...)` indefinitely.
- **Fix**: Whenever generation stops on Die 0 (EOS, repetition loop, or limit reached), it sets `mStopped.store(true)` and broadcasts notifications to `mPostSyncCv` and `mTokenIdSyncCv`, allowing all die threads to exit synchronously.

---

### Bug 6: Missing EOS Tokens in `config.json`
- **Location**: `config.json` `"eos_list"`.
- **Cause**: Only `"<｜end▁of▁sentence｜>"` was listed. Qwen2 / DeepSeek models use `<|im_end|>` (token 151645) and `<|endoftext|>` (token 151643) at the end of assistant turns. Because `<|im_end|>` was missing, generation never stopped after completing an answer.
- **Fix**:
  1. Added `<|im_end|>` and `<|endoftext|>` to `eos_list`.
  2. Added hard token ID checks for 151643 and 151645 in `generate()`.

---

### Bug 7: Memory Leak & Unhandled Exception in `loadEmbedding()`
- **Location**: `loadEmbedding()` and `taskThreadFunc()`.
- **Cause**:
  1. If `embedding.bin` failed to open or had an invalid size, `mWordEmbeddings` was allocated with `new fp16_t[]` but never freed, leaking memory and leaving dangling pointers.
  2. `taskThreadFunc` threw `std::runtime_error` inside a `std::thread`, which terminates the process via `std::terminate()`.
  3. `getTopK()` dereferenced `mWordEmbeddings` without null checking or token index bounds validation.
- **Fix**:
  1. Safely deallocate `mWordEmbeddings` and set to `nullptr` on any file read error.
  2. Return clean `ES_FAILURE` and notify `mCompleteCv` on error in `taskThreadFunc`.
  3. Added null check and bounds clamp in `getTopK()`.

---

## 3. Standard Interactive Chat Engine & Usage

The demo has been completely refactored from the vendor's multi-choice menu into a standard conversational AI chatbot REPL with full command-line option support.

### 3.1 Command-Line Options
```text
Usage:
  es_qwen2 [options]

Options:
  -c, --config <path>           Path to model JSON configuration file.
                                (Default: /opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_7b/config.json)
  -s, --system <prompt>         Custom system prompt / assistant persona.
                                (Default: "You are a helpful and intelligent AI assistant.")
  -t, --temp <float>            Sampling temperature (e.g. 0.6, 0.7). Set to 0.0 for greedy argmax.
  -k, --top-k <int>             Top-K vocabulary filtering (e.g. 20, 40).
  -p, --top-p <float>           Top-P nucleus sampling threshold (e.g. 0.85, 0.90).
  -r, --repeat-penalty <float>  Repetition penalty factor (e.g. 1.10, 1.15).
  -w, --repeat-window <int>     Sliding window size for repetition penalty (Default: 64).
  -m, --max-tokens <int>        Maximum tokens generated per response.
  -h, --help                    Display this help message and exit.
```

### 3.2 Building
On the board (`roma`):
```bash
cd /home/chri/qwen2_fix
./build.sh
```
This produces `build/es_qwen2` (1.2 MB) and `build/libes_qwen2_process.so` (1.4 MB).

### 3.3 Running Dual-Die Chat
```bash
# Standard interactive chat with DeepSeek-7B Dual-Die Peer NPU:
sudo ./build/es_qwen2 -c deepseek_7b_1k_int8_peer/config.json

# Custom persona / system prompt:
sudo ./build/es_qwen2 \
  -c deepseek_7b_1k_int8_peer/config.json \
  -s "You are an expert Linux kernel and RISC-V systems engineer." \
  --temp 0.7 \
  --repeat-penalty 1.15

# Deterministic greedy completion:
sudo ./build/es_qwen2 \
  -c deepseek_7b_1k_int8_peer/config.json \
  --temp 0.0
```

### 3.4 Interactive Chat Session Example
```text
================================================================================
  ESWIN EIC7700X Large Language Model Engine (DeepSeek / Qwen2)
  Dual-NPU Accelerated Hardware Inference
================================================================================

[*] Loading configuration: deepseek_7b_1k_int8_peer/config.json
[*] System Prompt        : "You are an expert Linux kernel and RISC-V systems engineer."

[*] Active Generation Hyperparameters:
    - Sampling Mode      : Top-P / Top-K Sampling
    - Temperature        : 0.70
    - Top-P Nucleus      : 0.85
    - Top-K              : 20
    - Repetition Penalty : 1.15 (Sliding Window: 64)
    - Max Response Length: 1024 tokens

[+] Model ready! Interactive chat session initiated.
    Type your question and press Enter. Type 'exit' or 'quit' to leave.

[YOU]: Explain how NUMA interleaving works on the EIC7700X dual-die SoC.
[deepseek_r1_distill_qwen]: On the ESWIN EIC7700X, the two dies are interconnected via high-speed D2D links...
--------------------------------------------------------
Throughput: 14.8 tokens/s
--------------------------------------------------------

[YOU]: exit
[*] Exiting chat session...
[+] Goodbye!
```

