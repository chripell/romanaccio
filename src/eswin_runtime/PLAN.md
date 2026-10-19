# Open ESWIN NPU Runtime (`eswin_runtime`) Implementation Plan

## 1. Project Overview & Objectives

The goal of `eswin_runtime` is to provide a modern, production-grade, open-source C/C++ library linkable into user applications. It abstracts hardware execution across the dual NPUs (`/dev/npu0`, `/dev/npu1`) and DSP cores, manages model packaging and automated downloading, and integrates complete modern LLM runtime tooling (tokenization, sampling, KV caching, streaming generation) inspired by `llama.cpp`.

### 1.1 Key Objectives
- **Developer-Friendly C API**: Replace the cumbersome, error-prone vendor API with an ergonomic, object-oriented C API (`es_context`, `es_model`, `es_tokenizer`, `es_sampler`).
- **Seamless Dual-Die Orchestration**: Circumvent the fatal vendor driver limitation (`ES_NPU_ERROR_DEV_MULTI_SET` / `0xa00f6046`) by providing a zero-overhead multi-process worker pool pinned to NUMA nodes 0 and 1 with shared-memory DMA passing.
- **Model Manifest & Automated Downloader**: Enable loading models via Hugging Face repo IDs or URL manifests with automatic downloading, SHA-256 verification, and local caching.
- **Native Embedded LLM Tooling**:
  - Port fast, dependency-free BPE / TikToken / SentencePiece tokenizers from `llama.cpp`.
  - Port advanced samplers (Temperature, Top-K, Top-P, Min-P, sliding-window repetition penalty) with built-in Issue #47 loop protection.
  - Manage high-speed dynamic KV-cache in contiguous MMZ DMA memory.
- **Direct Kernel Mode Backend (Future L0')**: Provide an optional direct kernel ioctl backend targeting `/dev/npu0` and `/dev/npu1` without requiring vendor userspace shared libraries.

---

## 2. Programming Language Evaluation & Selection

### Options Considered
- **Pure Python**: Good for rapid scripting, but introduces significant GIL latency, high runtime memory overhead, and poor embeddability into C/C++ applications (robotics, edge inference servers, C gRPC services).
- **Pure C (C11)**: Maximum portability and standard ABI, but lacks RAII, template safety for tensor buffers, and requires excessive boilerplate for concurrent multi-die thread/process synchronization.
- **Modern C++17 Core with C11 Public ABI (Selected)**:
  - **Public Interface in C11 (`es_runtime.h`)**: Pure C function declarations and opaque handles (`es_model_t*`, `es_session_t*`). Guarantees zero ABI issues when linked into C, C++, Rust, Go, or Python (via `ctypes`/`cffi`).
  - **Internal Core in C++17**: Utilizes RAII for DMA buffer cleanup (`std::unique_ptr`), safe lock-free queues, condition variables, and allows direct, clean porting of `llama.cpp` vocabularies and samplers.
  - **Optional Python Bindings**: Lightweight Python wrapper (`eswin_runtime.py`) wrapping the C ABI.

---

## 3. Layered Architectural Design

```
+---------------------------------------------------------------------------------------+
|                              Application Layer                                        |
|         (C/C++ Applications, Interactive REPL CLI, Python Bindings, Web API)           |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                Layer 2: LLM Services                                  |
|  +------------------------+  +------------------------+  +-------------------------+  |
|  |     Model Manager      |  |   Embedded Tokenizer   |  |    Sampling Engine      |  |
|  |  (HF Download, Cache,  |  |   (BPE, TikToken, SPM  |  | (Temp, Top-P, Top-K,    |  |
|  |   SHA256 Manifest)     |  |    from llama.cpp)     |  |  Sliding Rep-Penalty)   |  |
|  +------------------------+  +------------------------+  +-------------------------+  |
|  +---------------------------------------------------------------------------------+  |
|  |                   KV-Cache Manager & Transformer Sequence Engine                |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                         Layer 1: High-Level Model & Session API                       |
|           (es_model_t, es_session_t, Multi-Die Worker Daemon / NUMA Scheduler)        |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v (Current Primary)                           v (Future Alternative)
+---------------------------------------+     +-----------------------------------------+
|     Layer 0: Vendor Driver Wrapper    |     |   Layer 0': Direct Kernel ioctl Driver  |
|  - libedla_runtime.so.1.0             |     |  - Direct /dev/npu0, /dev/npu1 ioctls   |
|  - libes_memory.so (ES_SYS_MemAlloc)  |     |  - Direct /dev/dma_heap SMMU mapping    |
+---------------------------------------+     +-----------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           v
+---------------------------------------------------------------------------------------+
|                    Hardware Layer: EIC7702 (Dual-Die RISC-V SoC)                      |
|                  Die 0 (NPU0 + 4x DSP)  <===>  Die 1 (NPU1 + 4x DSP)                  |
+---------------------------------------------------------------------------------------+
```

---

## 4. Public C API Specification Design (`include/es_runtime.h`)

```c
#ifndef ES_RUNTIME_H
#define ES_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct es_context_t es_context_t;
typedef struct es_model_t es_model_t;
typedef struct es_session_t es_session_t;
typedef struct es_tokenizer_t es_tokenizer_t;
typedef struct es_sampler_t es_sampler_t;

typedef enum {
    ES_SUCCESS = 0,
    ES_ERR_INVALID_ARG = -1,
    ES_ERR_NO_MEMORY = -2,
    ES_ERR_DEVICE_BUSY = -3,
    ES_ERR_HARDWARE_FAULT = -4,
    ES_ERR_DOWNLOAD_FAILED = -5,
    ES_ERR_CHECKSUM_MISMATCH = -6
} es_status_t;

typedef enum {
    ES_DEVICE_DIE0 = 0,
    ES_DEVICE_DIE1 = 1,
    ES_DEVICE_DUAL_DIE_AUTO = 2
} es_device_mode_t;

// 1. Platform & Context Management
es_status_t es_init(es_context_t **ctx);
void        es_free(es_context_t *ctx);

// 2. Model Management & Automated Downloading
typedef struct {
    const char *repo_id;          // e.g. "HuggingFaceTB/SmolLM2-135M-Instruct" or local path
    const char *cache_dir;        // e.g. "~/.cache/eswin/models"
    const char *manifest_url;     // Optional custom download URL
    es_device_mode_t device_mode; // Target single die or dual die
} es_model_config_t;

es_status_t es_model_load(es_context_t *ctx, const es_model_config_t *config, es_model_t **model);
void        es_model_unload(es_model_t *model);

// 3. Tokenizer Interface (BPE / TikToken / SentencePiece)
es_status_t es_tokenizer_load(const char *vocab_path, es_tokenizer_t **tok);
es_status_t es_tokenize(es_tokenizer_t *tok, const char *text, int32_t *tokens, size_t *num_tokens, bool add_bos);
es_status_t es_detokenize(es_tokenizer_t *tok, int32_t token, char *buf, size_t buf_len, size_t *written);
void        es_tokenizer_free(es_tokenizer_t *tok);

// 4. Sampler Interface
typedef struct {
    float   temperature;          // Sampling temperature (0.0 = greedy)
    int32_t top_k;                // Top-K sampling cutoff
    float   top_p;                // Top-P nucleus sampling threshold
    float   min_p;                // Min-P sampling threshold
    float   repeat_penalty;       // Repetition penalty multiplier (e.g. 1.15)
    size_t  repeat_window;        // Sliding token window for penalty (e.g. 64)
} es_sampler_params_t;

es_status_t es_sampler_create(const es_sampler_params_t *params, es_sampler_t **sampler);
int32_t     es_sample(es_sampler_t *sampler, const float *logits, size_t vocab_size, const int32_t *history, size_t history_len);
void        es_sampler_free(es_sampler_t *sampler);

// 5. High-Level Chat Session & Streaming
typedef void (*es_stream_callback_t)(const char *token_str, void *user_data);

typedef struct {
    const char *system_prompt;
    size_t max_tokens;
    es_stream_callback_t callback;
    void *user_data;
} es_generate_params_t;

es_status_t es_chat_generate(es_session_t *session, const char *prompt, const es_generate_params_t *params);

#ifdef __cplusplus
}
#endif

#endif // ES_RUNTIME_H
```

---

## 5. Incremental Step-by-Step Implementation Roadmap

Development is divided into **6 small, independently verifiable milestones**:

### Milestone 1: Low-Level Hardware Abstraction & Contiguous Memory Manager [COMPLETED]
- **Scope**:
  - Implement Layer 0 hardware wrapper in C++17 (`src/backend/driver_wrapper.cpp`).
  - Dynamically probe and bind `libes_memory.so` (`ES_SYS_MemAlloc`) and `libedla_runtime.so.1.0`.
  - Implement RAII DMA buffer allocator `DmaBuffer` with automatic 4KB/2MB page alignment and cache flushing (`ES_SYS_MemFlushCache`).
  - Wrap `ES_NPU_Prepare` and `ES_NPU_Submit` with strict timeouts and error code logging.
- **Deliverables**:
  - `src/backend/driver_wrapper.hpp`, `src/backend/dma_buffer.hpp`.
  - Standalone unit test `tests/test_dma_alloc.cpp`.
  - Build system: `Makefile` and `CMakeLists.txt` (cross-compilation and host testing support).
  - Public C11 ABI header: `include/es_runtime.h`.
- **Testing Requirements**:
  - Allocate, map, flush, and free 1 MB, 16 MB, and 128 MB contiguous DMA buffers on `roma` (PASSED: 141ms, 98ms, 770ms).
  - Assert zero memory leaks via `/proc/meminfo` and DMA heap checks.
  - SMMU preparation and unprepare verified across both Die 0 and Die 1.
- **Documentation**:
  - Memory management guide and error code resolution handbook (`docs/MEMORY_AND_ERROR_HANDBOOK.md`).
  - Technical reverse-engineering report updated in `romanaccio/FINDINGS.md` (Section 18).

### Milestone 2: Multi-Die Process Isolation & NUMA Worker Daemon [COMPLETED]
- **Scope**:
  - Resolve single-process multi-die driver state collision and deadlock bugs.
  - Implement `src/backend/worker_pool.cpp` and `src/backend/worker_pool.hpp`: spawns two isolated worker child processes:
    - Worker 0: Pinned to NUMA Node 0 (cores 0–3), binds `/dev/npu0`.
    - Worker 1: Pinned to NUMA Node 1 (cores 4–7), binds `/dev/npu1`.
  - Implement zero-overhead sequence packet IPC (`socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0)`) for structured request/response framing.
  - Implement cross-process DMA handle translation using `ES_SYS_RcvMemFd` with `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` allowing unprivileged operation.
  - Fix DWARF-verified 744-byte hardware task structure contract (`NPU_TASK_S`) with 10 input/output `DevBuf` arrays starting at offset 16.
- **Deliverables**:
  - `src/backend/worker_pool.hpp`, `src/backend/worker_pool.cpp`.
  - Extended public C11 ABI in `include/es_runtime.h` (`es_worker_pool_create`, `es_worker_pool_submit`, etc.).
  - Concurrency test suite `tests/test_dual_die_concurrency.cpp`.
- **Testing Requirements**:
  - Verified concurrent dual-die execution on board `roma` with YOLOv3:
    - Single Die: 17.03 ms / step (~58.7 FPS).
    - Concurrent Dual Die: 17.03 ms / step (117.41 FPS total across both dies).
    - Verified 2.00x linear throughput scaling with 0% contention overhead.
  - Assert zero deadlocks, zero `0xa00f6046` errors, and verified 100% parallel scaling.
- **Documentation**:
  - Architecture document on inter-process IPC and NUMA worker isolation (`docs/MULTI_DIE_IPC_ARCHITECTURE.md`).
  - Technical reverse-engineering report updated in `romanaccio/FINDINGS.md` (Section 19).

### Milestone 3: Model Package Manifest & Automated Downloader [COMPLETED]
- **Scope**:
  - Implement `src/model/model_manifest.cpp` and `src/model/model_manifest.hpp`: JSON package manifests (`manifest.json` specifying model components: weights, tokenizer, config, hyperparameters, SHA-256 hashes).
  - Implement `src/model/sha256.cpp` and `src/model/sha256.hpp`: Standalone, zero-dependency NIST SHA-256 implementation for streaming file validation.
  - Implement `src/model/downloader.cpp` and `src/model/downloader.hpp`: Fetches models over HTTPS / Hugging Face with local caching (`~/.cache/eswin/models/`), offline fallback, and SHA-256 verification.
  - Implement lightweight, zero-dependency JSON parser and serializer in `src/model/simple_json.hpp`.
- **Deliverables**:
  - Manifest parser and CLI download tool `eswin-fetch` (`tools/eswin_fetch.cpp`).
  - Unit test suite `tests/test_manifest.cpp` verifying JSON parsing, NIST SHA-256 vectors, and corruption rejection.
- **Testing Requirements**:
  - Verified JSON parsing and serialization round-trip.
  - Verified NIST reference test vectors for SHA-256.
  - Verified manifest integrity checks and corrupt hash rejection (`ES_ERR_CHECKSUM_MISMATCH`).
- **Documentation**:
  - Manifest JSON schema specification (`docs/MANIFEST_SPECIFICATION.md`).

### Milestone 4: Embedded Tokenizer & Advanced Sampler (Ported from llama.cpp) [COMPLETED]
- **Scope**:
  - Implement `src/nlp/tokenizer.cpp` & `src/nlp/tokenizer.hpp`: self-contained Byte-Level BPE, TikToken, and SentencePiece tokenizer supporting Hugging Face `tokenizer.json`, OpenAI TikToken, and compact binary vocabularies without external dependencies.
  - Implement `src/nlp/sampler.cpp` & `src/nlp/sampler.hpp`:
    - Port temperature, top-k, top-p (nucleus), and min-p sampling from `llama.cpp` (`llama-sampling.cpp`).
    - Integrate the Issue #47 sliding-window repetition penalty (`repeat_window = 64`, `repeat_penalty = 1.15`).
    - Integrate consecutive duplicate token suppressor (prevents stutter loops like `"the the the"`) and cyclic n-gram loop detector (breaks $ABABAB$ cycles).
  - Implement public C11 ABI in `include/es_runtime.h` (`es_tokenizer_*`, `es_sampler_*`).
- **Deliverables**:
  - `src/nlp/tokenizer.hpp`, `src/nlp/sampler.hpp`, `src/nlp/tokenizer.cpp`, `src/nlp/sampler.cpp`.
  - Comprehensive unit test `tests/test_tokenizer_sampler.cpp`.
- **Testing Requirements**:
  - Verified greedy argmax selection and temperature scaling.
  - Verified sliding-window repetition penalty demotion, consecutive repeat breaker, and cyclic loop detector.
  - Verified JSON vocabulary loading, encoding, and round-trip decoding with ChatML template formatting.
- **Documentation**:
  - API documentation for tokenization and sampling functions (`docs/TOKENIZER_AND_SAMPLER_GUIDE.md`).

### Milestone 5: LLM Context, KV-Cache & Dual-Die Session Engine
- **Scope**:
  - Implement `src/llm/kv_cache.cpp`: manages circular dynamic KV-cache in contiguous MMZ memory with RoPE positional updating.
  - Implement `src/llm/transformer_session.cpp`: orchestrates autoregressive generation:
    - Layer-by-layer forward execution across NPU workers.
    - Dual-Die LM Head parallel reduction (Die 0 + Die 1 logits).
    - Top-K sampling integration.
- **Deliverables**:
  - Core Transformer session execution engine.
- **Testing Requirements**:
  - Run full SmolLM-135M and Qwen2.5-0.5B inference loops on `roma`.
  - Assert decode speed $\ge 35\text{ tokens/s}$ for SmolLM and $\ge 20\text{ tokens/s}$ for Qwen2.5.
- **Documentation**:
  - KV-cache layout and sequence indexing guide.

### Milestone 6: High-Level C API, Streaming REPL & Python Bindings
- **Scope**:
  - Implement the complete public C11 API in `src/api/es_runtime.cpp` implementing `include/es_runtime.h`.
  - Build the interactive chat CLI `eswin-chat` with streaming terminal output, Markdown colors, and live tokens/second telemetry.
  - Implement lightweight Python ctypes module (`python/eswin_runtime/`).
- **Deliverables**:
  - Shared library `libes_runtime.so`
  - CLI binary `eswin-chat`
  - Python package
- **Testing Requirements**:
  - End-to-end interactive chat sessions on `roma` with streaming responses.
  - Memory leak verification across 10,000 continuous chat turns using Valgrind and heap profiler.
- **Documentation**:
  - Complete C API Reference, Getting Started Guide, and Example Applications.

---

## 6. Coding Standards & Review Guidelines

1. **Clean C Interface**: The public header `es_runtime.h` must remain strictly C11 compliant with zero C++ syntax or standard library leaks.
2. **Resource Safety**: Every resource (DMA buffer, file descriptor, worker process, thread) must be guarded by RAII within internal C++ code to guarantee zero leaks on error return paths or `SIGINT` signals.
3. **Thorough Commenting**: Source files must feature detailed header comments and inline rationales for all hardware and OS interactions.
