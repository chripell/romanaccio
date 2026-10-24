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

Development is structured into **11 independently verifiable milestones**:
- **Phase I (Milestones 1–6)**: Core Hardware Abstraction, Multi-Die Isolation, Tokenization, Sampler, and Public C11/Python Runtime [COMPLETED]
- **Phase II (Milestones 7–10)**: Multi-Layer Block Chaining, Zero-Copy Weights/Embeddings, Hardware MMZ KV-Cache, and Full E2E DeepSeek/Qwen2 Native Execution [PLANNED]
- **Phase III (Milestone 11)**: Direct Kernel ioctl Mode Backend (Zero-Vendor-Blob Engine) [PLANNED]

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

### Milestone 5: LLM Context, KV-Cache & Dual-Die Session Engine [COMPLETED]
- **Scope**:
  - Implement `src/llm/kv_cache.hpp` and `src/llm/kv_cache.cpp`: manages multi-layer circular dynamic KV-cache in contiguous memory with linear/circular sequence indexing and reset/clear capabilities.
  - Implement `src/llm/transformer_session.hpp` and `src/llm/transformer_session.cpp`: orchestrates autoregressive generation:
    - Layer-by-layer forward execution across NPU workers.
    - Dual-Die LM Head parallel reduction (Die 0 evaluates $0 \dots V/2-1$, Die 1 evaluates $V/2 \dots V-1$).
    - Autoregressive generation loop with integrated Sampler and Tokenizer callbacks.
- **Deliverables**:
  - `src/llm/kv_cache.hpp`, `src/llm/kv_cache.cpp`.
  - `src/llm/transformer_session.hpp`, `src/llm/transformer_session.cpp`.
  - Unit test suite `tests/test_transformer_session.cpp`.
  - Build integration in `Makefile` and `CMakeLists.txt`.
- **Testing Requirements**:
  - Verified KV-cache layer indexing, token slot writing, and sequence position wrapping.
  - Verified dual-die LM Head logit reduction and top-1 argmax agreement.
  - Verified prompt ingestion, KV caching, and autoregressive multi-token generation on both host and RISC-V.
- **Documentation**:
  - KV-cache layout and sequence indexing guide (`docs/KV_CACHE_AND_DUAL_DIE_SESSION.md`).

### Milestone 6: High-Level C API, Streaming REPL & Python Bindings [COMPLETED]
- **Scope**:
  - Implement the complete public C11 API in `src/api/es_runtime.cpp` implementing `include/es_runtime.h` (`es_model_load`, `es_model_unload`, `es_model_get_info`, `es_session_create`, `es_session_generate`, `es_session_reset`, `es_session_free`, `es_version`).
  - Build the interactive chat CLI `eswin-chat` (`tools/eswin_chat.cpp`) with streaming terminal output, ANSI Markdown colors, graceful `SIGINT` interruption, and live tokens/second telemetry.
  - Implement Python ctypes bindings package (`python/eswin_runtime/`) with Pythonic context managers (`RuntimeContext`, `Model`, `Session`, `DmaBuffer`) and example interactive REPL (`python/examples/chat_repl.py`).
- **Deliverables**:
  - Shared library `libes_runtime.so`
  - CLI binary `eswin-chat` (`tools/eswin_chat.cpp`)
  - Integration test suite `tests/test_runtime_e2e.cpp`
  - Python package (`python/eswin_runtime/`, `setup.py`, `pyproject.toml`, `examples/chat_repl.py`)
- **Testing Requirements**:
  - Verified 100% test pass on host across Milestones 1 to 6 (`make test`).
  - Verified cross-compilation with `riscv64-linux-gnu-` and hardware execution on board `roma`:
    - Discovered 2 operational hardware dies.
    - Autoregressive generation completed with streaming callback.
    - Verified multi-turn execution and KV-cache reset.
- **Documentation**:
  - Public C11 API Reference & Python SDK Guide (`docs/C_API_AND_PYTHON_GUIDE.md`).
  - Technical reverse-engineering report updated in `romanaccio/FINDINGS.md` (Section 20).

### Milestone 7: Multi-Layer Block Chaining & Pipeline Engine (Dual-Die Sequential Graphs) [COMPLETED]
- **Implementation Status** (2026-10-24):
  - `src/backend/pipeline_executor.{hpp,cpp}` implemented: per-die ping-pong `Cached` MMZ activation buffers, per-die SMMU prepare, one CPU-mediated handoff per die boundary, per-stage timing (`PipelineStats`).
  - `ModelManifest` extended with `BlockModelEntry` / `blocks` / `blocks_for_die()`, parsing both the open `"blocks"` spelling and the vendor ENNP `"model_list"`/`"block_models"` spelling (`device_id` → `die_id`).
  - `tests/test_layer_chaining.cpp` and `docs/MULTI_LAYER_PIPELINE_GUIDE.md` added; wired into `Makefile` (`make test`) and `CMakeLists.txt` (`ctest`).
  - Verified: host-fallback suite green (2/8/28-stage chains, single- and dual-die, manifest-driven chain, 40-cycle RSS leak check); clean `riscv64-linux-gnu-` cross-compile.
  - **Outstanding**: the 28-layer board run and the <40 µs/layer handoff target are asserted only when hardware is present, so they still need a `roma` run. Folding the executor into `TransformerSession::step()` is deferred to Milestone 10 — that path interleaves CPU work (RoPE, softmax, KV update) between NPU GEMMs and is currently silicon-validated, so it should not be refactored without board verification.
- **Motivation & Objectives**:
  - `TransformerSession` in Milestone 5 implemented single-stage dual-die offload (specifically LM-head reduction). Large-scale models like DeepSeek-7B and Qwen2 consist of 28 to 32 sequential transformer layer block models across Die 0 and Die 1 (totaling 56+ hardware models).
  - Implement a zero-copy pipeline execution engine capable of sequentially chaining $N$ layer block models per die, passing activation tensors directly between layer stages without CPU-side round-trips.
- **Scope**:
  - Extend `ModelManifest` to support multi-block model definitions:
    - Lists of sequential block models per die (e.g. `block_models: ["modified_block_0_d0.model", ...]`).
    - Topology descriptors for inter-die peer tensor communication.
  - Implement `PipelineExecutor` (`src/backend/pipeline_executor.hpp`, `src/backend/pipeline_executor.cpp`):
    - Allocate double-buffered (ping-pong) contiguous DMA activation buffers in `mmz_nid_0_part_0` and `mmz_nid_1_part_0`.
    - Layer $i$ output DMA buffer maps directly as layer $i+1$ input DMA buffer.
    - Asynchronous batch submission of layer tasks to `WorkerPool` with lockstep die synchronization barriers.
  - Implement inter-die peer activation exchange:
    - Coordinate tensor-parallel intermediate representations between Die 0 and Die 1 at each layer boundary using shared DMA regions.
- **Deliverables**:
  - `src/backend/pipeline_executor.hpp`, `src/backend/pipeline_executor.cpp`.
  - Extended `TransformerSession` integrating `PipelineExecutor`.
  - Unit/benchmark test `tests/test_layer_chaining.cpp`.
- **Testing Requirements**:
  - Execute 2-layer, 8-layer, and full 28-layer synthetic pipeline chains on `roma`.
  - Measure per-layer activation handoff latency (target: $< 40\,\mu\text{s}$ per layer).
  - Assert zero memory leaks and proper resource unwinding on worker interruption.
- **Documentation**:
  - Pipeline execution architecture guide (`docs/MULTI_LAYER_PIPELINE_GUIDE.md`).

### Milestone 8: Zero-Copy Embedding & Weight Memory Management (`embedding.bin`, Safetensors, GGUF) [COMPLETED]
- **Implementation Status** (2026-10-24):
  - `src/model/embedding_table.{hpp,cpp}`: `mmap(PROT_READ, MAP_SHARED)` + `MADV_WILLNEED`, fp32/fp16/bf16 rows decoded to fp32 on lookup, bounds-checked `get_row()`/`gather()` (out-of-range tokens zero-fill), exact `rows x cols x elem` size validation, `open_auto()` precision inference that rejects ambiguous sizes, move-only ownership.
  - `src/model/safetensors_loader.{hpp,cpp}`: zero-dependency container reader (8-byte LE header length, JSON header, dense payload) over the same mapping; validates every `data_offsets` range against the data section and `shape x dtype` before publishing a tensor; `to_f32()` for F64/F32/F16/BF16/I64/I32/I16/I8/U8/BOOL; `__metadata__` parsed; zero-copy `data()` accessor.
  - `tests/test_embedding_loader.cpp` and `docs/WEIGHTS_AND_EMBEDDING_SPEC.md` added; wired into `Makefile` (`make test`) and `CMakeLists.txt` (`ctest`).
  - Verified: suite green on host, including mapping the exact 466,747,392-byte Qwen2.5-Coder table (151936 x 1536 fp16) in **0.003 ms** against the PLAN's <5 ms target, and rejection of truncated files, shape/dtype mismatch, unknown dtype, oversized header length, missing files and zero dimensions. Clean `riscv64-linux-gnu-` cross-compile.
  - Bug found and fixed while testing: `JsonValue::parse` only writes `err` on failure, so a caller reusing one error string saw a stale message attributed to the current file; both loaders now clear their out-param at entry.
  - **Deferred**: GGUF (named in the milestone title) is not implemented — no ESWIN package in `deep/rootfs-roma` or on `roma` ships GGUF, so it was left out rather than shipped untested. RVV-vectorised fp16 unpacking is also not done; the portable scalar path is used on both host and RISC-V. `TransformerSession` still loads via its existing `std::vector` paths — switching it over is Milestone 10 work and needs a board run.
- **Motivation & Objectives**:
  - Large language models require external embedding tables (e.g. 1.08 GB raw fp16 for DeepSeek-7B, 272 MB for Qwen2-0.5B). Currently, `TransformerSession` uses a fallback dummy constant (`0.01f`), producing uniform logits and gibberish sampling output if real weights are not mapped.
  - Implement an ultra-fast zero-copy memory-mapped embedding and weight manager supporting raw binary tables (`embedding.bin`), Hugging Face `safetensors`, and GGUF.
- **Scope**:
  - Implement `EmbeddingTable` (`src/model/embedding_table.hpp`, `src/model/embedding_table.cpp`):
    - Memory-map huge embedding files via `mmap(PROT_READ)` with `MAP_SHARED` / `MADV_WILLNEED` to eliminate multi-gigabyte heap allocations and enable sub-millisecond cold starts.
    - Provide fast token-to-embedding vector lookups with boundary validation.
    - Precision handling: support fp16, bf16, and fp32 embeddings, with auto-detection from file size vs. vocabulary/hidden dimensions.
    - Optimized fp16-to-fp32 unpacking on RISC-V with vector extensions (`rvv`) where available, and clean portable scalar fallback.
  - Implement standalone `SafetensorsLoader` (`src/model/safetensors_loader.hpp`, `src/model/safetensors_loader.cpp`):
    - Zero-dependency parsing of Hugging Face `.safetensors` headers and tensor offset maps.
- **Deliverables**:
  - `src/model/embedding_table.hpp`, `src/model/embedding_table.cpp`.
  - `src/model/safetensors_loader.hpp`, `src/model/safetensors_loader.cpp`.
  - Unit test suite `tests/test_embedding_loader.cpp`.
- **Testing Requirements**:
  - Memory-map 1.08 GB DeepSeek `embedding.bin` in $< 5\,\text{ms}$ on `roma`.
  - Verify embedding lookups against golden Hugging Face reference vectors.
  - Verify error handling on corrupted, truncated, or size-mismatched embedding files.
- **Documentation**:
  - Weights format and memory mapping specification (`docs/WEIGHTS_AND_EMBEDDING_SPEC.md`).

### Milestone 9: Hardware Contiguous MMZ Dynamic KV-Cache with Circular Address Offsetting [PLANNED]
- **Motivation & Objectives**:
  - In `TransformerSession`, KV caching is currently implemented as a host software buffer. The hardware NPU block models expect K and V caches directly mapped to contiguous physical DMA memory (`IN_K_CACHE_IDX`, `IN_V_CACHE_IDX`) with circular address offsets computed per sequence step.
- **Scope**:
  - Implement `HardwareKvCache` (`src/llm/hardware_kv_cache.hpp`, `src/llm/hardware_kv_cache.cpp`):
    - Allocate physical DMA buffers for K and V caches per layer in MMZ (`mmz_nid_0_part_0` / `mmz_nid_1_part_0`).
    - Bind physical memory descriptors (`memFd`, physical base address) into the task input/output structures passed to the NPU kernel.
    - Implement circular sequence address offsets per token step (`calcKCacheOutputAddrOffset`, `calcVCacheOutputAddrOffset`) to allow context sliding up to 2048/4096 tokens.
    - Support multi-turn conversation caching, slot recycling, and fast reset without reallocating physical memory.
- **Deliverables**:
  - `src/llm/hardware_kv_cache.hpp`, `src/llm/hardware_kv_cache.cpp`.
  - Unit test suite `tests/test_hardware_kv_cache.cpp`.
- **Testing Requirements**:
  - Allocate and bind 28 layers of dual-die K and V caches in physical MMZ memory on `roma`.
  - Verify sequential token position writes without overwriting or address misalignments.
  - Verify multi-turn conversational context retention and prompt caching.
- **Documentation**:
  - Hardware KV cache layout and DMA address mapping guide (`docs/HARDWARE_KV_CACHE_GUIDE.md`).

### Milestone 10: Full End-to-End DeepSeek & Qwen2 Native Execution in Open Runtime (`qwen2_v2`) [PLANNED]
- **Motivation & Objectives**:
  - Integrate Milestones 7, 8, and 9 to enable `qwen2_v2_openruntime` to run the complete 58-model `deepseek_7b_1k_int8_peer` hardware pipeline entirely through `libes_runtime.so`, replacing the legacy vendor `Qwen2Client` with clean, maintainable open-source code.
- **Scope**:
  - Parse vendor `config.json` or native `manifest.json` for `deepseek_7b_1k_int8_peer` and `qwen2_0_5b_int8`.
  - Wire `TransformerSession` to execute the full 58-model chained pipeline through `es_session_generate`.
  - Benchmark against `es_qwen2` (target: match or exceed 10.5 tokens/sec).
  - Maintain full interactive REPL features:
    - Real-time gray thinking token streaming (`[Thinking: ...]`) with automatic token capping.
    - Token debug mode (`[#id:"piece"]`).
    - Issue #47 loop-breaking sampler (sliding repetition penalty, anti-stutter, cyclic loop breaker).
    - Comprehensive startup component card with validation and missing-piece warnings.
- **Deliverables**:
  - Fully functional `qwen2_v2` powered 100% by `libes_runtime.so`.
  - End-to-end integration test `tests/test_deepseek_e2e.cpp`.
  - Automated performance and perplexity regression script.
- **Testing Requirements**:
  - Coherent response generation for complex prompts (e.g. math reasoning, code generation) on board `roma`.
  - Sustained generation speed $\ge 10.5$ tokens/sec on dual NPUs.
  - Zero kernel driver faults, zero memory leaks, and zero deadlocks over 1,000 generated tokens.
- **Documentation**:
  - DeepSeek & Qwen2 Open Runtime deployment guide (`docs/DEEPSEEK_QWEN2_RUNTIME_GUIDE.md`).

### Milestone 11: Direct Kernel Mode Backend (L0' Zero-Vendor-Blob Engine) [PLANNED]
- **Motivation & Objectives**:
  - Eliminate the last proprietary vendor userspace binary blob (`libedla_runtime.so.1.0` and `libes_memory.so`) by implementing direct kernel ioctls to `/dev/npu0` and `/dev/npu1` and standard Linux DMA-heap buffer allocation.
- **Scope**:
  - Reverse-engineer ioctl command opcodes (`_IOWR(...)`) for job submission and status polling targeting `/dev/npu0` and `/dev/npu1`.
  - Implement direct DMA heap buffer allocation via Linux standard `/dev/dma_heap/reserved` or `/dev/dma_heap/system`, bypassing `libes_memory.so`.
  - Firmware mailbox queue management and interrupt handling in userspace.
  - Compile-time switch `ENABLE_DIRECT_KERNEL_MODE=ON` allowing completely blob-free execution.
- **Deliverables**:
  - `src/backend/direct_kernel_backend.hpp`, `src/backend/direct_kernel_backend.cpp`.
  - Standalone unit test `tests/test_direct_kernel_mode.cpp`.
- **Testing Requirements**:
  - Submit and complete NPU hardware inference jobs directly via kernel ioctls on `roma` without loading any vendor `.so` libraries.
  - Validate performance parity with vendor userspace runtime.
- **Documentation**:
  - Direct kernel ioctl specification and firmware mailbox protocol (`docs/DIRECT_KERNEL_IOCTL_SPEC.md`).

---

## 6. Coding Standards & Review Guidelines

1. **Clean C Interface**: The public header `es_runtime.h` must remain strictly C11 compliant with zero C++ syntax or standard library leaks.
2. **Resource Safety**: Every resource (DMA buffer, file descriptor, worker process, thread) must be guarded by RAII within internal C++ code to guarantee zero leaks on error return paths or `SIGINT` signals.
3. **Thorough Commenting**: Source files must feature detailed header comments and inline rationales for all hardware and OS interactions.
