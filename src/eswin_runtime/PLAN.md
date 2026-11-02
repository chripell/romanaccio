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

### Milestone 9: Hardware Contiguous MMZ Dynamic KV-Cache with Circular Address Offsetting [COMPLETED]
- **Implementation Status** (2026-10-24):
  - `src/llm/hardware_kv_cache.{hpp,cpp}`: per-layer K and V `DmaBuffer`s in the owning die's MMZ zone, SMMU-prepared **through that die's worker** (preparing from the master fails `0xA00F604A`), `Cached` mode because the CPU reads them back (NoCache MMZ reads are 10–60 MB/s), with `flush_cache()` after every CPU write.
  - Both KV layouts are implemented and selected explicitly, because they are **not interchangeable and a wrong choice fails silently** as Issue #47-style gibberish:
    - `HeadMajor` — K `[head][dim][token]`, V `[head][token][dim]`. Verified against a real loadable: `eswin-sim --info -v` on Qwen2-0.5B 1k INT8 `modified_block_0_npu_b1.model` reports K `(2, 64, 1, 1024)` and V `(2, 1024, 1, 64)`, both 256.00 KiB, with `kv_len == token_num == 1024`. This is the default.
    - `TokenMajorTiled` — the FINDINGS §5 vendor formulas verbatim (`K: elem * kv_dim * token`, `V: elem * (token_n2 * head_dim * (t / token_n2) + (t % token_n2))`), exposed as `k_token_offset()` / `v_token_offset()` for task descriptors.
  - Circular addressing via `slot_for()`, `advance()` saturating at capacity in wrap mode and **refusing** past the window in strict mode, and `reset()` that rewinds without reallocating or re-preparing (multi-turn reuse).
  - `tests/test_hardware_kv_cache.cpp` and `docs/HARDWARE_KV_CACHE_GUIDE.md` added; wired into `Makefile` and `CMakeLists.txt`.
  - Verified: suite green on host, including `bytes_per_layer() == 262144` for the real Qwen2-0.5B geometry (24 layers × 2 × 256 KiB = 12 MiB), round-trips in **both** layouts over tokens {0,1,7,63,1023}, highest-element offsets in bounds, vendor formula agreement at tile boundaries, wrap/saturation/strict refusal, dual-die placement, 6 invalid configs, and 200 reset cycles reusing one allocation. Clean `riscv64-linux-gnu-` cross-compile.
  - **Outstanding**: layout cannot yet be derived automatically — `ModelInfo` crossing the worker IPC carries only counts and buffer *sizes*, not `NPU_DIMS4_S` dims/strides, so the caller must declare `KvLayout`. Extending `IpcResponse`/`ModelInfo` to carry tensor dims is the right fix. Board allocation of real MMZ and the `TransformerSession` switch-over are Milestone 10 work.
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

### Milestone 10: Full End-to-End DeepSeek & Qwen2 Native Execution in Open Runtime (`qwen2_v2`) [PARTIALLY COMPLETED]
- **Implementation Status** (2026-10-24):
  - `es_model_load()` now falls back to a vendor `config.json` when no `manifest.json` is present, so `deepseek_7b_1k_int8_peer` and `qwen2_0_5b_int8` packages load without conversion. Root-level field mapping confirmed against the real Qwen2-0.5B 1k INT8 package and its compiled loadable: `model_name`→architecture, `vocab_size`, `embedding_size`→hidden_size (896), `embedding_dim`→head_dim (64), `num_head`→**num_key_value_heads** (2), `token_num`→context_length (1024), `token_n2`→kv_token_n2 (16), `precision`→elem_bytes (2); `num_attention_heads` derived as hidden_size/head_dim = 14. A nested `config` object still wins over root keys.
  - New `es_session_perplexity()` C API + `TransformerSession::score_sequence()`: teacher-forced, prefill-only, deterministic. Off-board with dummy weights it returns exactly the vocabulary size (uniform distribution), which is the tell that real weights are absent.
  - `tests/test_deepseek_e2e.cpp` (host mode synthesises a vendor package and drives the full C ABI, including window clamping when the session asks for 4096 against a 256-token package; benchmark mode via `ES_E2E_MODEL_DIR` emits `PERF`/`PPL` lines) and `tools/e2e_regression.sh` (cross-compiles, deploys over SSH with `--host`, gates throughput and perplexity against a recorded baseline, one-directional per metric). Both verified: gates return exit 1 on a synthetic regression.
  - `docs/DEEPSEEK_QWEN2_RUNTIME_GUIDE.md` added; wired into `Makefile` and `CMakeLists.txt`.
  - M9 hardened while wiring this up: `TokenMajorTiled` now **refuses** to guess `token_n2` and requires the package value, because the divisor heuristic picks 256 where the real compile used 16 and a wrong tile height silently mis-addresses every V write.
  - **Not done — the switch-overs this milestone exists for.** `TransformerSession` still uses its own per-layer GEMM submission with a single `m_d{0,1}_gemm_in/out` pair rather than `PipelineExecutor`; KV state is still the host `m_k_cache_flat`/`m_v_cache_flat` vectors rather than `HardwareKvCache` in MMZ; embeddings are still read into `std::vector` rather than mmap'd by `EmbeddingTable`. All three components are built and unit-tested but unwired. Each swap touches the decode path that currently produces correct output on silicon (Qwen2.5-Coder at 3.88–5.49 tok/s on `roma`), so each needs its own board-verified change: record a baseline with `tools/e2e_regression.sh --host roma --record` first, then compare after.
  - The 10.5 tok/s target in this milestone applies to the Qwen2-0.5B class; measured 1.5B numbers are 3.88–5.49 tok/s because decode is bandwidth-bound. Compare against the same model size, not across sizes.
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

### Milestone 11: Direct Kernel Mode Backend (L0' Zero-Vendor-Blob Engine) [PARTIALLY COMPLETED]
- **Implementation Status** (2026-10-24):
  - `src/backend/direct_kernel_backend.{hpp,cpp}`: blob-free allocation through the standard Linux dma-heap uAPI (`/dev/dma_heap/mmz_nid_<die>_part_0`, `DMA_HEAP_IOCTL_ALLOC` = `0xc0184800` over the 24-byte request struct), `mmap` of the returned dma-buf, cache maintenance via `DMA_BUF_IOCTL_SYNC` instead of `ES_SYS_MemFlushCache`, `/dev/npu*` open/enumerate, and `PREPARE_DMA`/`UNPREPARE_DMA`. No `dlopen` anywhere; `readelf -d libes_runtime.so` shows only `libstdc++/libm/libgomp/libgcc_s/libc` in `DT_NEEDED`.
  - All 13 catalogued ioctls pinned as literal observed encodings with `static_assert`s on decoded dir/type/nr/size, so the table is verifiable on any host without a board.
  - `ENABLE_DIRECT_KERNEL_MODE` switch added to `Makefile` (`make ENABLE_DIRECT_KERNEL_MODE=1`) and `CMakeLists.txt`.
  - `tests/test_direct_kernel_mode.cpp` and `docs/DIRECT_KERNEL_IOCTL_SPEC.md` added; wired into both build systems.
  - **New finding:** every NPU command encodes a **4-byte** payload size, including `PREPARE_DMA`, which the catalog annotates as `_IOWR('n', 0x0F, struct ES_DEV_BUF)` — but `ES_DEV_BUF` is 32 bytes and would encode as `0xc0206e0f`. Since the kernel dispatches on the full command word, the naive `_IOWR` would be rejected outright. The backend passes the buffer fd as `uint32_t`; that is a hypothesis, not verified fact.
  - Verified: host suite green (13 encodings, node naming, graceful failure naming the missing node, blob-free `DT_NEEDED`); clean `riscv64-linux-gnu-` cross-compile.
  - **Not verified on silicon:** the board run was attempted and blocked — SSH to `roma` failed with `Permission denied (publickey,password)` even though it worked earlier in the same session. So dma-heap allocation, the mapping round-trip and the `PREPARE_DMA` payload hypothesis are all unconfirmed. The test *reports* `PREPARE_DMA`'s result rather than asserting it, on purpose: asserting a guess would either fake a pass or fail the suite for a reason that is not a code defect.
  - **Not implemented:** `SUBMIT_MODEL` (needs `struct npu_model_desc`, not yet recovered — without it there is no `modelId` for `SUBMIT_TASK`, and this is the blocker for a fully blob-free inference path), firmware mailbox/completion handling (`PROCESS_REPORT` wait queues), and any integration — `DmaBuffer`/`WorkerPool` still use the `dlopen` path, so this backend is standalone and additive.
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

### Milestone 12: MoE Router & Expert-Parallel Dispatch (DeepSeek-V2-Lite Phase 2) [COMPLETED]
- **Implementation Status** (2026-10-24):
  - `src/llm/moe_router.{hpp,cpp}`: host-side gate GEMM, scoring, top-k selection and
    per-die dispatch grouping. Semantics come from the verified `DeepSeek-V2-Lite/config.json`
    (`scoring_func: softmax`, `norm_topk_prob: false`, `topk_method: greedy`,
    `n_group/topk_group: 1/1`, `routed_scaling_factor: 1.0`, `attention_bias: false`), not
    from assumption. Sigmoid scoring and `GroupLimitedGreedy` are implemented for the
    V3 family even though V2-Lite does not use them.
  - Allocation-free `route()` (all scratch sized in `init()`), OpenMP over experts,
    deterministic tie-breaking toward the lower expert index (`std::sort` is not stable, so
    the index is part of the comparator).
  - Addressing helpers for the "weights as a dynamic input tensor" execution model:
    `expert_blob_offset(layer, expert, bytes_per_expert)` yields the `DevBuf.offset` for one
    expert inside **its own die's** blob, so one loadable per layer can run any of 64 experts
    by changing an offset instead of a model. Offsets are per-die-relative by design, since
    each die has a separate `DmaBuffer`/`mem_fd`.
  - `tests/test_moe_router.cpp` added; wired into `Makefile` (`make test`) and
    `CMakeLists.txt` (`ctest`).
  - **Verified on `roma`**: 64×2048 gate, top-6 ⇒ **53.4 µs average per layer**, inside the
    <60 µs target; 1.39 ms/token over 26 MoE layers = **0.87% of a 6.25 tok/s budget**, so
    routing is confirmed not to be the bottleneck. x86 host measures 47.5 µs. Both show a
    multi-ms worst-case outlier from scheduler preemption (3.4 ms board / 10.6 ms host),
    which is the argument for pinning the routing thread in the decode loop.
  - Differential verification: 200 random tokens matched against an independent
    double-precision reference written inline in the test (no shared code with the router) on
    expert set, ordering, weights and argmax. Plus sigmoid/scaling, gate bias, group masking
    with the degenerate case, per-die offset density and collision checks, repeat-call
    determinism, and 11 rejected invalid configurations. Clean `riscv64-linux-gnu-` cross-compile.
  - **Outstanding**: no RVV-vectorised gate kernel (scalar+OpenMP already meets target, so
    this is optimisation rather than a gap); no golden comparison against HF's exact
    `DeepseekV2MoE` gating path — the inline reference covers selection semantics but not
    the library's code. Not yet called from `TransformerSession`; the MoE decode loop itself
    is DeepSeek-V2-Lite Phase 3/4 work.

---

### Milestone 13: Slice-Binding DMA Buffers for Expert Dispatch (audit finding 1.2) [COMPLETED 2026-10-28 — silicon-verified, FINDINGS §32]
- **Why it blocked MoE**: `WorkerPool::submit_task()` took `const std::vector<DmaBuffer*>`
  and hardcoded `offset = 0` / `size = buffer->size()` (`worker_pool.cpp:513, 526, 551, 559`).
  `MoERouter::expert_blob_offset()` computes exactly the offset needed to select an expert
  inside one resident per-die blob, so **the router's arithmetic was unusable
  end-to-end**. The worker side already propagates offsets correctly
  (`worker_pool.cpp:309, 337, 369, 378` → `NpuTask::inputFd[i].offset`) and the IPC wire
  format already carries `mem_fd`/`offset`/`size` per buffer — **only the master-side
  convenience API drops them.** A plumbing gap, not a design problem.

- **Semantics already settled empirically — do not re-derive**: `size` is the **slice
  length**, not the allocation size. `probe_dynamic_weight` submitted a two-expert blob as
  `DevBuf{fd, offset = 1,769,472, size = 1,769,472}` and the NPU read the second expert
  correctly, producing different output from `offset = 0` (§26) and byte-identical output to
  a separate buffer holding the same bytes (§27). INT8 reproduced it at `offset = 8,908,800`
  (§30). So `{mem_fd, offset, length}` behaves as a slice.

- **API to add** (keep the existing overload so every current caller stays untouched):
  ```cpp
  struct DmaBufferSlice {
      const DmaBuffer *buffer = nullptr;
      size_t offset = 0;
      size_t size   = 0;   // 0 => buffer->size() - offset
  };
  bool submit_task(uint16_t die_id, uint32_t model_id,
                   const std::vector<DmaBufferSlice> &inputs,
                   const std::vector<DmaBufferSlice> &outputs,
                   uint64_t *out_elapsed_ns = nullptr);
  ```
  The `std::vector<DmaBuffer*>` version becomes a thin adapter building slices with
  `offset = 0`, `size = buffer->size()`. Do **not** modify existing call sites — that overload
  is what every silicon-validated path currently uses.

- **Required validation, in the master before sending:**
  1. `buffer != nullptr` and `offset + size <= buffer->size()`. Overflow would make the NPU
     read past a prepared MMZ region into a neighbouring allocation — the Issue #47 class of
     corruption, so reject it rather than clamp it.
  2. `offset % 4096 == 0` until proven otherwise. Every offset verified on silicon so far is
     page-aligned; arbitrary offsets are **untested**, so fail loudly instead of silently
     returning wrong numbers. Add an escape hatch only after measuring.
  3. `offset`/`size` must fit the `uint64_t` IPC field (true for any realistic blob — assert,
     don't branch).
  4. Reuse the existing whole-buffer `prepare_buffer()`; do **not** add per-slice preparation.
     Measured on `roma` (§32.1): preparing the 17,817,600-byte blob once and submitting slices of
     it binds the right bytes exactly as well as preparing each range — both scopes work, so a
     blob is prepared once and sliced many times. That also stays off finding 1.3's path, since
     re-preparing one `snd_fd` is what overwrites `rcvd_fds[snd_fd]` and leaks the imported fd.

- **Tested off-board**: host-fallback `submit_task` simulates and never marshals, so the
  marshalling lives in the pure helper `fill_ipc_buffers(slices, out, cap, out_count)`.
  `tests/test_buffer_slice.cpp` (12 cases) pins offset/size propagation, the `size == 0` default,
  every rejection, and that the whole-buffer default still emits `offset = 0`,
  `size = buffer->size()` — i.e. the pre-Milestone-13 wire content.

- **Verified on-board** (`tools/probe_buffer_slice.cpp`, INT8 V2-Lite expert FFN, in both
  `Cached` and `NoCache`): each slice reproduces the whole-buffer bind of the same bytes
  byte-for-byte (`sum=216864` for the real expert, `sum=21767` for a page-rotated stand-in for
  expert B), the two slices of one blob differ, both equal the direct-driver golden of §29/§30,
  and all five malformed slice forms are refused in the master without disturbing the next good
  submit. `tools/probe_prepare_scope.cpp` is the prepare-scope isolation run behind item 4.
  Pitfall found while writing it: a *uniform* synthetic activation makes this INT8 graph return
  all zeros regardless of the weights — indistinguishable from a binding failure (§32.3).

- **Deliverables (all landed except the last)**: `DmaBufferSlice`, `SliceError`,
  `kIpcBufferSlots`/`kSliceAlignment` and the slice overload in `worker_pool.{hpp,cpp}`; the
  `std::vector<DmaBuffer*>` entry point is now a thin adapter; `tests/test_buffer_slice.cpp`
  wired into `make test` and `ctest`; the two probes above. **Remaining**: the real call site in
  the MoE decode loop, which needs Milestone 10's `TransformerSession` switch-over.

- **Two latent defects fixed on the way** (§32.4): an over-capacity vector set `input_count` from
  its own length while filling only 10 slots, handing the driver uninitialized descriptors; and
  null entries were silently skipped instead of rejected. Every call site was checked before the
  contract changed.

- **Sequencing**: independent of the cache-mode question (finding 1.1), which is still blocked on
  the empirical `MemFlushCache` direction test. Slicing as designed does *not* make 1.3 reachable
  — one prepare per blob, many slices — so 1.3 stays a cleanup rather than a MoE blocker.

---

### Milestone 14: Per-Expert Command-Stream Patching [GATE ANSWERED 2026-10-28 — route is DEAD, see FINDINGS §34]
- **The gate result**: the loaded `kmd-task-op_list` *was* located — one 99.95%-matching copy per
  model at the base of its own 2 MiB `/dmabuf:npu` mapping (`rw-s`, writable), the 48 differing
  bytes being applied relocations. Patching all 6 expert-specific bytes there (plus
  `ES_SYS_MemFlushCacheAll`) changed nothing, and the liveness control settles why: XOR-corrupting
  **64 bytes** of that copy left the output byte-identical to the golden. The NPU does not fetch
  this copy at submit; the driver took its own at load. **User-space post-load patching is
  impossible**, so do not build a patcher.
- Two traps worth keeping: a `/proc/self/maps` scanner must parse `dev` as `%x:%x` (a single
  integer conversion silently yields **zero** regions — the planted-bait self-test is what caught
  it), and stream ownership cannot be read from a match score, since two experts' streams differ
  in only a few bytes and both score 99.95%.
- **Replacement route (Milestone 15)**: one resident weight blob + offset selection (proven
  byte-exact, §33.1) + **one loadable per stream class**, where a class is a set of experts whose
  Class B files are byte-identical — identical files need no board test, since the same bytes
  cannot compute differently. Measured: 8 real experts of layer 1 under one merge-max table gave
  **3 classes (6/1/1)**; power-of-two thresholds gave 4, so widening the shared table helps and
  threshold rounding does not.
- **The one number that decides feasibility**: distinct stream classes across all 26 MoE layers
  under a single global table, versus the measured ceiling of **62 resident models per die**
  (§33.3, exceeding which Oopses the kernel). Every MoE layer shares the 2048→1408→2048 graph, so
  classes may repeat across layers. ≤ ~60 ⇒ MoE works with no patching and no per-expert loads;
  hundreds ⇒ INT8 MoE has no viable dispatch path on this driver and the design must change.
- **Deliverables for the measurement**: `make_quant_table.py --merge-max` across a global expert
  set; a compile-and-hash sweep (EsAAC is deterministic — verified: same weights under a different
  filename give an md5-identical loadable, so hashes are reproducible class IDs); and a residency
  test that loads one loadable per class and stays well under 60.
- **MEASURED 2026-11-02 (FINDINGS §35)**: 104 real experts — 4 from each of the 26 MoE layers, one
  shared table — give **15 classes**, and classes span layers (the largest covers 17 of them). The
  marginal rate falls from 0.42 to 0.06 classes/expert across the sample; a log fit extrapolates to
  **~25 classes for all 1664 experts**, i.e. inside the 62-model ceiling with ~2.5× margin. Widening
  thresholds ×2/×4 changes nothing (the requant bytes are scale-invariant), so there is no
  accuracy-vs-class-count lever to trade. Proven on silicon: layer 1's loadable computes layer 26's
  expert **byte-identically** from one shared blob (`probe_two_experts --same-class`).

---

### Milestone 15: Class-Based Expert Dispatch (the route §35 made feasible) [PLANNED]
- **Shape**: one resident INT8 weight blob per die (14.2 GiB total) + ~15–25 resident Class B
  loadables (283 KB each) + `DevBuf.offset` expert selection (Milestone 13) + a compile-time
  class→experts map. No patching, no per-expert loads, inside the 62-model ceiling.
- **Compiler deliverable**: emit the class map next to the weight regions — for each
  (layer, expert) the class id and the expert's offset in the per-die blob. `count_stream_classes.py`
  already computes the grouping; it needs a machine-readable output mode (`--json`) rather than a
  report, plus a guard that refuses to publish a map whose class count would not fit resident.
- **Runtime deliverable**: `ExpertLoadableSet` in `src/llm/` — loads one model id per class at
  session start (asserting the count stays under ~50, never near 62: §33.4), and answers
  `model_id_for(layer, expert)` + `slice_for(layer, expert)` for the dispatch loop. Reuse
  `MoERouter::expert_blob_offset()`; do not re-derive offsets.
- **Then** the Milestone 10 `TransformerSession` switch-over can call the router for real.
- **Risks to check before building**: (1) the 15→25 extrapolation samples 4 of 64 experts per
  layer — a full sweep, or a rule predicting the requant bytes from weight statistics, would turn
  the estimate into a measurement; (2) the shared table costs ×3–6 mean weight NMSE versus
  per-expert tables (absolute ≤2.6e-3), which needs an end-to-end accuracy check on real hidden
  states before it is trusted; (3) rare classes may need an LRU with 34 ms reloads if the tail is
  heavier than the fit suggests.
- **Do not** preload near the 62-model ceiling in any test: it Oopses the kernel (§33.4).

---

## 6. Coding Standards & Review Guidelines

1. **Clean C Interface**: The public header `es_runtime.h` must remain strictly C11 compliant with zero C++ syntax or standard library leaks.
2. **Resource Safety**: Every resource (DMA buffer, file descriptor, worker process, thread) must be guarded by RAII within internal C++ code to guarantee zero leaks on error return paths or `SIGINT` signals.
3. **Thorough Commenting**: Source files must feature detailed header comments and inline rationales for all hardware and OS interactions.
