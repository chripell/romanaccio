Explain the following C code:

/* =========================================================================
 * File: es_runtime.h
 * ========================================================================= */

/*
 * es_runtime.h - Open ESWIN NPU Runtime Public C11 API
 *
 * Provides a modern, ergonomic, linkable C API for accelerating deep learning
 * and LLM inference across the EIC7700X dual-die NPU and Cadence Vision Q7 DSP cores.
 *
 * Strictly C11 compliant with zero C++ syntax or dependencies.
 */

#ifndef ES_RUNTIME_H
#define ES_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * 1. Status & Error Codes
 * ============================================================================= */
typedef enum {
    ES_SUCCESS               =  0,  /* Operation succeeded */
    ES_ERR_INVALID_ARG       = -1,  /* Invalid argument or null pointer */
    ES_ERR_NO_MEMORY         = -2,  /* Out of host or contiguous DMA memory */
    ES_ERR_DEVICE_BUSY       = -3,  /* NPU hardware engine or stream busy */
    ES_ERR_HARDWARE_FAULT    = -4,  /* NPU kernel driver or hardware timeout */
    ES_ERR_NOT_INITIALIZED   = -5,  /* Subsystem or context not initialized */
    ES_ERR_NOT_SUPPORTED     = -6,  /* Requested feature or format unsupported */
    ES_ERR_IO                = -7,  /* File I/O or filesystem failure */
    ES_ERR_CHECKSUM_MISMATCH = -8,  /* Model artifact SHA-256 verification failed */
    ES_ERR_TIMEOUT           = -9   /* Task execution exceeded deadline */
} es_status_t;

/* Returns a constant human-readable description of an es_status_t code */
const char* es_status_to_string(es_status_t status);

/* =============================================================================
 * 2. Hardware Device & Cache Modes
 * ============================================================================= */
typedef enum {
    ES_DEVICE_DIE0           = 0,   /* Target Die 0 (/dev/npu0, NUMA node 0) */
    ES_DEVICE_DIE1           = 1,   /* Target Die 1 (/dev/npu1, NUMA node 1) */
    ES_DEVICE_DUAL_DIE_AUTO  = 2    /* Automatic concurrent dual-die execution */
} es_device_mode_t;

typedef enum {
    ES_MEM_NOCACHE           = 0,   /* Direct DMA memory (no CPU caching) */
    ES_MEM_CACHED            = 1    /* Cached memory (requires manual flush before DMA) */
} es_cache_mode_t;

/* =============================================================================
 * 3. Opaque Handles
 * ============================================================================= */
typedef struct es_context_t   es_context_t;   /* Primary runtime context */
typedef struct es_dma_buf_t   es_dma_buf_t;   /* Contiguous DMA buffer */
typedef struct es_model_t     es_model_t;     /* Loaded NPU model */
typedef struct es_session_t   es_session_t;   /* Active inference session */
typedef struct es_tokenizer_t es_tokenizer_t; /* Tokenizer instance */
typedef struct es_sampler_t   es_sampler_t;   /* Token sampling engine */

/* =============================================================================
 * 4. Runtime Context Management
 * ============================================================================= */

/**
 * @brief Initializes the ESWIN runtime subsystem.
 *
 * Probes hardware drivers (libedla_runtime.so, libes_memory.so), verifies device
 * count, initializes memory zones, and creates the primary runtime context.
 *
 * @param[out] ctx Pointer to receive the created context handle.
 * @return ES_SUCCESS on success, or appropriate error code.
 */
es_status_t es_init(es_context_t **ctx);

/**
 * @brief Releases all resources associated with the runtime context.
 *
 * @param ctx Context handle to release (safe if NULL).
 */
void es_free(es_context_t *ctx);

/**
 * @brief Queries the number of operational NPU dies present in the system.
 *
 * @param ctx Valid runtime context.
 * @param[out] num_devices Pointer to receive device count (typically 2 on EIC7700X).
 * @return ES_SUCCESS on success.
 */
es_status_t es_get_device_count(es_context_t *ctx, uint16_t *num_devices);

/* =============================================================================
 * 5. Contiguous DMA Memory Management
 * ============================================================================= */

/**
 * @brief Allocates a physically contiguous DMA buffer suitable for zero-copy NPU execution.
 *
 * @param ctx Valid runtime context.
 * @param size Buffer size in bytes.
 * @param cache_mode Caching behavior (ES_MEM_NOCACHE or ES_MEM_CACHED).
 * @param[out] buf Pointer to receive the allocated DMA buffer handle.
 * @return ES_SUCCESS on success, or ES_ERR_NO_MEMORY if allocation fails.
 */
es_status_t es_dma_alloc(es_context_t *ctx, size_t size, es_cache_mode_t cache_mode, es_dma_buf_t **buf);

/**
 * @brief Frees a contiguous DMA buffer and unmaps its memory.
 *
 * @param buf DMA buffer handle to free (safe if NULL).
 */
void es_dma_free(es_dma_buf_t *buf);

/**
 * @brief Obtains the userspace virtual address pointer to a mapped DMA buffer.
 *
 * @param buf Valid DMA buffer handle.
 * @return Pointer to accessible host memory, or NULL on error.
 */
void* es_dma_get_host_ptr(es_dma_buf_t *buf);

/**
 * @brief Obtains the physical memory file descriptor of a DMA buffer.
 *
 * @param buf Valid DMA buffer handle.
 * @return File descriptor (memFd) used for driver ioctls.
 */
uint64_t es_dma_get_fd(es_dma_buf_t *buf);

/**
 * @brief Obtains the capacity of the allocated DMA buffer in bytes.
 *
 * @param buf Valid DMA buffer handle.
 * @return Buffer capacity in bytes.
 */
size_t es_dma_get_size(es_dma_buf_t *buf);

/**
 * @brief Flushes CPU data cache to memory for cached DMA buffers.
 *
 * Must be invoked prior to dispatching NPU hardware tasks if ES_MEM_CACHED was used.
 *
 * @param buf Valid DMA buffer handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_dma_flush_cache(es_dma_buf_t *buf);

// =============================================================================
// Multi-Die Process Isolation & Worker Pool APIs (Milestone 2)
// =============================================================================

typedef struct es_worker_pool_t es_worker_pool_t;

/**
 * @brief Creates a multi-process isolated worker pool pinned to NUMA nodes.
 *
 * Spawns isolated worker daemons for Die 0 (cores 0-3) and Die 1 (cores 4-7)
 * to circumvent the vendor single-process driver limitation (0xA00F6046).
 *
 * @param ctx Valid runtime context.
 * @param num_dies Number of NPU dies to manage (1 or 2).
 * @param[out] pool Pointer to receive the allocated worker pool handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_create(es_context_t *ctx, uint16_t num_dies, es_worker_pool_t **pool);

/**
 * @brief Destroys the worker pool and terminates all worker daemon processes.
 *
 * @param pool Valid worker pool handle (safe if NULL).
 */
void es_worker_pool_destroy(es_worker_pool_t *pool);

/**
 * @brief Loads a compiled .model into the target die's worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_path Path to the .model file on the filesystem.
 * @param[out] model_id Assigned model identifier on that die.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_load_model(es_worker_pool_t *pool, uint16_t die_id, const char *model_path, uint32_t *model_id);

/**
 * @brief Unloads a model from the target die's worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_id Model identifier to unload.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_unload_model(es_worker_pool_t *pool, uint16_t die_id, uint32_t model_id);

/**
 * @brief Prepares and maps a DMA buffer into the target die's SMMU page table.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param buf Valid DMA buffer handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_prepare_buffer(es_worker_pool_t *pool, uint16_t die_id, es_dma_buf_t *buf);

/**
 * @brief Submits a synchronous inference task to the target die worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_id Loaded model ID.
 * @param input_buf Input DMA buffer.
 * @param output_buf Output DMA buffer.
 * @param[out] elapsed_ns Optional pointer to receive hardware latency in nanoseconds.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_submit_task(
    es_worker_pool_t *pool,
    uint16_t die_id,
    uint32_t model_id,
    es_dma_buf_t *input_buf,
    es_dma_buf_t *output_buf,
    uint64_t *elapsed_ns
);

// =============================================================================
// NLP Tokenizer & Sampler APIs (Milestone 4)
// =============================================================================

typedef struct {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    int32_t  repeat_window;
    int32_t  max_consecutive_duplicates;
    uint32_t seed;
} es_sampler_params_t;

es_status_t es_tokenizer_load(const char *filepath, es_tokenizer_t **out_tokenizer);
void es_tokenizer_free(es_tokenizer_t *tokenizer);
size_t es_tokenizer_vocab_size(const es_tokenizer_t *tokenizer);
int32_t es_tokenizer_encode(const es_tokenizer_t *tokenizer, const char *text, int32_t *out_tokens, size_t max_tokens);
int32_t es_tokenizer_decode(const es_tokenizer_t *tokenizer, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_len);

es_status_t es_sampler_create(const es_sampler_params_t *params, es_sampler_t **out_sampler);
void es_sampler_free(es_sampler_t *sampler);
int32_t es_sampler_sample(es_sampler_t *sampler, float *logits, size_t vocab_size);
void es_sampler_accept(es_sampler_t *sampler, int32_t token_id);
void es_sampler_reset(es_sampler_t *sampler);

// =============================================================================
// High-Level Model & Session APIs (Milestone 6)
// =============================================================================

/**
 * @brief Returns the runtime version string.
 */
const char* es_version(void);

typedef struct {
    const char       *model_path;     /* Directory containing manifest.json or path to .model */
    const char       *tokenizer_path; /* Path to tokenizer.json / vocab file (optional if in model dir) */
    const char       *cache_dir;      /* Model cache directory (default ~/.cache/eswin/models if NULL) */
    es_device_mode_t  device_mode;    /* Target device mode (default ES_DEVICE_DUAL_DIE_AUTO) */
} es_model_config_t;

typedef struct {
    char     name[128];
    char     architecture[64];
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t max_seq_len;
} es_model_info_t;

/**
 * @brief Loads an NPU model package and prepares execution resources.
 *
 * @param ctx Valid runtime context.
 * @param config Model configuration structure.
 * @param[out] out_model Pointer to receive the allocated model handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_model_load(es_context_t *ctx, const es_model_config_t *config, es_model_t **out_model);

/**
 * @brief Unloads a model and releases all associated resources.
 *
 * @param model Model handle to unload (safe if NULL).
 */
void es_model_unload(es_model_t *model);

/**
 * @brief Queries metadata and structural parameters of a loaded model.
 *
 * @param model Valid model handle.
 * @param[out] info Pointer to receive model information.
 * @return ES_SUCCESS on success.
 */
es_status_t es_model_get_info(const es_model_t *model, es_model_info_t *info);

/**
 * @brief Obtains a reference to the model's internal tokenizer.
 *
 * @param model Valid model handle.
 * @return Pointer to tokenizer handle, or NULL if unavailable.
 */
const es_tokenizer_t* es_model_get_tokenizer(const es_model_t *model);

typedef struct {
    size_t              max_seq_len;     /* Maximum sequence context window (default: 2048) */
    int32_t             max_new_tokens;  /* Maximum tokens to generate per turn (default: 512) */
    bool                enable_dual_die; /* Enable dual-die parallel execution */
    es_sampler_params_t sampler_params;  /* Sampling parameters */
} es_session_config_t;

/**
 * @brief Token generation callback.
 *
 * Invoked during autoregressive generation as each token piece is produced.
 *
 * @param token The sampled token ID.
 * @param piece The decoded UTF-8 text piece for the token.
 * @param user_data User-provided context pointer.
 * @return true to continue generation, or false to halt/cancel early.
 */
typedef bool (*es_token_callback_t)(int32_t token, const char *piece, void *user_data);

typedef struct {
    int32_t prompt_tokens;
    int32_t eval_tokens;
    double  prompt_eval_ms;
    double  eval_ms;
    double  tokens_per_sec;
} es_perf_stats_t;

/**
 * @brief Creates an autoregressive inference session for a loaded model.
 *
 * Allocates KV-cache, working scratch buffers, and initializes samplers.
 *
 * @param model Valid loaded model handle.
 * @param config Session configuration (or NULL for defaults).
 * @param[out] out_session Pointer to receive allocated session handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_create(es_model_t *model, const es_session_config_t *config, es_session_t **out_session);

/**
 * @brief Frees an inference session and its KV-cache.
 *
 * @param session Session handle to free (safe if NULL).
 */
void es_session_free(es_session_t *session);

/**
 * @brief Resets sequence position and clears KV-cache for a new conversation turn.
 *
 * @param session Valid session handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_reset(es_session_t *session);

/**
 * @brief Executes prompt prefill and autoregressive token generation.
 *
 * Streams generated tokens sequentially to the provided callback.
 *
 * @param session Valid session handle.
 * @param prompt Input text prompt to encode and evaluate.
 * @param callback Callback invoked for each generated token piece (optional).
 * @param user_data Context pointer passed to the callback.
 * @param[out] stats Optional pointer to receive performance telemetry.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_generate(
    es_session_t *session,
    const char *prompt,
    es_token_callback_t callback,
    void *user_data,
    es_perf_stats_t *stats
);

/**
 * @brief Teacher-forced perplexity of @p text under the loaded model.
 *
 * Runs the prefill path only (no sampling), so the result is deterministic and usable
 * as a quality regression signal across builds. Scoring stops at the compiled KV window
 * rather than writing past it.
 *
 * @param session         Valid session handle.
 * @param text            Text to tokenize and score; needs at least 2 tokens.
 * @param[out] perplexity exp(-mean log P(token[i+1] | prefix)). May be NULL.
 * @param[out] mean_logprob Mean natural log-probability per predicted token. May be NULL.
 * @return ES_SUCCESS, or ES_ERR_INVALID_ARG when nothing could be scored.
 */
es_status_t es_session_perplexity(
    es_session_t *session,
    const char *text,
    double *perplexity,
    double *mean_logprob
);

#ifdef __cplusplus
}
#endif

#endif /* ES_RUNTIME_H */


/* =========================================================================
 * File: es_runtime.cpp
 * ========================================================================= */

/**
 * @file es_runtime.cpp
 * @brief Implementation of Public C11 API for eswin_runtime.
 */

#include "es_runtime.h"
#include "../backend/driver_wrapper.hpp"
#include "../backend/dma_buffer.hpp"

#include <memory>
#include <new>
#include <cstring>

// =============================================================================
// Internal Context & Handle Implementations
// =============================================================================

struct es_context_t {
    eswin::EswinDriver &driver;
    uint16_t num_devices;

    es_context_t(eswin::EswinDriver &d, uint16_t count)
        : driver(d), num_devices(count) {}
};

struct es_dma_buf_t {
    std::unique_ptr<eswin::DmaBuffer> buffer;

    explicit es_dma_buf_t(std::unique_ptr<eswin::DmaBuffer> b)
        : buffer(std::move(b)) {}
};

// =============================================================================
// Status Helper
// =============================================================================

const char* es_status_to_string(es_status_t status) {
    switch (status) {
        case ES_SUCCESS:               return "ES_SUCCESS: Operation completed successfully";
        case ES_ERR_INVALID_ARG:       return "ES_ERR_INVALID_ARG: Invalid argument or null pointer";
        case ES_ERR_NO_MEMORY:         return "ES_ERR_NO_MEMORY: Out of host or DMA memory";
        case ES_ERR_DEVICE_BUSY:       return "ES_ERR_DEVICE_BUSY: Hardware device or engine busy";
        case ES_ERR_HARDWARE_FAULT:    return "ES_ERR_HARDWARE_FAULT: Hardware fault or driver failure";
        case ES_ERR_NOT_INITIALIZED:   return "ES_ERR_NOT_INITIALIZED: Runtime subsystem not initialized";
        case ES_ERR_NOT_SUPPORTED:     return "ES_ERR_NOT_SUPPORTED: Requested operation is not supported";
        case ES_ERR_IO:                return "ES_ERR_IO: File system or I/O failure";
        case ES_ERR_CHECKSUM_MISMATCH: return "ES_ERR_CHECKSUM_MISMATCH: SHA-256 verification mismatch";
        case ES_ERR_TIMEOUT:           return "ES_ERR_TIMEOUT: Operation exceeded execution deadline";
        default:                       return "ES_ERR_UNKNOWN: Unknown error code";
    }
}

// =============================================================================
// Context Management
// =============================================================================

es_status_t es_init(es_context_t **ctx) {
    if (!ctx) return ES_ERR_INVALID_ARG;

    auto &driver = eswin::EswinDriver::instance();
    driver.initialize();

    uint16_t devices = driver.get_num_devices();
    auto new_ctx = std::make_unique<es_context_t>(driver, devices);
    *ctx = new_ctx.release();

    return ES_SUCCESS;
}

void es_free(es_context_t *ctx) {
    if (ctx) {
        delete ctx;
    }
}

es_status_t es_get_device_count(es_context_t *ctx, uint16_t *num_devices) {
    if (!ctx || !num_devices) return ES_ERR_INVALID_ARG;
    *num_devices = ctx->num_devices;
    return ES_SUCCESS;
}

// =============================================================================
// DMA Memory Management
// =============================================================================

es_status_t es_dma_alloc(
    es_context_t *ctx,
    size_t size,
    es_cache_mode_t cache_mode,
    es_dma_buf_t **buf
) {
    if (!ctx || !buf || size == 0) return ES_ERR_INVALID_ARG;

    try {
        eswin::CacheMode mode = (cache_mode == ES_MEM_CACHED)
            ? eswin::CacheMode::Cached
            : eswin::CacheMode::NoCache;

        auto dma = std::make_unique<eswin::DmaBuffer>(size, mode);
        if (!dma->is_valid()) {
            return ES_ERR_NO_MEMORY;
        }

        auto wrapper = std::make_unique<es_dma_buf_t>(std::move(dma));
        *buf = wrapper.release();
        return ES_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ES_ERR_NO_MEMORY;
    } catch (...) {
        return ES_ERR_HARDWARE_FAULT;
    }
}

void es_dma_free(es_dma_buf_t *buf) {
    if (buf) {
        delete buf;
    }
}

void* es_dma_get_host_ptr(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return nullptr;
    return buf->buffer->data();
}

uint64_t es_dma_get_fd(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return 0;
    return buf->buffer->mem_fd();
}

size_t es_dma_get_size(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return 0;
    return buf->buffer->size();
}

es_status_t es_dma_flush_cache(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return ES_ERR_INVALID_ARG;
    return buf->buffer->flush_cache() ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

// =============================================================================
// Worker Pool C API Implementation
// =============================================================================

#include "../backend/worker_pool.hpp"

struct es_worker_pool_t {
    std::unique_ptr<eswin::WorkerPool> pool;

    explicit es_worker_pool_t(std::unique_ptr<eswin::WorkerPool> p)
        : pool(std::move(p)) {}
};

es_status_t es_worker_pool_create(es_context_t *ctx, uint16_t num_dies, es_worker_pool_t **pool) {
    if (!ctx || !pool) return ES_ERR_INVALID_ARG;

    auto p = std::make_unique<eswin::WorkerPool>();
    if (!p->start(num_dies)) {
        return ES_ERR_HARDWARE_FAULT;
    }

    auto wrapper = std::make_unique<es_worker_pool_t>(std::move(p));
    *pool = wrapper.release();
    return ES_SUCCESS;
}

void es_worker_pool_destroy(es_worker_pool_t *pool) {
    if (pool) {
        delete pool;
    }
}

es_status_t es_worker_pool_load_model(es_worker_pool_t *pool, uint16_t die_id, const char *model_path, uint32_t *model_id) {
    if (!pool || !pool->pool || !model_path || !model_id) return ES_ERR_INVALID_ARG;
    return pool->pool->load_model(die_id, model_path, *model_id) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_unload_model(es_worker_pool_t *pool, uint16_t die_id, uint32_t model_id) {
    if (!pool || !pool->pool) return ES_ERR_INVALID_ARG;
    return pool->pool->unload_model(die_id, model_id) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_prepare_buffer(es_worker_pool_t *pool, uint16_t die_id, es_dma_buf_t *buf) {
    if (!pool || !pool->pool || !buf || !buf->buffer) return ES_ERR_INVALID_ARG;
    return pool->pool->prepare_buffer(die_id, *buf->buffer) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_submit_task(
    es_worker_pool_t *pool,
    uint16_t die_id,
    uint32_t model_id,
    es_dma_buf_t *input_buf,
    es_dma_buf_t *output_buf,
    uint64_t *elapsed_ns
) {
    if (!pool || !pool->pool || !input_buf || !input_buf->buffer || !output_buf || !output_buf->buffer) {
        return ES_ERR_INVALID_ARG;
    }

    std::vector<eswin::DmaBuffer*> inputs = { input_buf->buffer.get() };
    std::vector<eswin::DmaBuffer*> outputs = { output_buf->buffer.get() };

    return pool->pool->submit_task(die_id, model_id, inputs, outputs, elapsed_ns)
        ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

// =============================================================================
// NLP Tokenizer & Sampler Implementations
// =============================================================================

#include "../nlp/tokenizer.hpp"
#include "../nlp/sampler.hpp"

struct es_tokenizer_t {
    std::unique_ptr<eswin::Tokenizer> tokenizer;
};

struct es_sampler_t {
    std::unique_ptr<eswin::Sampler> sampler;
};

es_status_t es_tokenizer_load(const char *filepath, es_tokenizer_t **out_tokenizer) {
    if (!filepath || !out_tokenizer) return ES_ERR_INVALID_ARG;
    auto tok = std::make_unique<eswin::Tokenizer>();
    std::string err;
    if (!tok->load(filepath, err)) {
        return ES_ERR_HARDWARE_FAULT;
    }
    auto handle = new es_tokenizer_t();
    handle->tokenizer = std::move(tok);
    *out_tokenizer = handle;
    return ES_SUCCESS;
}

void es_tokenizer_free(es_tokenizer_t *tokenizer) {
    if (tokenizer) {
        delete tokenizer;
    }
}

size_t es_tokenizer_vocab_size(const es_tokenizer_t *tokenizer) {
    if (!tokenizer || !tokenizer->tokenizer) return 0;
    return tokenizer->tokenizer->vocab_size();
}

int32_t es_tokenizer_encode(const es_tokenizer_t *tokenizer, const char *text, int32_t *out_tokens, size_t max_tokens) {
    if (!tokenizer || !tokenizer->tokenizer || !text || !out_tokens || max_tokens == 0) return -1;
    auto tokens = tokenizer->tokenizer->encode(text);
    size_t count = std::min(max_tokens, tokens.size());
    for (size_t i = 0; i < count; ++i) {
        out_tokens[i] = tokens[i];
    }
    return static_cast<int32_t>(count);
}

int32_t es_tokenizer_decode(const es_tokenizer_t *tokenizer, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_len) {
    if (!tokenizer || !tokenizer->tokenizer || !tokens || !out_text || max_len == 0) return -1;
    std::vector<int32_t> tok_vec(tokens, tokens + num_tokens);
    std::string text = tokenizer->tokenizer->decode(tok_vec);
    size_t copy_len = std::min(max_len - 1, text.size());
    std::memcpy(out_text, text.data(), copy_len);
    out_text[copy_len] = '\0';
    return static_cast<int32_t>(copy_len);
}

es_status_t es_sampler_create(const es_sampler_params_t *params, es_sampler_t **out_sampler) {
    if (!out_sampler) return ES_ERR_INVALID_ARG;
    eswin::SamplerParams p;
    if (params) {
        p.temperature = params->temperature;
        p.top_k = params->top_k;
        p.top_p = params->top_p;
        p.min_p = params->min_p;
        p.repeat_penalty = params->repeat_penalty;
        p.repeat_window = params->repeat_window;
        p.max_consecutive_duplicates = params->max_consecutive_duplicates;
        p.seed = params->seed;
    }
    auto handle = new es_sampler_t();
    handle->sampler = std::make_unique<eswin::Sampler>(p);
    *out_sampler = handle;
    return ES_SUCCESS;
}

void es_sampler_free(es_sampler_t *sampler) {
    if (sampler) {
        delete sampler;
    }
}

int32_t es_sampler_sample(es_sampler_t *sampler, float *logits, size_t vocab_size) {
    if (!sampler || !sampler->sampler || !logits || vocab_size == 0) return -1;
    return sampler->sampler->sample(logits, vocab_size);
}

void es_sampler_accept(es_sampler_t *sampler, int32_t token_id) {
    if (sampler && sampler->sampler) {
        sampler->sampler->accept(token_id);
    }
}

void es_sampler_reset(es_sampler_t *sampler) {
    if (sampler && sampler->sampler) {
        sampler->sampler->reset();
    }
}

// =============================================================================
// High-Level Model & Session APIs (Milestone 6)
// =============================================================================

#include "../model/model_manifest.hpp"
#include "../model/downloader.hpp"
#include "../llm/transformer_session.hpp"
#include "../llm/kv_cache.hpp"
#include <sys/stat.h>

const char* es_version(void) {
    return "1.0.0-open";
}

struct es_model_t {
    eswin::ModelManifest manifest;
    std::string model_dir;
    std::string tokenizer_path;
    es_device_mode_t device_mode = ES_DEVICE_DUAL_DIE_AUTO;
    std::unique_ptr<es_tokenizer_t> tokenizer_handle;
};

struct es_session_t {
    es_model_t *model = nullptr;
    es_session_config_t config{};
    std::unique_ptr<eswin::TransformerSession> session;
};

static bool file_exists(const std::string &path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

static bool dir_exists(const std::string &path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

es_status_t es_model_load(es_context_t *ctx, const es_model_config_t *config, es_model_t **out_model) {
    if (!ctx || !config || !config->model_path || !out_model) return ES_ERR_INVALID_ARG;

    std::string path = config->model_path;
    std::string model_dir;
    std::string manifest_path;

    if (dir_exists(path)) {
        model_dir = path;
        manifest_path = path + "/manifest.json";
    } else if (file_exists(path)) {
        if (path.length() >= 5 && path.substr(path.length() - 5) == ".json") {
            manifest_path = path;
            size_t slash = path.find_last_of('/');
            model_dir = (slash != std::string::npos) ? path.substr(0, slash) : ".";
        } else {
            size_t slash = path.find_last_of('/');
            model_dir = (slash != std::string::npos) ? path.substr(0, slash) : ".";
        }
    } else {
        std::string cache = config->cache_dir ? config->cache_dir : (std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.cache/eswin/models");
        std::string cached_manifest = cache + "/" + path + "/manifest.json";
        if (file_exists(cached_manifest)) {
            model_dir = cache + "/" + path;
            manifest_path = cached_manifest;
        } else {
            return ES_ERR_IO;
        }
    }

    // Vendor ENNP packages ship config.json, not manifest.json. ModelManifest reads both
    // (root-level token_num, model_list/block_models), so prefer that over silently
    // falling back to hard-coded SmolLM defaults for a DeepSeek/Qwen2 package.
    if (manifest_path.empty() || !file_exists(manifest_path)) {
        std::string vendor_cfg = model_dir + "/config.json";
        if (file_exists(vendor_cfg)) {
            manifest_path = vendor_cfg;
        }
    }

    auto model = std::make_unique<es_model_t>();
    model->model_dir = model_dir;
    model->device_mode = config->device_mode;

    if (!manifest_path.empty() && file_exists(manifest_path)) {
        std::string err;
        if (!model->manifest.load_from_file(manifest_path, err)) {
            return ES_ERR_IO;
        }
    } else {
        model->manifest.name = "Custom NPU Model";
        model->manifest.config.architecture = "Transformer";
        model->manifest.config.vocab_size = 49152;
        model->manifest.config.hidden_size = 576;
        model->manifest.config.num_hidden_layers = 30;
        model->manifest.config.num_attention_heads = 9;
        model->manifest.config.num_key_value_heads = 3;
    }

    if (config->tokenizer_path && *config->tokenizer_path) {
        model->tokenizer_path = config->tokenizer_path;
    } else {
        const auto *tok_desc = model->manifest.find_file_by_role("tokenizer");
        if (tok_desc && !tok_desc->name.empty() && file_exists(model_dir + "/" + tok_desc->name)) {
            model->tokenizer_path = model_dir + "/" + tok_desc->name;
        } else if (file_exists(model_dir + "/tokenizer.json")) {
            model->tokenizer_path = model_dir + "/tokenizer.json";
        } else if (file_exists(model_dir + "/qwen.tiktoken")) {
            model->tokenizer_path = model_dir + "/qwen.tiktoken";
        } else if (file_exists(model_dir + "/vocab.json")) {
            model->tokenizer_path = model_dir + "/vocab.json";
        } else if (file_exists(model_dir + "/tokenizer.model")) {
            model->tokenizer_path = model_dir + "/tokenizer.model";
        } else if (file_exists("./qwen.tiktoken")) {
            model->tokenizer_path = "./qwen.tiktoken";
        } else if (file_exists("../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken")) {
            model->tokenizer_path = "../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken";
        } else if (file_exists("/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken")) {
            model->tokenizer_path = "/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken";
        }
    }

    if (!model->tokenizer_path.empty()) {
        es_tokenizer_t *tok = nullptr;
        if (es_tokenizer_load(model->tokenizer_path.c_str(), &tok) == ES_SUCCESS) {
            model->tokenizer_handle.reset(tok);
        }
    }

    *out_model = model.release();
    return ES_SUCCESS;
}

void es_model_unload(es_model_t *model) {
    if (model) {
        delete model;
    }
}

es_status_t es_model_get_info(const es_model_t *model, es_model_info_t *info) {
    if (!model || !info) return ES_ERR_INVALID_ARG;
    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, model->manifest.name.c_str(), sizeof(info->name) - 1);
    std::strncpy(info->architecture, model->manifest.config.architecture.c_str(), sizeof(info->architecture) - 1);
    info->vocab_size = model->manifest.config.vocab_size;
    info->hidden_size = model->manifest.config.hidden_size;
    info->num_layers = model->manifest.config.num_hidden_layers;
    info->num_heads = model->manifest.config.num_attention_heads;
    info->num_kv_heads = model->manifest.config.num_key_value_heads;
    info->max_seq_len = model->manifest.config.context_length;
    return ES_SUCCESS;
}

const es_tokenizer_t* es_model_get_tokenizer(const es_model_t *model) {
    if (!model) return nullptr;
    return model->tokenizer_handle.get();
}

es_status_t es_session_create(es_model_t *model, const es_session_config_t *config, es_session_t **out_session) {
    if (!model || !out_session) return ES_ERR_INVALID_ARG;

    auto sess_obj = std::make_unique<eswin::TransformerSession>();
    std::string err;
    if (!sess_obj->init(model->manifest, model->model_dir, model->tokenizer_path, err)) {
        return ES_ERR_HARDWARE_FAULT;
    }

    if (config) {
        eswin::SamplerParams sp;
        sp.temperature = config->sampler_params.temperature;
        sp.top_k = config->sampler_params.top_k;
        sp.top_p = config->sampler_params.top_p;
        sp.min_p = config->sampler_params.min_p;
        sp.repeat_penalty = config->sampler_params.repeat_penalty;
        sp.repeat_window = config->sampler_params.repeat_window;
        sp.max_consecutive_duplicates = config->sampler_params.max_consecutive_duplicates;
        sp.seed = config->sampler_params.seed;
        sess_obj->set_sampler_params(sp);
    }

    auto sess_wrapper = std::make_unique<es_session_t>();
    sess_wrapper->model = model;
    if (config) {
        sess_wrapper->config = *config;
    } else {
        sess_wrapper->config.max_seq_len = model->manifest.config.context_length;
        sess_wrapper->config.max_new_tokens = 512;
        sess_wrapper->config.enable_dual_die = true;
        sess_wrapper->config.sampler_params.temperature = 0.7f;
        sess_wrapper->config.sampler_params.top_k = 40;
        sess_wrapper->config.sampler_params.top_p = 0.9f;
        sess_wrapper->config.sampler_params.repeat_penalty = 1.15f;
    }
    sess_wrapper->session = std::move(sess_obj);
    *out_session = sess_wrapper.release();
    return ES_SUCCESS;
}

void es_session_free(es_session_t *session) {
    if (session) {
        delete session;
    }
}

es_status_t es_session_reset(es_session_t *session) {
    if (!session || !session->session) return ES_ERR_INVALID_ARG;
    session->session->reset();
    return ES_SUCCESS;
}

es_status_t es_session_generate(
    es_session_t *session,
    const char *prompt,
    es_token_callback_t callback,
    void *user_data,
    es_perf_stats_t *stats
) {
    if (!session || !session->session || !prompt) return ES_ERR_INVALID_ARG;

    eswin::SessionStats s_stats;
    int max_tokens = session->config.max_new_tokens > 0 ? session->config.max_new_tokens : 512;

    auto cb_wrapper = [callback, user_data](const std::string &piece, int32_t token_id) -> bool {
        if (!callback) return true;
        return callback(token_id, piece.c_str(), user_data);
    };

    bool ok = session->session->generate(prompt, max_tokens, cb_wrapper, &s_stats);
    if (!ok) return ES_ERR_HARDWARE_FAULT;

    if (stats) {
        stats->prompt_tokens = s_stats.prompt_tokens;
        stats->eval_tokens = s_stats.eval_tokens;
        stats->prompt_eval_ms = s_stats.prompt_eval_ms;
        stats->eval_ms = s_stats.eval_ms;
        stats->tokens_per_sec = s_stats.tokens_per_sec;
    }

    return ES_SUCCESS;
}

es_status_t es_session_perplexity(
    es_session_t *session,
    const char *text,
    double *perplexity,
    double *mean_logprob
) {
    if (!session || !session->session || !text) return ES_ERR_INVALID_ARG;

    // Score the raw text as given: Qwen2/DeepSeek prompts already carry their ChatML
    // delimiters, and adding a BOS here would shift every position by one.
    const bool add_bos =
        (session->session->manifest().config.architecture != "qwen2" &&
         std::strstr(text, "<|im_start|>") == nullptr);
    std::vector<int32_t> tokens = session->session->tokenizer().encode(text, add_bos);
    if (tokens.size() < 2) return ES_ERR_INVALID_ARG;

    if (!session->session->score_sequence(tokens, mean_logprob, perplexity)) {
        return ES_ERR_INVALID_ARG;
    }
    return ES_SUCCESS;
}


/* =========================================================================
 * File: dma_buffer.cpp
 * ========================================================================= */

/**
 * @file dma_buffer.cpp
 * @brief Implementation of RAII Contiguous DMA Buffer Allocator.
 */

#include "dma_buffer.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace eswin {

static constexpr size_t PAGE_ALIGN = 4096;

static inline size_t align_to_page(size_t s) {
    return (s + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
}

DmaBuffer::DmaBuffer(size_t size, CacheMode cache_mode, const char *zone)
    : m_size(align_to_page(size > 0 ? size : PAGE_ALIGN)),
      m_cache_mode(cache_mode)
{
    auto &driver = EswinDriver::instance();

    if (driver.is_hardware_available()) {
        int ret = driver.mem_alloc(&m_mem_fd, m_size, m_cache_mode, zone);
        if (ret == 0 && m_mem_fd != 0) {
            m_vaddr = driver.mem_mmap(m_mem_fd, m_size, m_cache_mode);
            if (m_vaddr) {
                m_dev_buf.mem_fd = m_mem_fd;
                m_dev_buf.offset = 0;
                m_dev_buf.size = m_size;
                m_dev_buf.reserved = 0;
                m_is_fallback = false;
                std::memset(m_vaddr, 0, m_size);
                return;
            }
            // Mmap failed, release memory
            driver.mem_free(m_mem_fd);
            m_mem_fd = 0;
        }
    }

    // Host development / fallback path
    void *ptr = nullptr;
    if (posix_memalign(&ptr, PAGE_ALIGN, m_size) == 0 && ptr != nullptr) {
        m_vaddr = ptr;
        m_mem_fd = 0xDEADBEEF; // Mock file descriptor
        m_dev_buf.mem_fd = m_mem_fd;
        m_dev_buf.offset = 0;
        m_dev_buf.size = m_size;
        m_dev_buf.reserved = 0;
        m_is_fallback = true;
        std::memset(m_vaddr, 0, m_size);
    } else {
        m_vaddr = nullptr;
        m_size = 0;
        throw std::bad_alloc();
    }
}

DmaBuffer::~DmaBuffer() {
    release();
}

void DmaBuffer::release() {
    if (m_is_prepared) {
        unprepare(m_prepared_device);
    }

    if (m_vaddr) {
        if (!m_is_fallback) {
            auto &driver = EswinDriver::instance();
            driver.mem_munmap(m_vaddr, m_size);
            if (m_mem_fd != 0) {
                driver.mem_free(m_mem_fd);
            }
        } else {
            std::free(m_vaddr);
        }
        m_vaddr = nullptr;
    }

    m_mem_fd = 0;
    m_size = 0;
    m_is_prepared = false;
    m_prepared_device = 0;
    m_is_fallback = false;
}

DmaBuffer::DmaBuffer(DmaBuffer &&other) noexcept
    : m_mem_fd(other.m_mem_fd),
      m_vaddr(other.m_vaddr),
      m_size(other.m_size),
      m_cache_mode(other.m_cache_mode),
      m_is_prepared(other.m_is_prepared),
      m_prepared_device(other.m_prepared_device),
      m_is_fallback(other.m_is_fallback),
      m_dev_buf(other.m_dev_buf)
{
    other.m_mem_fd = 0;
    other.m_vaddr = nullptr;
    other.m_size = 0;
    other.m_is_prepared = false;
    other.m_prepared_device = 0;
    other.m_is_fallback = false;
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer &&other) noexcept {
    if (this != &other) {
        release();

        m_mem_fd = other.m_mem_fd;
        m_vaddr = other.m_vaddr;
        m_size = other.m_size;
        m_cache_mode = other.m_cache_mode;
        m_is_prepared = other.m_is_prepared;
        m_prepared_device = other.m_prepared_device;
        m_is_fallback = other.m_is_fallback;
        m_dev_buf = other.m_dev_buf;

        other.m_mem_fd = 0;
        other.m_vaddr = nullptr;
        other.m_size = 0;
        other.m_is_prepared = false;
        other.m_prepared_device = 0;
        other.m_is_fallback = false;
    }
    return *this;
}

bool DmaBuffer::flush_cache() {
    if (!m_vaddr) return false;
    if (m_is_fallback) return true; // Host memory is already coherent

    if (m_cache_mode == CacheMode::Cached) {
        return (EswinDriver::instance().mem_flush_cache(m_mem_fd) == 0);
    }
    return true;
}

bool DmaBuffer::prepare(uint16_t device_id) {
    if (!m_vaddr || m_is_prepared) return m_is_prepared;
    if (m_is_fallback) {
        m_is_prepared = true;
        m_prepared_device = device_id;
        return true;
    }

    auto &driver = EswinDriver::instance();
    // ES_NPU_Prepare requires an active device context set on the thread
    driver.set_device(device_id);

    int ret = driver.prepare_dev_buf(&m_dev_buf);
    if (ret == 0) {
        m_is_prepared = true;
        m_prepared_device = device_id;
        return true;
    }
    std::cerr << "DmaBuffer::prepare failed on device " << device_id 
              << ": " << driver.decode_error(ret) << std::endl;
    return false;
}

bool DmaBuffer::unprepare(uint16_t device_id) {
    if (!m_is_prepared) return true;
    if (m_is_fallback) {
        m_is_prepared = false;
        m_prepared_device = 0;
        return true;
    }

    auto &driver = EswinDriver::instance();
    driver.set_device(device_id);

    int ret = driver.unprepare_dev_buf(&m_dev_buf);
    if (ret == 0) {
        m_is_prepared = false;
        m_prepared_device = 0;
        return true;
    }
    std::cerr << "DmaBuffer::unprepare failed on device " << device_id 
              << ": " << driver.decode_error(ret) << std::endl;
    return false;
}

} // namespace eswin


/* =========================================================================
 * File: driver_wrapper.cpp
 * ========================================================================= */

/**
 * @file driver_wrapper.cpp
 * @brief Dynamic loader implementation and error decoder for ESWIN NPU/MMZ drivers.
 */

#include "driver_wrapper.hpp"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace eswin {

EswinDriver& EswinDriver::instance() {
    static EswinDriver s_instance;
    return s_instance;
}

EswinDriver::EswinDriver() = default;

EswinDriver::~EswinDriver() {
    shutdown();
}

void* EswinDriver::try_dlopen(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            return handle;
        }
    }
    return nullptr;
}

bool EswinDriver::load_libraries() {
    // 1. Load NPU Runtime (libedla_runtime.so)
    const std::vector<std::string> npu_paths = {
        "/lib/libedla_runtime.so.1.0",
        "/usr/lib/libedla_runtime.so.1.0",
        "/opt/eswin/lib/libedla_runtime.so.1.0",
        "libedla_runtime.so.1.0",
        "libedla_runtime.so"
    };
    m_lib_npu = try_dlopen(npu_paths);

    // 2. Load Memory Driver (libes_memory.so)
    const std::vector<std::string> mem_paths = {
        "/lib/libes_memory.so",
        "/usr/lib/libes_memory.so",
        "/opt/eswin/lib/libes_memory.so",
        "libes_memory.so"
    };
    m_lib_mem = try_dlopen(mem_paths);

    // 3. Load System Services (libes_sys.so.1)
    const std::vector<std::string> sys_paths = {
        "/lib/libes_sys.so.1",
        "/usr/lib/libes_sys.so.1",
        "/opt/eswin/lib/libes_sys.so.1",
        "libes_sys.so.1",
        "libes_sys.so"
    };
    m_lib_sys = try_dlopen(sys_paths);

    if (!m_lib_npu || !m_lib_mem) {
        return false;
    }

    // Resolve NPU functions
    m_pGetNumDevices  = (fn_ES_NPU_GetNumDevices)dlsym(m_lib_npu, "ES_NPU_GetNumDevices");
    m_pSetDevice      = (fn_ES_NPU_SetDevice)dlsym(m_lib_npu, "ES_NPU_SetDevice");
    m_pReleaseDevice  = (fn_ES_NPU_ReleaseDevice)dlsym(m_lib_npu, "ES_NPU_ReleaseDevice");
    m_pGetDevice      = (fn_ES_NPU_GetDevice)dlsym(m_lib_npu, "ES_NPU_GetDevice");
    m_pCreateContext  = (fn_ES_NPU_CreateContext)dlsym(m_lib_npu, "ES_NPU_CreateContext");
    m_pDestroyContext = (fn_ES_NPU_DestroyContext)dlsym(m_lib_npu, "ES_NPU_DestroyContext");
    m_pCreateStream   = (fn_ES_NPU_CreateStream)dlsym(m_lib_npu, "ES_NPU_CreateStream");
    m_pDestroyStream  = (fn_ES_NPU_DestroyStream)dlsym(m_lib_npu, "ES_NPU_DestroyStream");
    m_pSyncStream     = (fn_ES_NPU_SynchronizeStream)dlsym(m_lib_npu, "ES_NPU_SynchronizeStream");
    m_pLoadModel      = (fn_ES_NPU_LoadModelFromFile)dlsym(m_lib_npu, "ES_NPU_LoadModelFromFile");
    m_pUnloadModel    = (fn_ES_NPU_UnloadModel)dlsym(m_lib_npu, "ES_NPU_UnloadModel");
    m_pGetNumInputTensors  = (fn_ES_NPU_GetNumInputTensors)dlsym(m_lib_npu, "ES_NPU_GetNumInputTensors");
    m_pGetNumOutputTensors = (fn_ES_NPU_GetNumOutputTensors)dlsym(m_lib_npu, "ES_NPU_GetNumOutputTensors");
    m_pGetInputTensorDesc  = (fn_ES_NPU_GetInputTensorDesc)dlsym(m_lib_npu, "ES_NPU_GetInputTensorDesc");
    m_pGetOutputTensorDesc = (fn_ES_NPU_GetOutputTensorDesc)dlsym(m_lib_npu, "ES_NPU_GetOutputTensorDesc");
    m_pPrepare        = (fn_ES_NPU_Prepare)dlsym(m_lib_npu, "ES_NPU_Prepare");
    m_pUnprepare      = (fn_ES_NPU_Unprepare)dlsym(m_lib_npu, "ES_NPU_Unprepare");
    m_pSubmit         = (fn_ES_NPU_Submit)dlsym(m_lib_npu, "ES_NPU_Submit");
    m_pSubmitFlexible = (fn_ES_NPU_SubmitFlexible)dlsym(m_lib_npu, "ES_NPU_SubmitFlexibleTask");

    // Resolve Memory functions
    m_pMemAlloc       = (fn_ES_SYS_MemAlloc)dlsym(m_lib_mem, "ES_SYS_MemAlloc");
    m_pMemFree        = (fn_ES_SYS_MemFree)dlsym(m_lib_mem, "ES_SYS_MemFree");
    m_pMmap           = (fn_ES_SYS_Mmap)dlsym(m_lib_mem, "ES_SYS_Mmap");
    m_pMunmap         = (fn_ES_SYS_Munmap)dlsym(m_lib_mem, "ES_SYS_Munmap");
    m_pMemFlushCache  = (fn_ES_SYS_MemFlushCache)dlsym(m_lib_mem, "ES_SYS_MemFlushCache");
    m_pRcvMemFd       = (fn_ES_SYS_RcvMemFd)dlsym(m_lib_mem, "ES_SYS_RcvMemFd");

    // Resolve System functions if available
    if (m_lib_sys) {
        m_pSysInit = (fn_ES_SYS_Init)dlsym(m_lib_sys, "ES_SYS_Init");
        m_pSysExit = (fn_ES_SYS_Exit)dlsym(m_lib_sys, "ES_SYS_Exit");
    }

    return (m_pGetNumDevices && m_pSetDevice && m_pLoadModel && m_pMemAlloc && m_pMmap);
}

bool EswinDriver::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized) {
        return true;
    }

    if (!load_libraries()) {
        m_hardware_available = false;
        m_initialized = true;
        return false;
    }

    // 1. Initialize subsystem if ES_SYS_Init is present
    if (m_pSysInit) {
        m_pSysInit();
    }

    // 2. Query device count (CRITICAL: initializes internal device manager table)
    uint16_t count = 0;
    if (m_pGetNumDevices && m_pGetNumDevices(&count) == 0) {
        m_num_devices = count;
        m_hardware_available = (m_num_devices > 0);
    } else {
        m_hardware_available = false;
    }

    m_initialized = true;
    return m_hardware_available;
}

void EswinDriver::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_initialized) return;

    if (m_pSysExit) {
        m_pSysExit();
    }

    if (m_lib_npu) { dlclose(m_lib_npu); m_lib_npu = nullptr; }
    if (m_lib_mem) { dlclose(m_lib_mem); m_lib_mem = nullptr; }
    if (m_lib_sys) { dlclose(m_lib_sys); m_lib_sys = nullptr; }

    m_initialized = false;
    m_hardware_available = false;
    m_num_devices = 0;
}

int EswinDriver::set_device(uint16_t device_id) {
    if (!m_hardware_available || !m_pSetDevice) return -1;
    return m_pSetDevice(device_id);
}

int EswinDriver::release_device(uint16_t device_id) {
    if (!m_hardware_available || !m_pReleaseDevice) return -1;
    return m_pReleaseDevice(device_id);
}

int EswinDriver::create_context(void **context, uint16_t device_id) {
    if (!m_hardware_available || !m_pCreateContext) return -1;
    return m_pCreateContext(context, device_id);
}

int EswinDriver::destroy_context(void *context) {
    if (!m_hardware_available || !m_pDestroyContext) return -1;
    return m_pDestroyContext(context);
}

int EswinDriver::create_stream(void **stream) {
    if (!m_hardware_available || !m_pCreateStream) return -1;
    return m_pCreateStream(stream);
}

int EswinDriver::destroy_stream(void *stream) {
    if (!m_hardware_available || !m_pDestroyStream) return -1;
    return m_pDestroyStream(stream);
}

int EswinDriver::sync_stream(void *stream) {
    if (!m_hardware_available || !m_pSyncStream) return -1;
    return m_pSyncStream(stream);
}

int EswinDriver::load_model(uint32_t *model_id, const std::string &path) {
    if (!m_hardware_available || !m_pLoadModel) return -1;
    return m_pLoadModel(model_id, path.c_str());
}

int EswinDriver::unload_model(uint32_t model_id) {
    if (!m_hardware_available || !m_pUnloadModel) return -1;
    return m_pUnloadModel(model_id);
}

int EswinDriver::get_num_input_tensors(uint32_t model_id, uint32_t *num) {
    if (!m_hardware_available || !m_pGetNumInputTensors) return -1;
    return m_pGetNumInputTensors(model_id, num);
}

int EswinDriver::get_num_output_tensors(uint32_t model_id, uint32_t *num) {
    if (!m_hardware_available || !m_pGetNumOutputTensors) return -1;
    return m_pGetNumOutputTensors(model_id, num);
}

int EswinDriver::get_input_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc) {
    if (!m_hardware_available || !m_pGetInputTensorDesc) return -1;
    return m_pGetInputTensorDesc(model_id, tensor_id, desc);
}

int EswinDriver::get_output_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc) {
    if (!m_hardware_available || !m_pGetOutputTensorDesc) return -1;
    return m_pGetOutputTensorDesc(model_id, tensor_id, desc);
}

int EswinDriver::mem_alloc(uint64_t *mem_fd, size_t size, CacheMode cache_mode, const char *zone) {
    if (!m_hardware_available || !m_pMemAlloc) return -1;
    return m_pMemAlloc(mem_fd, static_cast<int>(cache_mode), "es_runtime", zone, size);
}

int EswinDriver::mem_free(uint64_t mem_fd) {
    if (!m_hardware_available || !m_pMemFree) return -1;
    return m_pMemFree(mem_fd);
}

int EswinDriver::mem_rcv_fd(uint64_t snd_mem_fd, uint64_t *rcvd_mem_fd) {
    if (!m_hardware_available || !m_pRcvMemFd) return -1;
    return m_pRcvMemFd(snd_mem_fd, rcvd_mem_fd);
}

void* EswinDriver::mem_mmap(uint64_t mem_fd, size_t size, CacheMode cache_mode) {
    if (!m_hardware_available || !m_pMmap) return nullptr;
    return m_pMmap(mem_fd, size, static_cast<int>(cache_mode));
}

int EswinDriver::mem_munmap(void *vaddr, size_t size) {
    if (!m_hardware_available || !m_pMunmap) return -1;
    return m_pMunmap(vaddr, size);
}

int EswinDriver::mem_flush_cache(uint64_t mem_fd) {
    if (!m_hardware_available || !m_pMemFlushCache) return -1;
    return m_pMemFlushCache(mem_fd);
}

int EswinDriver::prepare_dev_buf(DevBuf *buf) {
    if (!m_hardware_available || !m_pPrepare) return -1;
    return m_pPrepare(buf);
}

int EswinDriver::unprepare_dev_buf(DevBuf *buf) {
    if (!m_hardware_available || !m_pUnprepare) return -1;
    return m_pUnprepare(buf);
}

int EswinDriver::submit_task(NpuTask *task, uint32_t num_tasks) {
    if (!m_hardware_available || !m_pSubmit) return -1;
    return m_pSubmit(task, num_tasks);
}

int EswinDriver::submit_flexible_task(NpuTask *tasks, uint32_t num_tasks, void *stream) {
    if (!m_hardware_available || !m_pSubmitFlexible) return -1;
    return m_pSubmitFlexible(tasks, num_tasks, stream);
}

std::string EswinDriver::decode_error(int error_code) {
    if (error_code == 0) return "SUCCESS (0)";

    uint32_t err = static_cast<uint32_t>(error_code);
    uint32_t mod_id = (err >> 16) & 0xFF;
    uint32_t code   = err & 0x1FFF;

    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(8) << std::setfill('0') << err
        << " (Mod: " << std::dec << mod_id << ", ErrId: " << code << "): ";

    // Known specific ESWIN error mappings
    if (mod_id == 15) { // NPU Module
        switch (code) {
            case 3:  oss << "ES_NPU_ERROR_ILLEGAL_PARAM (Invalid model parameter or shape)"; break;
            case 6:  oss << "ES_NPU_ERROR_NULL_PTR (Null pointer passed to API)"; break;
            case 12: oss << "ES_NPU_ERROR_NOMEM (NPU internal memory allocation failure)"; break;
            case 18: oss << "ES_NPU_ERROR_BUSY (Device or engine is busy)"; break;
            case 64: oss << "ES_NPU_ERROR_NOT_INIT (NPU subsystem not initialized)"; break;
            case 69: oss << "ES_NPU_ERROR_DEV_NOT_FOUND (Device index not found)"; break;
            case 70: oss << "ES_NPU_ERROR_DEV_MULTI_SET (Die collision: process already bound a device)"; break;
            case 71: oss << "ES_NPU_ERROR_DEV_OPEN_FAILED (Failed to open /dev/npu*)"; break;
            case 72: oss << "ES_NPU_ERROR_MODEL_NOT_PRESENT (Model uninitialized or device context missing)"; break;
            case 73: oss << "ES_NPU_ERROR_MODEL_NOT_FOUND (Model ID not found)"; break;
            case 85: oss << "ES_NPU_ERROR_IOCTL_FAIL (Driver ioctl call failed)"; break;
            case 88: oss << "ES_NPU_ERROR_LOAD_OP_FAILED (DSP/NPU kernel loader failure)"; break;
            default: oss << "Unknown NPU Error"; break;
        }
    } else if (mod_id == 2) { // SYS Module
        switch (code) {
            case 12: oss << "ES_SYS_ERROR_NOMEM (MMZ out of contiguous DMA memory)"; break;
            case 17: oss << "ES_SYS_ERROR_BADADDR (Invalid memory address or unmapped DMA buffer)"; break;
            default: oss << "System/Memory Subsystem Error"; break;
        }
    } else {
        oss << "General ESWIN Subsystem Error";
    }

    return oss.str();
}

} // namespace eswin


/* =========================================================================
 * File: dsp_driver.cpp
 * ========================================================================= */

/**
 * @file dsp_driver.cpp
 * @brief Dynamic loader and hardware implementation for Cadence Vision Q7 DSP execution.
 */

#include "dsp_driver.hpp"

#include <dlfcn.h>
#include <iostream>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

#define BUFFER_CNT_MAXSIZE 64
#define SIZE_2M 0x200000
#define ALIGN_2M(size) (((size) + (SIZE_2M - 1)) & ~(SIZE_2M - 1))

#pragma pack(push, 8)
struct RawDevBuf {
    uint64_t memFd;
    uint64_t offset;
    uint64_t size;
    uint64_t reserved;
};

struct RawDspTask {
    uint32_t operatorHandle;
    RawDevBuf dspBuffers[BUFFER_CNT_MAXSIZE];
    uint32_t bufferCntCfg;
    uint32_t bufferCntInput;
    uint32_t bufferCntOutput;
    uint32_t priority;
    uint32_t syncCache;
    uint32_t pollMode;
    uint32_t taskHandle;
    void* callback;
    void* cbArg;
    uint32_t reserved;
};
#pragma pack(pop)

using fn_ES_DSP_LL_GetVersion = int (*)(uint64_t* version);
using fn_ES_DSP_LL_GetVersionString = int (*)(char* str, uint32_t maxLen);
using fn_ES_DSP_LL_Open = int (*)(int dspId, int* dspFd);
using fn_ES_DSP_LL_Close = int (*)(int dspFd);
using fn_ES_DSP_LL_LoadOperator = int (*)(int dspFd, const char* opName, const char* opDir, uint32_t* handle);
using fn_ES_DSP_LL_UnloadOperator = int (*)(int dspFd, uint32_t handle);
using fn_ES_DSP_LL_PrepareDMABuffer = int (*)(int dspFd, RawDevBuf buf);
using fn_ES_DSP_LL_UnprepareDMABuffer = int (*)(int dspFd, uint64_t memFd);
using fn_ES_DSP_LL_SubmitTask = int (*)(int dspFd, RawDspTask* task);

using fn_ES_SYS_MemAlloc = int (*)(uint64_t* memFd, int cacheMode, const char* name, const char* zone, size_t size);
using fn_ES_SYS_MemFree = int (*)(uint64_t memFd);
using fn_ES_SYS_Mmap = void* (*)(uint64_t memFd, size_t size, int cacheMode);
using fn_ES_SYS_Munmap = int (*)(void* ptr, size_t size);

struct DspSymbols {
    fn_ES_DSP_LL_GetVersion GetVersion{nullptr};
    fn_ES_DSP_LL_GetVersionString GetVersionString{nullptr};
    fn_ES_DSP_LL_Open Open{nullptr};
    fn_ES_DSP_LL_Close Close{nullptr};
    fn_ES_DSP_LL_LoadOperator LoadOperator{nullptr};
    fn_ES_DSP_LL_UnloadOperator UnloadOperator{nullptr};
    fn_ES_DSP_LL_PrepareDMABuffer PrepareDMABuffer{nullptr};
    fn_ES_DSP_LL_UnprepareDMABuffer UnprepareDMABuffer{nullptr};
    fn_ES_DSP_LL_SubmitTask SubmitTask{nullptr};

    fn_ES_SYS_MemAlloc MemAlloc{nullptr};
    fn_ES_SYS_MemFree MemFree{nullptr};
    fn_ES_SYS_Mmap Mmap{nullptr};
    fn_ES_SYS_Munmap Munmap{nullptr};
};

static DspSymbols g_syms;
static bool g_syms_loaded = false;

void init_dsp_symbols() {
    if (g_syms_loaded) return;

    void* dsp_lib = dlopen("libes_dsp_runtime_ll.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dsp_lib) {
        dsp_lib = dlopen("/usr/lib/libes_dsp_runtime_ll.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!dsp_lib) {
        throw std::runtime_error(std::string("Failed to load libes_dsp_runtime_ll.so: ") + dlerror());
    }

    void* mem_lib = dlopen("libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
    if (!mem_lib) {
        mem_lib = dlopen("/usr/lib/libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!mem_lib) {
        throw std::runtime_error(std::string("Failed to load libes_memory.so: ") + dlerror());
    }

    g_syms.GetVersion = reinterpret_cast<fn_ES_DSP_LL_GetVersion>(dlsym(dsp_lib, "ES_DSP_LL_GetVersion"));
    g_syms.GetVersionString = reinterpret_cast<fn_ES_DSP_LL_GetVersionString>(dlsym(dsp_lib, "ES_DSP_LL_GetVersionString"));
    g_syms.Open = reinterpret_cast<fn_ES_DSP_LL_Open>(dlsym(dsp_lib, "ES_DSP_LL_Open"));
    g_syms.Close = reinterpret_cast<fn_ES_DSP_LL_Close>(dlsym(dsp_lib, "ES_DSP_LL_Close"));
    g_syms.LoadOperator = reinterpret_cast<fn_ES_DSP_LL_LoadOperator>(dlsym(dsp_lib, "ES_DSP_LL_LoadOperator"));
    g_syms.UnloadOperator = reinterpret_cast<fn_ES_DSP_LL_UnloadOperator>(dlsym(dsp_lib, "ES_DSP_LL_UnloadOperator"));
    g_syms.PrepareDMABuffer = reinterpret_cast<fn_ES_DSP_LL_PrepareDMABuffer>(dlsym(dsp_lib, "ES_DSP_LL_PrepareDMABuffer"));
    g_syms.UnprepareDMABuffer = reinterpret_cast<fn_ES_DSP_LL_UnprepareDMABuffer>(dlsym(dsp_lib, "ES_DSP_LL_UnprepareDMABuffer"));
    g_syms.SubmitTask = reinterpret_cast<fn_ES_DSP_LL_SubmitTask>(dlsym(dsp_lib, "ES_DSP_LL_SubmitTask"));

    g_syms.MemAlloc = reinterpret_cast<fn_ES_SYS_MemAlloc>(dlsym(mem_lib, "ES_SYS_MemAlloc"));
    g_syms.MemFree = reinterpret_cast<fn_ES_SYS_MemFree>(dlsym(mem_lib, "ES_SYS_MemFree"));
    g_syms.Mmap = reinterpret_cast<fn_ES_SYS_Mmap>(dlsym(mem_lib, "ES_SYS_Mmap"));
    g_syms.Munmap = reinterpret_cast<fn_ES_SYS_Munmap>(dlsym(mem_lib, "ES_SYS_Munmap"));

    g_syms_loaded = true;
}

} // namespace

DspContext::DspContext(DspCore core)
    : core_(core), dsp_fd_(-1) {
    init_dsp_symbols();
}

DspContext::~DspContext() {
    close();
}

void DspContext::open() {
    if (is_open()) return;
    int fd = -1;
    int ret = g_syms.Open(static_cast<int>(core_), &fd);
    if (ret != 0 || fd < 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_Open failed for core " << static_cast<int>(core_)
            << " with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
    dsp_fd_ = fd;
}

void DspContext::close() {
    if (is_open()) {
        g_syms.Close(dsp_fd_);
        dsp_fd_ = -1;
    }
}

uint32_t DspContext::load_operator(const std::string& op_name, const std::string& op_dir) {
    if (!is_open()) open();
    uint32_t handle = 0;
    int ret = g_syms.LoadOperator(dsp_fd_, op_name.c_str(), op_dir.c_str(), &handle);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_LoadOperator('" << op_name << "') failed with code 0x"
            << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
    return handle;
}

void DspContext::unload_operator(uint32_t handle) {
    if (!is_open() || handle == 0) return;
    g_syms.UnloadOperator(dsp_fd_, handle);
}

DspBuffer DspContext::allocate_buffer(size_t size) {
    DspBuffer buf;
    buf.size = size;
    buf.alloc_size = ALIGN_2M(size);
    buf.offset = 0;

    int ret = g_syms.MemAlloc(&buf.mem_fd, 0, "dsp_runtime", nullptr, buf.alloc_size);
    if (ret != 0) {
        throw std::runtime_error("ES_SYS_MemAlloc failed for DSP buffer");
    }

    buf.host_ptr = g_syms.Mmap(buf.mem_fd, buf.alloc_size, 0);
    if (!buf.host_ptr) {
        g_syms.MemFree(buf.mem_fd);
        throw std::runtime_error("ES_SYS_Mmap failed for DSP buffer");
    }

    return buf;
}

void DspContext::free_buffer(DspBuffer& buf) {
    if (buf.host_ptr && buf.alloc_size > 0) {
        g_syms.Munmap(buf.host_ptr, buf.alloc_size);
        buf.host_ptr = nullptr;
    }
    if (buf.mem_fd != 0) {
        g_syms.MemFree(buf.mem_fd);
        buf.mem_fd = 0;
    }
}

void DspContext::prepare_dma(const DspBuffer& buf) {
    if (!is_open()) open();
    RawDevBuf dev_buf;
    dev_buf.memFd = buf.mem_fd;
    dev_buf.offset = buf.offset;
    dev_buf.size = buf.size;
    dev_buf.reserved = 0;

    int ret = g_syms.PrepareDMABuffer(dsp_fd_, dev_buf);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_PrepareDMABuffer failed with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
}

void DspContext::unprepare_dma(const DspBuffer& buf) {
    if (!is_open()) return;
    g_syms.UnprepareDMABuffer(dsp_fd_, buf.mem_fd);
}

void DspContext::submit_task(
    uint32_t op_handle,
    const std::vector<DspBuffer>& cfgs,
    const std::vector<DspBuffer>& inputs,
    const std::vector<DspBuffer>& outputs,
    bool sync_cache
) {
    if (!is_open()) open();

    RawDspTask task;
    std::memset(&task, 0, sizeof(task));
    task.operatorHandle = op_handle;
    task.bufferCntCfg = static_cast<uint32_t>(cfgs.size());
    task.bufferCntInput = static_cast<uint32_t>(inputs.size());
    task.bufferCntOutput = static_cast<uint32_t>(outputs.size());
    task.syncCache = sync_cache ? 1 : 0;
    task.pollMode = 0;

    int idx = 0;
    for (const auto& b : cfgs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }
    for (const auto& b : inputs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }
    for (const auto& b : outputs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }

    int ret = g_syms.SubmitTask(dsp_fd_, &task);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_SubmitTask failed with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
}

std::string DspContext::get_version_string() {
    char buf[128] = {0};
    if (g_syms.GetVersionString) {
        g_syms.GetVersionString(buf, sizeof(buf));
    }
    return std::string(buf);
}

uint64_t DspContext::get_version() {
    uint64_t ver = 0;
    if (g_syms.GetVersion) {
        g_syms.GetVersion(&ver);
    }
    return ver;
}

} // namespace eswin


/* =========================================================================
 * File: direct_kernel_backend.cpp
 * ========================================================================= */

/**
 * @file direct_kernel_backend.cpp
 * @brief Layer 0' implementation: dma-heap allocation and /dev/npu* ioctls (Milestone 11).
 *
 * Nothing here calls dlopen. Allocation goes through the standard Linux dma-heap uAPI on
 * /dev/dma_heap/mmz_nid_<die>_part_0, mapping through mmap, and SMMU binding through the
 * eic7700_npu.ko ioctl dispatch table. That is what makes the path blob-free.
 *
 * Cache maintenance uses the standard DMA_BUF_IOCTL_SYNC rather than the vendor
 * ES_SYS_MemFlushCache, so CPU writes are published to the device the same way on any
 * dma-buf exporter.
 */

#include "direct_kernel_backend.hpp"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

constexpr size_t kPageAlign = 4096;

// Linux dma-buf synchronization uAPI (dma-buf.h). Magic is 'H', same letter as dma-heap
// but a different command number.
constexpr uint32_t kDmaBufIoctlMagic = 0x48;
struct DmaBufSync {
    uint64_t flags;
};
static_assert(sizeof(DmaBufSync) == 8, "dma_buf_sync layout");

constexpr uint64_t kDmaBufSyncRead  = 1;
constexpr uint64_t kDmaBufSyncWrite = 2;
constexpr uint64_t kDmaBufSyncStart = 0;
constexpr uint64_t kDmaBufSyncEnd   = (1ull << 4);

// _IOW('H', 0, struct dma_buf_sync): write-only from userspace, 8-byte payload.
constexpr uint32_t kDmaBufIoctlSync =
    (static_cast<uint32_t>(kIocWrite) << 30) |
    (static_cast<uint32_t>(sizeof(DmaBufSync)) << 16) |
    (static_cast<uint32_t>(kDmaBufIoctlMagic) << 8) | 0x00u;

size_t align_up(size_t n) {
    return (n + kPageAlign - 1) & ~(kPageAlign - 1);
}

} // namespace

const char *dma_heap_node(uint16_t die_id) {
    return (die_id == 1) ? "/dev/dma_heap/mmz_nid_1_part_0"
                         : "/dev/dma_heap/mmz_nid_0_part_0";
}

const char *npu_device_node(uint16_t die_id) {
    return (die_id == 1) ? "/dev/npu1" : "/dev/npu0";
}

void DirectDmaBuffer::release() {
    if (vaddr && vaddr != MAP_FAILED) {
        ::munmap(vaddr, size);
    }
    if (fd >= 0) {
        ::close(fd);
    }
    vaddr = nullptr;
    fd = -1;
    size = 0;
}

DirectKernelBackend::~DirectKernelBackend() {
    close();
}

int DirectKernelBackend::probe_device_count() {
    int count = 0;
    for (uint16_t die = 0; die < 2; ++die) {
        if (::access(npu_device_node(die), F_OK) == 0) ++count;
    }
    return count;
}

bool DirectKernelBackend::open(uint16_t die_id, std::string &err) {
    close();
    err.clear();

    if (die_id > 1) {
        err = "DirectKernelBackend: die_id must be 0 or 1";
        return false;
    }

    // O_RDWR: the driver needs write access for SMMU mapping. The nodes are 0660
    // root:root, so this fails with EACCES unless run as root or a udev rule relaxes
    // the mode - report that specifically, because it is the most common board failure.
    int fd = ::open(npu_device_node(die_id), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: cannot open " << npu_device_node(die_id) << ": "
           << std::strerror(errno);
        if (errno == EACCES) {
            os << " (nodes are 0660 root:root - use sudo or add "
                  "/etc/udev/rules.d/99-npu.rules)";
        }
        err = os.str();
        return false;
    }

    m_fd = fd;
    m_die_id = die_id;
    return true;
}

void DirectKernelBackend::close() {
    if (m_fd >= 0) {
        ::close(m_fd);
    }
    m_fd = -1;
    m_die_id = 0;
}

bool DirectKernelBackend::alloc_dma(size_t bytes, uint16_t die_id, DirectDmaBuffer &out,
                                    std::string &err) {
    out.release();
    err.clear();

    if (bytes == 0) {
        err = "DirectKernelBackend: refusing a zero-byte allocation";
        return false;
    }
    const size_t len = align_up(bytes);

    int heap_fd = ::open(dma_heap_node(die_id), O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: cannot open dma-heap " << dma_heap_node(die_id)
           << ": " << std::strerror(errno);
        err = os.str();
        return false;
    }

    DmaHeapAllocationData req{};
    req.len = len;
    req.fd = 0;
    req.fd_flags = kDmaHeapFdCloexec;
    req.heap_flags = 0;

    if (::ioctl(heap_fd, kDmaHeapIoctlAlloc, &req) != 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: DMA_HEAP_IOCTL_ALLOC of " << len << " bytes on "
           << dma_heap_node(die_id) << " failed: " << std::strerror(errno)
           << " (is the MMZ carveout large enough?)";
        err = os.str();
        ::close(heap_fd);
        return false;
    }
    ::close(heap_fd);   // the buffer fd outlives the heap fd

    void *va = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED,
                      static_cast<int>(req.fd), 0);
    if (va == MAP_FAILED) {
        std::ostringstream os;
        os << "DirectKernelBackend: mmap of the dma-buf failed: " << std::strerror(errno);
        err = os.str();
        ::close(static_cast<int>(req.fd));
        return false;
    }

    out.fd = static_cast<int>(req.fd);
    out.vaddr = va;
    out.size = len;
    out.die_id = die_id;
    return true;
}

int DirectKernelBackend::prepare_dma(const DirectDmaBuffer &buf) {
    if (m_fd < 0 || buf.fd < 0) return -1;

    // The command word encodes a 4-byte payload (see header), so pass the buffer fd as a
    // uint32_t rather than the 32-byte vendor ES_DEV_BUF. If the kernel actually wants
    // the full struct this returns EINVAL, which is the signal to revisit the RE.
    uint32_t payload = static_cast<uint32_t>(buf.fd);
    return ::ioctl(m_fd, kNpuIoctlPrepareDma, &payload);
}

int DirectKernelBackend::unprepare_dma(const DirectDmaBuffer &buf) {
    if (m_fd < 0 || buf.fd < 0) return -1;
    uint32_t payload = static_cast<uint32_t>(buf.fd);
    return ::ioctl(m_fd, kNpuIoctlUnprepareDma, &payload);
}

bool DirectKernelBackend::flush_cache(const DirectDmaBuffer &buf) {
    if (buf.fd < 0) return false;

    // Publish CPU writes to the device. START then END brackets the access window; the
    // exporter is free to no-op either when the heap is uncached.
    DmaBufSync start{kDmaBufSyncStart | kDmaBufSyncWrite};
    DmaBufSync end{kDmaBufSyncEnd | kDmaBufSyncWrite};
    if (::ioctl(buf.fd, kDmaBufIoctlSync, &start) != 0) return false;
    return ::ioctl(buf.fd, kDmaBufIoctlSync, &end) == 0;
}

} // namespace eswin


/* =========================================================================
 * File: pipeline_executor.cpp
 * ========================================================================= */

/**
 * @file pipeline_executor.cpp
 * @brief Zero-copy multi-layer block chaining engine (Milestone 7).
 *
 * See pipeline_executor.hpp for the architecture. Two implementation notes that are
 * easy to get wrong on this SoC:
 *
 *  1. Activation buffers are allocated CacheMode::Cached, not NoCache. The cross-die
 *     handoff is performed by the CPU through the mapped virtual addresses, and
 *     measured NoCache CPU reads collapse to 10-60 MB/s (single uncacheable bus
 *     beats) versus 1.74-2.89 GB/s cached. Every CPU write is followed by
 *     flush_cache() so the NPU sees it.
 *  2. Buffers are prepared on their own die only. Preparing a Die 0 buffer on Die 1
 *     wastes an SMMU mapping and, on the vendor driver, can leave stale IOVA entries
 *     behind after a fault.
 */

#include "pipeline_executor.hpp"
#include "driver_wrapper.hpp"

#include <chrono>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

const char *zone_for_die(uint16_t die_id) {
    return (die_id == 1) ? "mmz_nid_1_part_0" : "mmz_nid_0_part_0";
}

} // namespace

PipelineExecutor::PipelineExecutor(WorkerPool &pool)
    : m_pool(pool)
    , m_hardware(EswinDriver::instance().is_hardware_available()) {
    for (int s = 0; s < 2; ++s) {
        m_buf[s].resize(2);
        m_prepared[s].assign(2, false);
    }
}

PipelineExecutor::~PipelineExecutor() {
    release();
}

void PipelineExecutor::release() {
    for (int slot = 0; slot < 2; ++slot) {
        for (uint16_t die = 0; die < m_buf[slot].size(); ++die) {
            if (m_buf[slot][die]) {
                if (m_prepared[slot][die]) {
                    m_pool.unprepare_buffer(die, *m_buf[slot][die]);
                    m_prepared[slot][die] = false;
                }
                m_buf[slot][die].reset();
            }
        }
    }
    m_stages.clear();
    m_ready = false;
}

bool PipelineExecutor::init(const std::vector<PipelineStage> &stages, std::string &err) {
    if (stages.empty()) {
        err = "PipelineExecutor: stage list is empty";
        return false;
    }

    release();
    m_stages = stages;

    // Largest activation any stage on a given die needs; both ping-pong slots are the
    // same size so a stage's output buffer can always serve as the next stage's input.
    size_t die_bytes[2] = {0, 0};
    for (const auto &st : m_stages) {
        if (st.die_id > 1) {
            std::ostringstream os;
            os << "PipelineExecutor: stage '" << st.label << "' has invalid die_id "
               << st.die_id;
            err = os.str();
            release();
            return false;
        }
        size_t need = st.in_bytes > st.out_bytes ? st.in_bytes : st.out_bytes;
        if (need > die_bytes[st.die_id]) die_bytes[st.die_id] = need;
    }

    for (uint16_t die = 0; die < 2; ++die) {
        if (die_bytes[die] == 0) continue;
        for (int slot = 0; slot < 2; ++slot) {
            auto buf = std::make_unique<DmaBuffer>(die_bytes[die], CacheMode::Cached,
                                                   zone_for_die(die));
            if (!buf->is_valid()) {
                err = "PipelineExecutor: failed to allocate activation buffer (die " +
                      std::to_string(die) + ", slot " + std::to_string(slot) + ", " +
                      std::to_string(die_bytes[die]) + " bytes)";
                release();
                return false;
            }
            if (!m_pool.prepare_buffer(die, *buf)) {
                err = "PipelineExecutor: SMMU prepare failed for die " +
                      std::to_string(die) + " slot " + std::to_string(slot);
                release();
                return false;
            }
            m_prepared[slot][die] = true;
            m_buf[slot][die] = std::move(buf);
        }
    }

    m_ready = true;
    return true;
}

bool PipelineExecutor::copy_across_dies(const DmaBuffer &src, uint16_t src_die,
                                        DmaBuffer &dst, uint16_t dst_die,
                                        size_t bytes, uint64_t *elapsed_ns) {
    (void)src_die;
    (void)dst_die;
    if (!src.is_valid() || !dst.is_valid() || bytes == 0) return false;
    size_t n = bytes < src.size() ? bytes : src.size();
    if (n > dst.size()) n = dst.size();

    uint64_t t0 = now_ns();
    std::memcpy(dst.data(), src.data(), n);
    // CPU wrote the destination: make it visible to the receiving die's NPU.
    bool flushed = dst.flush_cache();
    if (elapsed_ns) *elapsed_ns = now_ns() - t0;
    return flushed;
}

bool PipelineExecutor::run(const void *seed, size_t seed_bytes,
                           void *out, size_t out_bytes,
                           PipelineStats *stats) {
    if (!m_ready || m_stages.empty()) return false;

    PipelineStats local;
    local.stages = m_stages.size();
    local.stage_ns.assign(m_stages.size(), 0);

    const uint64_t t_total0 = now_ns();

    // Seed the first stage's input slot.
    {
        const PipelineStage &st = m_stages.front();
        DmaBuffer *in = m_buf[read_slot(0)][st.die_id].get();
        if (!in) return false;
        size_t n = seed_bytes ? seed_bytes : st.in_bytes;
        if (n > in->size()) n = in->size();
        if (seed && n) {
            std::memcpy(in->data(), seed, n);
            in->flush_cache();
        }
    }

    uint64_t submit_ns = 0;
    uint64_t handoff_ns = 0;

    for (size_t i = 0; i < m_stages.size(); ++i) {
        const PipelineStage &st = m_stages[i];
        DmaBuffer *in  = m_buf[read_slot(i)][st.die_id].get();
        DmaBuffer *ot  = m_buf[read_slot(i) ^ 1u][st.die_id].get();
        if (!in || !ot) return false;

        const uint64_t t0 = now_ns();
        bool ok = true;

        if (m_hardware && st.model_id != 0) {
            std::vector<DmaBuffer *> ins{in};
            std::vector<DmaBuffer *> outs{ot};
            uint64_t elapsed = 0;
            ok = m_pool.submit_task(st.die_id, st.model_id, ins, outs, &elapsed);
            submit_ns += elapsed;
        } else {
            // Off-board: WorkerPool::submit_task succeeds without computing anything,
            // so run the caller's stand-in transform to keep chaining observable.
            size_t n_floats = st.out_bytes / sizeof(float);
            if (n_floats * sizeof(float) > ot->size()) {
                n_floats = ot->size() / sizeof(float);
            }
            if (m_host_stage_fn) {
                ok = m_host_stage_fn(static_cast<const float *>(in->data()),
                                     static_cast<float *>(ot->data()), n_floats);
            } else if (n_floats) {
                std::memcpy(ot->data(), in->data(), n_floats * sizeof(float));
            }
            ot->flush_cache();
        }

        local.stage_ns[i] = now_ns() - t0;
        if (!ok) return false;

        // Hand the activation to the next die when the chain crosses a die boundary.
        if (i + 1 < m_stages.size()) {
            const PipelineStage &nxt = m_stages[i + 1];
            if (nxt.die_id != st.die_id) {
                DmaBuffer *dst = m_buf[read_slot(i + 1)][nxt.die_id].get();
                if (!dst) return false;
                size_t bytes = st.out_bytes ? st.out_bytes : nxt.in_bytes;
                uint64_t elapsed = 0;
                if (!copy_across_dies(*ot, st.die_id, *dst, nxt.die_id, bytes, &elapsed)) {
                    return false;
                }
                handoff_ns += elapsed;
                ++local.cross_die_handoffs;
            }
        }
    }

    // Final activation back to the caller.
    if (out && out_bytes) {
        const PipelineStage &last = m_stages.back();
        DmaBuffer *src = m_buf[read_slot(m_stages.size() - 1) ^ 1u][last.die_id].get();
        if (src) {
            size_t n = last.out_bytes ? last.out_bytes : out_bytes;
            if (n > out_bytes) n = out_bytes;
            if (n > src->size()) n = src->size();
            std::memcpy(out, src->data(), n);
        }
    }

    local.total_ns = now_ns() - t_total0;
    local.submit_ns = submit_ns;
    local.handoff_ns = handoff_ns;
    local.avg_stage_us = m_stages.empty()
        ? 0.0
        : static_cast<double>(local.total_ns) / 1000.0 / static_cast<double>(m_stages.size());

    if (stats) *stats = local;
    return true;
}

} // namespace eswin


/* =========================================================================
 * File: worker_pool.cpp
 * ========================================================================= */

/**
 * @file worker_pool.cpp
 * @brief Implementation of Multi-Process Dual-Die Isolation & NUMA Worker Daemon.
 */

#include "worker_pool.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

namespace eswin {

// =============================================================================
// WorkerChannel Implementation
// =============================================================================

WorkerChannel::WorkerChannel(uint16_t die_id)
    : m_die_id(die_id) {}

WorkerChannel::~WorkerChannel() {
    shutdown();
}

bool WorkerChannel::start() {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, fds) != 0) {
        std::cerr << "[WorkerChannel] Failed to create socketpair: " << std::strerror(errno) << std::endl;
        return false;
    }

    m_master_fd = fds[0];
    m_worker_fd = fds[1];

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[WorkerChannel] Failed to fork worker process: " << std::strerror(errno) << std::endl;
        close(m_master_fd);
        close(m_worker_fd);
        m_master_fd = -1;
        m_worker_fd = -1;
        return false;
    }

    if (pid == 0) {
        // Child process (Worker Daemon)
        close(m_master_fd);
        m_master_fd = -1;

        // 1. Set NUMA CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (m_die_id == 0) {
            // NUMA Node 0 (Cores 0-3)
            CPU_SET(0, &cpuset);
            CPU_SET(1, &cpuset);
            CPU_SET(2, &cpuset);
            CPU_SET(3, &cpuset);
        } else {
            // NUMA Node 1 (Cores 4-7)
            CPU_SET(4, &cpuset);
            CPU_SET(5, &cpuset);
            CPU_SET(6, &cpuset);
            CPU_SET(7, &cpuset);
        }
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // 2. Initialize Hardware Subsystem for this die
        auto &driver = EswinDriver::instance();
        bool hw = driver.initialize();
        if (hw) {
            driver.set_device(m_die_id);
        }

        // 3. Send initial Handshake Ready message to master
        IpcResponse ready_resp{};
        ready_resp.status = IpcStatus::Success;
        ready_resp.seq_id = 0;
        send(m_worker_fd, &ready_resp, sizeof(ready_resp), 0);

        // 4. Enter worker event loop
        run_worker_loop();

        if (hw) {
            driver.release_device(m_die_id);
            driver.shutdown();
        }

        close(m_worker_fd);
        _exit(0);
    }

    // Parent process (Master)
    close(m_worker_fd);
    m_worker_fd = -1;
    m_pid = pid;

    // Explicitly allow child worker processes to import DMA buffer handles via pidfd
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);

    // Wait for handshake with timeout (5000 ms)
    struct pollfd pfd{};
    pfd.fd = m_master_fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 5000);
    if (ret <= 0) {
        std::cerr << "[WorkerChannel] Timeout waiting for worker " << m_die_id << " to initialize." << std::endl;
        shutdown();
        return false;
    }

    IpcResponse handshake{};
    ssize_t n = recv(m_master_fd, &handshake, sizeof(handshake), 0);
    if (n != sizeof(handshake) || handshake.status != IpcStatus::Success) {
        std::cerr << "[WorkerChannel] Invalid handshake from worker " << m_die_id << std::endl;
        shutdown();
        return false;
    }

    m_is_alive = true;
    return true;
}

void WorkerChannel::shutdown() {
    if (!m_is_alive && m_pid <= 0) return;

    if (m_master_fd >= 0) {
        IpcRequest req{};
        req.cmd = IpcCmd::Shutdown;
        req.seq_id = ++m_seq_counter;
        send(m_master_fd, &req, sizeof(req), MSG_NOSIGNAL);

        // Read acknowledge with short deadline
        struct pollfd pfd{};
        pfd.fd = m_master_fd;
        pfd.events = POLLIN;
        if (poll(&pfd, 1, 1000) > 0) {
            IpcResponse resp{};
            recv(m_master_fd, &resp, sizeof(resp), 0);
        }

        close(m_master_fd);
        m_master_fd = -1;
    }

    if (m_pid > 0) {
        int status = 0;
        // Wait up to 1 second for graceful exit
        for (int i = 0; i < 10; ++i) {
            pid_t r = waitpid(m_pid, &status, WNOHANG);
            if (r == m_pid) {
                m_pid = -1;
                break;
            }
            usleep(100000);
        }

        if (m_pid > 0) {
            // Force terminate if hung
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
            m_pid = -1;
        }
    }

    m_is_alive = false;
}

bool WorkerChannel::send_request(const IpcRequest &req, IpcResponse &resp) {
    if (!m_is_alive || m_master_fd < 0) return false;

    ssize_t sent = send(m_master_fd, &req, sizeof(req), MSG_NOSIGNAL);
    if (sent != sizeof(req)) {
        m_is_alive = false;
        return false;
    }

    // Wait for response (up to 30 second deadline)
    struct pollfd pfd{};
    pfd.fd = m_master_fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 30000);
    if (ret <= 0) {
        std::cerr << "[WorkerChannel] Timeout awaiting response from die " << m_die_id << std::endl;
        m_is_alive = false;
        return false;
    }

    ssize_t n = recv(m_master_fd, &resp, sizeof(resp), 0);
    if (n != sizeof(resp)) {
        m_is_alive = false;
        return false;
    }

    return (resp.status == IpcStatus::Success);
}

void WorkerChannel::run_worker_loop() {
    auto &driver = EswinDriver::instance();
    bool is_hw = driver.is_hardware_available();
    std::unordered_map<uint64_t, uint64_t> rcvd_fds;

    while (true) {
        IpcRequest req{};
        ssize_t n = recv(m_worker_fd, &req, sizeof(req), 0);
        if (n <= 0) {
            // Socket closed or EOF
            break;
        }

        IpcResponse resp{};
        resp.seq_id = req.seq_id;
        resp.status = IpcStatus::Success;

        switch (req.cmd) {
            case IpcCmd::Ping: {
                resp.status = IpcStatus::Success;
                break;
            }

            case IpcCmd::LoadModel: {
                if (is_hw) {
                    driver.set_device(m_die_id);
                    uint32_t model_id = 0;
                    int ret = driver.load_model(&model_id, req.model_path);
                    if (ret == 0) {
                        resp.model_id = model_id;
                        uint32_t num_in = 0, num_out = 0;
                        driver.get_num_input_tensors(model_id, &num_in);
                        driver.get_num_output_tensors(model_id, &num_out);
                        resp.num_inputs = static_cast<uint8_t>(num_in);
                        resp.num_outputs = static_cast<uint8_t>(num_out);

                        for (uint32_t i = 0; i < num_in && i < 10; ++i) {
                            NpuTensorDesc desc{};
                            driver.get_input_tensor_desc(model_id, i, &desc);
                            resp.input_sizes[i] = desc.buffer_size;
                        }

                        for (uint32_t i = 0; i < num_out && i < 10; ++i) {
                            NpuTensorDesc desc{};
                            driver.get_output_tensor_desc(model_id, i, &desc);
                            resp.output_sizes[i] = desc.buffer_size;
                        }

                        resp.status = IpcStatus::Success;
                    } else {
                        resp.status = IpcStatus::ErrorExecutionFailed;
                        resp.driver_error = ret;
                    }
                } else {
                    // Host fallback
                    resp.model_id = 100 + m_die_id;
                    resp.num_inputs = 1;
                    resp.num_outputs = 1;
                    resp.input_sizes[0] = 4096;
                    resp.output_sizes[0] = 4096;
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::UnloadModel: {
                if (is_hw) {
                    driver.set_device(m_die_id);
                    int ret = driver.unload_model(req.model_id);
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::PrepareBuffer: {
                if (is_hw) {
                    driver.set_device(m_die_id);

                    uint64_t snd_fd = req.inputs[0].mem_fd;
                    uint64_t rcvd_fd = snd_fd;

                    pid_t snd_pid = static_cast<pid_t>(snd_fd >> 32);
                    if (snd_pid != 0 && snd_pid != getpid()) {
                        int r = driver.mem_rcv_fd(snd_fd, &rcvd_fd);
                        if (r != 0) {
                            std::cerr << "[Worker " << m_die_id << "] mem_rcv_fd failed for snd_fd 0x"
                                      << std::hex << snd_fd << std::dec << ": " << driver.decode_error(r) << std::endl;
                            resp.status = IpcStatus::ErrorExecutionFailed;
                            resp.driver_error = r;
                            break;
                        }
                    }

                    rcvd_fds[snd_fd] = rcvd_fd;

                    DevBuf dev_buf{};
                    dev_buf.mem_fd = rcvd_fd;
                    dev_buf.offset = req.inputs[0].offset;
                    dev_buf.size = req.inputs[0].size;
                    int ret = driver.prepare_dev_buf(&dev_buf);
                    if (ret != 0) {
                        std::cerr << "[Worker " << m_die_id << "] prepare_dev_buf failed on rcvd_fd 0x"
                                  << std::hex << rcvd_fd << std::dec << ": " << driver.decode_error(ret) << std::endl;
                    }
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::UnprepareBuffer: {
                if (is_hw) {
                    driver.set_device(m_die_id);

                    uint64_t snd_fd = req.inputs[0].mem_fd;
                    uint64_t rcvd_fd = snd_fd;
                    auto it = rcvd_fds.find(snd_fd);
                    if (it != rcvd_fds.end()) {
                        rcvd_fd = it->second;
                    }

                    DevBuf dev_buf{};
                    dev_buf.mem_fd = rcvd_fd;
                    dev_buf.offset = req.inputs[0].offset;
                    dev_buf.size = req.inputs[0].size;
                    int ret = driver.unprepare_dev_buf(&dev_buf);

                    if (it != rcvd_fds.end() && it->second != snd_fd) {
                        driver.mem_free(it->second);
                        rcvd_fds.erase(it);
                    }

                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::SubmitTask: {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (is_hw) {
                    driver.set_device(m_die_id);

                    NpuTask task{};
                    task.modelId = req.model_id;
                    task.inputFdNum = req.input_count;
                    task.outputFdNum = req.output_count;

                    for (uint8_t i = 0; i < req.input_count && i < 10; ++i) {
                        uint64_t snd_fd = req.inputs[i].mem_fd;
                        auto it = rcvd_fds.find(snd_fd);
                        task.inputFd[i].mem_fd = (it != rcvd_fds.end()) ? it->second : snd_fd;
                        task.inputFd[i].offset = req.inputs[i].offset;
                        task.inputFd[i].size   = req.inputs[i].size;
                        task.inputFd[i].reserved = 0;
                    }

                    for (uint8_t i = 0; i < req.output_count && i < 10; ++i) {
                        uint64_t snd_fd = req.outputs[i].mem_fd;
                        auto it = rcvd_fds.find(snd_fd);
                        task.outputFd[i].mem_fd = (it != rcvd_fds.end()) ? it->second : snd_fd;
                        task.outputFd[i].offset = req.outputs[i].offset;
                        task.outputFd[i].size   = req.outputs[i].size;
                        task.outputFd[i].reserved = 0;
                    }

                    int ret = driver.submit_task(&task, 1);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    resp.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    // Host fallback simulation
                    auto t1 = std::chrono::high_resolution_clock::now();
                    resp.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::Shutdown: {
                if (is_hw) {
                    for (auto &pair : rcvd_fds) {
                        if (pair.second != pair.first) {
                            driver.mem_free(pair.second);
                        }
                    }
                    rcvd_fds.clear();
                }
                resp.status = IpcStatus::Success;
                send(m_worker_fd, &resp, sizeof(resp), 0);
                return;
            }

            default: {
                resp.status = IpcStatus::ErrorInvalidArg;
                break;
            }
        }

        send(m_worker_fd, &resp, sizeof(resp), 0);
    }
}

// =============================================================================
// WorkerPool Implementation
// =============================================================================

WorkerPool::WorkerPool() = default;

WorkerPool::~WorkerPool() {
    shutdown();
}

bool WorkerPool::start(uint16_t num_dies) {
    if (m_running) return true;

    auto &driver = EswinDriver::instance();
    driver.initialize();

    uint16_t available = driver.get_num_devices();
    if (available > 0 && num_dies > available) {
        num_dies = available;
    }
    if (num_dies == 0) num_dies = 1;

    m_channels.clear();
    for (uint16_t i = 0; i < num_dies; ++i) {
        auto ch = std::make_unique<WorkerChannel>(i);
        if (!ch->start()) {
            std::cerr << "[WorkerPool] Failed to launch worker daemon for die " << i << std::endl;
            shutdown();
            return false;
        }
        m_channels.push_back(std::move(ch));
    }

    m_running = true;
    return true;
}

void WorkerPool::shutdown() {
    for (auto &ch : m_channels) {
        if (ch) ch->shutdown();
    }
    m_channels.clear();
    m_running = false;
}

bool WorkerPool::load_model(uint16_t die_id, const std::string &path, ModelInfo &out_info) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::LoadModel;
    std::strncpy(req.model_path, path.c_str(), sizeof(req.model_path) - 1);

    IpcResponse resp{};
    if (m_channels[die_id]->send_request(req, resp)) {
        out_info.model_id = resp.model_id;
        out_info.num_inputs = resp.num_inputs;
        out_info.num_outputs = resp.num_outputs;
        for (int i = 0; i < 10; ++i) {
            out_info.input_sizes[i] = resp.input_sizes[i];
            out_info.output_sizes[i] = resp.output_sizes[i];
        }
        return true;
    }
    return false;
}

bool WorkerPool::load_model(uint16_t die_id, const std::string &path, uint32_t &out_model_id) {
    ModelInfo info;
    if (load_model(die_id, path, info)) {
        out_model_id = info.model_id;
        return true;
    }
    return false;
}

bool WorkerPool::unload_model(uint16_t die_id, uint32_t model_id) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::UnloadModel;
    req.model_id = model_id;

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

bool WorkerPool::prepare_buffer(uint16_t die_id, const DmaBuffer &buf) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::PrepareBuffer;
    req.inputs[0].mem_fd = buf.mem_fd();
    req.inputs[0].offset = 0;
    req.inputs[0].size   = buf.size();

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

bool WorkerPool::unprepare_buffer(uint16_t die_id, const DmaBuffer &buf) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::UnprepareBuffer;
    req.inputs[0].mem_fd = buf.mem_fd();
    req.inputs[0].offset = 0;
    req.inputs[0].size   = buf.size();

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

const char *describe_slice_error(SliceError err) {
    switch (err) {
        case SliceError::Success:           return "ok";
        case SliceError::NullBuffer:        return "slice has no buffer";
        case SliceError::TooManyBuffers:    return "more slices than the task has buffer slots";
        case SliceError::UnalignedOffset:   return "slice offset is not page-aligned";
        case SliceError::RangeExceedsBuffer:return "slice extends past the end of its buffer";
        case SliceError::EmptySlice:        return "slice length is zero";
    }
    return "unknown slice error";
}

SliceError fill_ipc_buffers(const std::vector<DmaBufferSlice> &slices,
                           IpcBufferDesc *out, size_t cap, size_t &out_count) {
    static_assert(sizeof(size_t) <= sizeof(uint64_t),
                  "slice offset/size must fit the uint64_t IPC fields");
    out_count = 0;
    if (slices.size() > cap) {
        out_count = cap;
        return SliceError::TooManyBuffers;
    }

    for (size_t i = 0; i < slices.size(); ++i) {
        const DmaBufferSlice &s = slices[i];
        if (!s.buffer) {
            out_count = i;
            return SliceError::NullBuffer;
        }

        const size_t total = s.buffer->size();
        if (s.offset > total) {
            out_count = i;
            return SliceError::RangeExceedsBuffer;
        }
        if (s.offset % kSliceAlignment != 0) {
            out_count = i;
            return SliceError::UnalignedOffset;
        }

        const size_t length = s.size ? s.size : (total - s.offset);
        if (length == 0) {
            out_count = i;
            return SliceError::EmptySlice;
        }
        // Compared against the remaining bytes, never as offset+length, so a huge length
        // cannot wrap around the check.
        if (length > total - s.offset) {
            out_count = i;
            return SliceError::RangeExceedsBuffer;
        }

        out[i].mem_fd   = s.buffer->mem_fd();
        out[i].offset   = s.offset;
        out[i].size     = length;
        out[i].reserved = 0;
        out_count = i + 1;
    }

    return SliceError::Success;
}

bool WorkerPool::submit_task(
    uint16_t die_id,
    uint32_t model_id,
    const std::vector<DmaBufferSlice> &inputs,
    const std::vector<DmaBufferSlice> &outputs,
    uint64_t *out_elapsed_ns
) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::SubmitTask;
    req.model_id = model_id;

    size_t n_inputs = 0, n_outputs = 0;
    bool failed_in_inputs = true;
    SliceError err = fill_ipc_buffers(inputs, req.inputs, kIpcBufferSlots, n_inputs);
    if (err == SliceError::Success) {
        err = fill_ipc_buffers(outputs, req.outputs, kIpcBufferSlots, n_outputs);
        failed_in_inputs = false;
    }
    if (err != SliceError::Success) {
        const size_t slot = failed_in_inputs ? n_inputs : n_outputs;
        std::cerr << "[Master] submit_task rejected (die " << die_id << ", model " << model_id
                  << "): " << describe_slice_error(err) << " at "
                  << (failed_in_inputs ? "input[" : "output[") << slot << "]" << std::endl;
        return false;
    }

    req.input_count = static_cast<uint8_t>(n_inputs);
    req.output_count = static_cast<uint8_t>(n_outputs);

    IpcResponse resp{};
    if (m_channels[die_id]->send_request(req, resp)) {
        if (out_elapsed_ns) {
            *out_elapsed_ns = resp.elapsed_ns;
        }
        return true;
    }
    return false;
}

bool WorkerPool::submit_task(
    uint16_t die_id,
    uint32_t model_id,
    const std::vector<DmaBuffer*> &inputs,
    const std::vector<DmaBuffer*> &outputs,
    uint64_t *out_elapsed_ns
) {
    std::vector<DmaBufferSlice> in_slices, out_slices;
    in_slices.reserve(inputs.size());
    out_slices.reserve(outputs.size());
    for (const DmaBuffer *b : inputs) in_slices.push_back(DmaBufferSlice{b});
    for (const DmaBuffer *b : outputs) out_slices.push_back(DmaBufferSlice{b});
    return submit_task(die_id, model_id, in_slices, out_slices, out_elapsed_ns);
}

} // namespace eswin


/* =========================================================================
 * File: hardware_kv_cache.cpp
 * ========================================================================= */

/**
 * @file hardware_kv_cache.cpp
 * @brief Physically contiguous MMZ KV-cache with per-package layout (Milestone 9).
 *
 * See hardware_kv_cache.hpp for the two supported layouts and the evidence behind them.
 *
 * Buffers are CacheMode::Cached even though the NPU is the primary writer: the CPU reads
 * them back for golden-data verification and in host fallback, and NoCache MMZ reads
 * collapse to 10-60 MB/s versus 1.74-2.89 GB/s cached. Every CPU write is followed by
 * flush_cache() so the NPU sees it.
 */

#include "hardware_kv_cache.hpp"
#include "../model/embedding_table.hpp"   // es_f32_to_f16 / es_f16_to_f32

#include <cstring>
#include <sstream>

namespace eswin {

namespace {

const char *zone_for_die(uint16_t die) {
    return (die == 1) ? "mmz_nid_1_part_0" : "mmz_nid_0_part_0";
}

} // namespace

uint32_t HardwareKvCache::default_token_n2(uint32_t max_seq_len) noexcept {
    static const uint32_t kCandidates[] = {256, 128, 64, 32, 16, 8, 4, 2, 1};
    for (uint32_t c : kCandidates) {
        if (max_seq_len % c == 0) return c;
    }
    return 1;
}

HardwareKvCache::~HardwareKvCache() {
    release();
}

size_t HardwareKvCache::bytes_per_layer() const noexcept {
    return kv_dim() * static_cast<size_t>(m_cfg.max_seq_len) * m_cfg.elem_bytes;
}

uint16_t HardwareKvCache::die_for_layer(uint32_t layer) const noexcept {
    return (layer < m_layer_die.size()) ? m_layer_die[layer] : 0;
}

bool HardwareKvCache::allocate(const HardwareKvConfig &cfg, WorkerPool *pool, std::string &err) {
    release();
    err.clear();

    if (cfg.num_layers == 0 || cfg.num_kv_heads == 0 || cfg.head_dim == 0 ||
        cfg.max_seq_len == 0) {
        err = "HardwareKvCache: num_layers, num_kv_heads, head_dim and max_seq_len "
              "must all be non-zero";
        return false;
    }
    if (cfg.elem_bytes != 2 && cfg.elem_bytes != 4) {
        err = "HardwareKvCache: elem_bytes must be 2 (fp16) or 4 (fp32), got " +
              std::to_string(cfg.elem_bytes);
        return false;
    }

    m_cfg = cfg;
    if (cfg.layout == KvLayout::TokenMajorTiled) {
        // The tile height is a compile-time property of the loadable, published by the
        // package as token_n2 (16 for Qwen2-0.5B 1k INT8). A divisor heuristic picks 256
        // there, and a wrong tile height mis-addresses every V write without crashing -
        // it just produces Issue #47-style corruption. Refuse to guess.
        if (cfg.token_n2 == 0) {
            err = "HardwareKvCache: TokenMajorTiled requires an explicit token_n2 taken "
                  "from the package (ModelConfig::kv_token_n2); refusing to guess a "
                  "V-cache tile height";
            return false;
        }
        m_token_n2 = cfg.token_n2;
    } else {
        m_token_n2 = cfg.token_n2 ? cfg.token_n2 : default_token_n2(cfg.max_seq_len);
    }
    if (m_token_n2 == 0 || cfg.max_seq_len % m_token_n2 != 0) {
        err = "HardwareKvCache: token_n2 (" + std::to_string(m_token_n2) +
              ") must divide max_seq_len (" + std::to_string(cfg.max_seq_len) + ")";
        return false;
    }

    const size_t per_layer = bytes_per_layer();
    if (per_layer == 0) {
        err = "HardwareKvCache: computed a zero-byte layer cache";
        return false;
    }

    m_pool = pool;
    m_layer_die.assign(cfg.num_layers, 0);
    m_k.reserve(cfg.num_layers);
    m_v.reserve(cfg.num_layers);

    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        // Pipeline-parallel split: first half of the layers on Die 0, rest on Die 1.
        const uint16_t die = (cfg.dual_die && layer >= cfg.num_layers / 2) ? 1 : 0;
        m_layer_die[layer] = die;

        auto k = std::make_unique<DmaBuffer>(per_layer, CacheMode::Cached, zone_for_die(die));
        auto v = std::make_unique<DmaBuffer>(per_layer, CacheMode::Cached, zone_for_die(die));
        if (!k->is_valid() || !v->is_valid()) {
            std::ostringstream os;
            os << "HardwareKvCache: allocation failed at layer " << layer << " ("
               << per_layer << " bytes per cache)";
            err = os.str();
            release();
            return false;
        }

        if (pool) {
            // Must be prepared inside the worker that submits, or ES_NPU_Prepare aborts
            // with 0xA00F604A (no thread-local device context).
            if (!pool->prepare_buffer(die, *k) || !pool->prepare_buffer(die, *v)) {
                std::ostringstream os;
                os << "HardwareKvCache: SMMU prepare failed for layer " << layer
                   << " on die " << die;
                err = os.str();
                release();
                return false;
            }
        }

        m_k.push_back(std::move(k));
        m_v.push_back(std::move(v));
        m_total_bytes += 2 * per_layer;
    }

    m_length = 0;
    return true;
}

void HardwareKvCache::release() {
    for (size_t i = 0; i < m_k.size(); ++i) {
        const uint16_t die = (i < m_layer_die.size()) ? m_layer_die[i] : 0;
        if (m_pool) {
            if (m_k[i]) m_pool->unprepare_buffer(die, *m_k[i]);
            if (i < m_v.size() && m_v[i]) m_pool->unprepare_buffer(die, *m_v[i]);
        }
    }
    m_k.clear();
    m_v.clear();
    m_layer_die.clear();
    m_length = 0;
    m_total_bytes = 0;
    m_pool = nullptr;
}

uint32_t HardwareKvCache::slot_for(uint32_t token) const noexcept {
    if (m_cfg.max_seq_len == 0) return 0;
    return m_cfg.wrap ? (token % m_cfg.max_seq_len) : token;
}

size_t HardwareKvCache::k_offset(uint32_t head, uint32_t dim, uint32_t token) const noexcept {
    const size_t t = slot_for(token);
    const size_t e = m_cfg.elem_bytes;
    if (m_cfg.layout == KvLayout::HeadMajor) {
        // [head][dim][token]: token is the contiguous axis, as in (2, 64, 1, 1024).
        return ((static_cast<size_t>(head) * m_cfg.head_dim + dim) *
                    static_cast<size_t>(m_cfg.max_seq_len) + t) * e;
    }
    // [token][kv_dim]: the FINDINGS section 5 K formula, elem * kv_dim * token.
    return (t * kv_dim() + static_cast<size_t>(head) * m_cfg.head_dim + dim) * e;
}

size_t HardwareKvCache::v_offset(uint32_t head, uint32_t token, uint32_t dim) const noexcept {
    const size_t t = slot_for(token);
    const size_t e = m_cfg.elem_bytes;
    if (m_cfg.layout == KvLayout::HeadMajor) {
        // [head][token][dim]: dim is the contiguous axis, as in (2, 1024, 1, 64).
        return ((static_cast<size_t>(head) * m_cfg.max_seq_len + t) * m_cfg.head_dim + dim) * e;
    }
    // Tiled: FINDINGS section 5 V formula, extended with the per-head and per-dim terms.
    const size_t n2 = m_token_n2;
    const size_t tiles = m_cfg.max_seq_len / n2;
    const size_t head_stride = static_cast<size_t>(m_cfg.head_dim) * tiles * n2;
    return (static_cast<size_t>(head) * head_stride +
            n2 * static_cast<size_t>(m_cfg.head_dim) * (t / n2) +
            static_cast<size_t>(dim) * n2 +
            (t % n2)) * e;
}

size_t HardwareKvCache::k_token_offset(uint32_t token) const noexcept {
    if (m_cfg.layout == KvLayout::TokenMajorTiled) {
        // Vendor K formula verbatim: precision * kv_dim * token_idx.
        return m_cfg.elem_bytes * kv_dim() * slot_for(token);
    }
    return k_offset(0, 0, token);
}

size_t HardwareKvCache::v_token_offset(uint32_t token) const noexcept {
    const size_t t = slot_for(token);
    if (m_cfg.layout == KvLayout::TokenMajorTiled) {
        // Vendor V formula verbatim:
        // precision * (token_n2 * head_dim * (t / token_n2) + (t % token_n2)).
        return m_cfg.elem_bytes * (static_cast<size_t>(m_token_n2) * m_cfg.head_dim *
                                       (t / m_token_n2) + (t % m_token_n2));
    }
    return v_offset(0, static_cast<uint32_t>(t), 0);
}

bool HardwareKvCache::advance(uint32_t n) {
    if (!is_allocated()) return false;
    if (!m_cfg.wrap && m_length + n > m_cfg.max_seq_len) {
        // Strict mode: refuse rather than write past the compiled window.
        return false;
    }
    const uint64_t next = static_cast<uint64_t>(m_length) + n;
    m_length = (next > m_cfg.max_seq_len) ? m_cfg.max_seq_len : static_cast<uint32_t>(next);
    return true;
}

void HardwareKvCache::reset(bool zero_fill) {
    m_length = 0;
    if (!zero_fill) return;
    for (size_t i = 0; i < m_k.size(); ++i) {
        if (m_k[i] && m_k[i]->is_valid()) {
            std::memset(m_k[i]->data(), 0, m_k[i]->size());
            m_k[i]->flush_cache();
        }
        if (i < m_v.size() && m_v[i] && m_v[i]->is_valid()) {
            std::memset(m_v[i]->data(), 0, m_v[i]->size());
            m_v[i]->flush_cache();
        }
    }
}

DmaBuffer *HardwareKvCache::k_buffer(uint32_t layer) {
    return (layer < m_k.size()) ? m_k[layer].get() : nullptr;
}

DmaBuffer *HardwareKvCache::v_buffer(uint32_t layer) {
    return (layer < m_v.size()) ? m_v[layer].get() : nullptr;
}

const DmaBuffer *HardwareKvCache::k_buffer(uint32_t layer) const {
    return (layer < m_k.size()) ? m_k[layer].get() : nullptr;
}

const DmaBuffer *HardwareKvCache::v_buffer(uint32_t layer) const {
    return (layer < m_v.size()) ? m_v[layer].get() : nullptr;
}

uint64_t HardwareKvCache::k_mem_fd(uint32_t layer) const {
    const DmaBuffer *b = k_buffer(layer);
    return b ? b->mem_fd() : 0;
}

uint64_t HardwareKvCache::v_mem_fd(uint32_t layer) const {
    const DmaBuffer *b = v_buffer(layer);
    return b ? b->mem_fd() : 0;
}

const void *HardwareKvCache::k_elem(uint32_t layer, uint32_t head, uint32_t dim,
                                    uint32_t token) const {
    const DmaBuffer *b = k_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = k_offset(head, dim, token);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<const char *>(b->data()) + off;
}

const void *HardwareKvCache::v_elem(uint32_t layer, uint32_t head, uint32_t token,
                                    uint32_t dim) const {
    const DmaBuffer *b = v_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = v_offset(head, token, dim);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<const char *>(b->data()) + off;
}

void *HardwareKvCache::k_elem_mut(uint32_t layer, uint32_t head, uint32_t dim,
                                  uint32_t token) {
    DmaBuffer *b = k_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = k_offset(head, dim, token);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<char *>(b->data()) + off;
}

void *HardwareKvCache::v_elem_mut(uint32_t layer, uint32_t head, uint32_t token,
                                  uint32_t dim) {
    DmaBuffer *b = v_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = v_offset(head, token, dim);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<char *>(b->data()) + off;
}

namespace {

void store_elem(void *dst, float value, size_t elem_bytes) {
    if (elem_bytes == 4) {
        std::memcpy(dst, &value, sizeof(float));
    } else {
        const uint16_t h = es_f32_to_f16(value);
        std::memcpy(dst, &h, sizeof(h));
    }
}

float load_elem(const void *src, size_t elem_bytes) {
    if (elem_bytes == 4) {
        float f;
        std::memcpy(&f, src, sizeof(f));
        return f;
    }
    uint16_t h;
    std::memcpy(&h, src, sizeof(h));
    return es_f16_to_f32(h);
}

} // namespace

bool HardwareKvCache::write_k(uint32_t layer, uint32_t token, const float *values) {
    if (!values || layer >= m_k.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            void *p = k_elem_mut(layer, h, d, token);
            if (!p) return false;
            store_elem(p, values[static_cast<size_t>(h) * m_cfg.head_dim + d], m_cfg.elem_bytes);
        }
    }
    if (m_k[layer]) m_k[layer]->flush_cache();
    return true;
}

bool HardwareKvCache::write_v(uint32_t layer, uint32_t token, const float *values) {
    if (!values || layer >= m_v.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            void *p = v_elem_mut(layer, h, token, d);
            if (!p) return false;
            store_elem(p, values[static_cast<size_t>(h) * m_cfg.head_dim + d], m_cfg.elem_bytes);
        }
    }
    if (m_v[layer]) m_v[layer]->flush_cache();
    return true;
}

bool HardwareKvCache::read_k(uint32_t layer, uint32_t token, float *out) const {
    if (!out || layer >= m_k.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            const void *p = k_elem(layer, h, d, token);
            if (!p) return false;
            out[static_cast<size_t>(h) * m_cfg.head_dim + d] = load_elem(p, m_cfg.elem_bytes);
        }
    }
    return true;
}

bool HardwareKvCache::read_v(uint32_t layer, uint32_t token, float *out) const {
    if (!out || layer >= m_v.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            const void *p = v_elem(layer, h, token, d);
            if (!p) return false;
            out[static_cast<size_t>(h) * m_cfg.head_dim + d] = load_elem(p, m_cfg.elem_bytes);
        }
    }
    return true;
}

} // namespace eswin


